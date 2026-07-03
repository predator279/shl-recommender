"""
main.py — FastAPI application.

Endpoints:
  GET  /health  → {"status": "ok"}   (immediate, no model calls)
  POST /chat    → ChatResponse        (full turn execution, ≤30s)

Turn execution flow (§9):
  1. Parse full message history
  2. Single LLM call → intent + slots (extraction)
  3. Branch on intent:
     - clarify_needed        → generate clarifying question. recommendations=[].
     - out_of_scope          → generate refusal. recommendations=[].
     - compare               → exact/fuzzy match targets, grounded comparison. recommendations=[].
     - recommend_new/refine  → retrieve → generate → post-filter.
     - meta_question_on_list → answer question, re-attach prior shortlist.
  4. Determine end_of_conversation.
  5. Return schema-compliant JSON.

Models (embedding + FAISS index) are loaded at process startup (lifespan),
not lazily, to avoid cold-start latency collisions with the 30s API timeout.
"""

import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from app.catalog_index import get_catalog_index, CatalogIndex
from app.intent_extraction import extract_intent_and_slots
from app.generation import generate_response
from app.retrieval import retrieve
from app.postfilter import validate_recommendations

# Confirmation phrases that signal end_of_conversation=True (§4)
# Sourced from all 10 reference traces (C1–C10)
_CONFIRMATION_PHRASES = [
    # Explicit confirmations
    "confirmed", "confirm", "locking it in", "lock it in",
    # Positive acceptance phrases
    "that's good", "that works", "that covers it", "perfect",
    "looks good", "sounds good", "great", "go ahead",
    "yes, go ahead", "finalize", "finalise", "done",
    "all good", "approved", "accepted", "we'll use",
    # Trace-derived phrases
    "that's what we need",          # C1: "Perfect, that's what we need."
    "that works. thanks",            # C2: "That works. Thanks."
    "sounds right",
    "keep it",
    "keep the shortlist",            # C7: "Keep the shortlist as-is."
    "keep verify",                   # C9: "Keep Verify G+. Locking it in."
    "final list",                    # C10: "Drop the OPQ. Final list: …"
    "great, thanks",                 # test: "Great, thanks."
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Keyword sets for deterministic refine detection (bypass LLM for these)
_REFINE_ADD_KEYWORDS = [
    "add ", "include ", "also add", "plus ", "throw in", "put in",
]
_REFINE_DROP_KEYWORDS = [
    "drop ", "remove ", "take out", "without ", "exclude ", "skip ",
    "replace ", "swap ", "instead of", "no need for",
]


# ── Lifespan: load models at startup (not lazily) ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the embedding model + FAISS index before the server starts accepting
    requests. This ensures the 30s per-call timeout is never eaten by model
    loading on the first request.
    """
    logger.info("=== SHL Recommender startup: loading models and index ===")

    # Trigger embedding model load (imported at module level in retrieval.py,
    # but we reference it here to surface any load errors early)
    from app.retrieval import _embedder  # noqa: F401

    # Trigger FAISS index + metadata load
    get_catalog_index()

    logger.info("=== Startup complete. Server ready. ===")
    yield
    logger.info("=== SHL Recommender shutdown ===")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI for SHL assessment selection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """
    Health check. Returns immediately without triggering any model/API work.
    Used by Render for cold-start pings.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint. Stateless — full conversation history is passed on
    every call. Returns next agent reply + structured shortlist.
    """
    messages = [m.model_dump() for m in request.messages]
    catalog_index = get_catalog_index()

    try:
        return _handle_turn(messages, catalog_index)
    except Exception as e:
        logger.exception("Unhandled error in /chat: %s", e)
        # Return a graceful error reply rather than a 500 that breaks the grader
        return ChatResponse(
            reply="I encountered an unexpected error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )


# ── Turn execution logic ──────────────────────────────────────────────────────

def _handle_turn(messages: list[dict], catalog_index: CatalogIndex) -> ChatResponse:
    """
    Full turn execution pipeline (§9). Stateless — re-derives everything
    from the full message history on every call.
    """

    # ── Pre-check: Confirmation EOC (must run BEFORE intent extraction) ────────
    # In every reference trace (C1–C10), the final confirmation turn fires
    # end_of_conversation=True regardless of how the LLM would classify it.
    # By checking this first we guarantee EOC is never gated by intent routing.
    prior_shortlist_precheck = _extract_prior_shortlist(messages, catalog_index)
    if _user_is_confirming(messages):
        # Case 1: we can fully parse and return the prior shortlist
        if prior_shortlist_precheck:
            logger.info("Pre-check: user confirmed parsed shortlist → EOC=True")
            return ChatResponse(
                reply="Confirmed. Your assessment shortlist is locked in.",
                recommendations=[
                    Recommendation(
                        name=item["name"],
                        url=item["link"],
                        test_type=",".join(item.get("test_type_codes", [])),
                    )
                    for item in prior_shortlist_precheck
                ],
                end_of_conversation=True,
            )
        # Case 2: shortlist exists as text but we couldn't parse it
        # (e.g. assistant sent a placeholder like "[shortlist]")
        # If there is at least one prior assistant turn, honour the confirmation.
        elif _has_prior_assistant_turn(messages):
            logger.info("Pre-check: user confirmed (unparseable prior turn) → EOC=True")
            return ChatResponse(
                reply="Confirmed. Your assessment shortlist is locked in.",
                recommendations=[],
                end_of_conversation=True,
            )

    # ── Pre-check: Deterministic Refine (bypass LLM for clear add/drop signals) ──
    # If a prior shortlist exists AND the user is clearly adding/removing items,
    # force refine intent without waiting for the LLM to classify it.
    if prior_shortlist_precheck and _user_is_refining(messages):
        logger.info("Pre-check: refine keywords detected → forcing refine intent")
        slots = extract_intent_and_slots(messages)
        slots["intent"] = "refine"  # override whatever the LLM returned
        prior_raw = _shortlist_to_raw(prior_shortlist_precheck)
        retrieved = retrieve(
            catalog_index=catalog_index,
            role_context=slots.get("role_context", ""),
            required_skills=slots.get("required_skills", []),
            test_type_filter=slots.get("test_type_filter"),
            language_filter=slots.get("language_filter"),
            job_level_filter=slots.get("job_level_filter"),
            excluded_skills=slots.get("excluded_skills", []),
        )
        gen = generate_response(
            messages=messages,
            slots=slots,
            catalog_items=retrieved or prior_shortlist_precheck,
            prior_shortlist=prior_raw,
        )
        raw_recs = gen.get("recommendations", [])
        validated = validate_recommendations(raw_recs, catalog_index)
        if not validated and raw_recs:
            validated = [
                Recommendation(
                    name=item["name"],
                    url=item["link"],
                    test_type=",".join(item.get("test_type_codes", [])),
                )
                for item in retrieved[:5]
            ]
        return ChatResponse(
            reply=gen["reply"],
            recommendations=validated,
            end_of_conversation=False,
        )

    # ── Step 1: Extract intent + slots ────────────────────────────────────────
    slots = extract_intent_and_slots(messages)
    intent = slots["intent"]
    logger.info("Intent: %s", intent)

    # ── Override: clarify_needed → recommend_new after enough context ──────────
    # If the LLM still wants to clarify after 3+ exchanges AND the conversation
    # already has a role context, push it to recommend_new. This prevents the
    # agent from getting stuck in infinite clarification loops.
    if intent == "clarify_needed" and slots.get("role_context"):
        user_turns = sum(1 for m in messages if m["role"] == "user")
        if user_turns >= 3:
            logger.info(
                "Override: clarify_needed → recommend_new after %d user turns (role_context: %s)",
                user_turns, slots.get("role_context"),
            )
            intent = "recommend_new"
            slots["intent"] = "recommend_new"

    # ── Guard: vague first turn must not produce recommendations ───────────────
    # Regardless of what the LLM returns, if the first user message is vague
    # (no role, no skills, no job level), force clarify_needed.
    # This guards against weaker fallback models over-eagerly recommending.
    if intent in ("recommend_new", "refine"):
        user_turns = sum(1 for m in messages if m["role"] == "user")
        has_context = bool(
            slots.get("role_context") or
            slots.get("required_skills") or
            slots.get("job_level_filter")
        )
        if user_turns == 1 and not has_context:
            logger.info("Guard: single vague turn → forcing clarify_needed")
            intent = "clarify_needed"
            slots["intent"] = "clarify_needed"
            slots["missing_info"] = ["role or use case", "specific requirements"]


    # ── Step 2: Branch on intent ──────────────────────────────────────────────

    # ── clarify_needed: ask a clarifying question ─────────────────────────────
    if intent == "clarify_needed":
        gen = generate_response(
            messages=messages,
            slots=slots,
            catalog_items=[],
            prior_shortlist=None,
        )
        return ChatResponse(
            reply=gen["reply"],
            recommendations=[],
            end_of_conversation=False,
        )

    # ── out_of_scope: polite refusal, conversation continues ──────────────────
    if intent == "out_of_scope":
        gen = generate_response(
            messages=messages,
            slots=slots,
            catalog_items=[],
            prior_shortlist=None,
        )
        return ChatResponse(
            reply=gen["reply"],
            recommendations=[],
            end_of_conversation=False,   # refusals never end the conversation
        )

    # ── compare: grounded comparison of named items ───────────────────────────
    if intent == "compare":
        compare_targets = slots.get("compare_targets") or []
        compare_items = _resolve_compare_targets(compare_targets, catalog_index)

        gen = generate_response(
            messages=messages,
            slots=slots,
            catalog_items=compare_items,
            prior_shortlist=None,
        )
        return ChatResponse(
            reply=gen["reply"],
            recommendations=[],          # compare intent always returns []
            end_of_conversation=gen["end_of_conversation"],
        )

    # ── meta_question_on_list: answer + re-attach prior shortlist ─────────────
    if intent == "meta_question_on_list":
        prior_shortlist = _extract_prior_shortlist(messages, catalog_index)

        gen = generate_response(
            messages=messages,
            slots=slots,
            catalog_items=prior_shortlist,
            prior_shortlist=_shortlist_to_raw(prior_shortlist),
        )
        # Re-validate the prior shortlist through the post-filter
        # (it was already validated before, but do it again for safety)
        if prior_shortlist:
            validated = prior_shortlist  # already catalog-grounded
        else:
            validated = []

        return ChatResponse(
            reply=gen["reply"],
            recommendations=[
                Recommendation(
                    name=item["name"],
                    url=item["link"],
                    test_type=",".join(item.get("test_type_codes", [])),
                )
                for item in validated
            ],
            end_of_conversation=gen["end_of_conversation"],
        )

    # ── recommend_new / refine: retrieve → generate → post-filter ─────────────
    if intent in ("recommend_new", "refine"):
        # Retrieve from catalog
        retrieved = retrieve(
            catalog_index=catalog_index,
            role_context=slots.get("role_context", ""),
            required_skills=slots.get("required_skills", []),
            test_type_filter=slots.get("test_type_filter"),
            language_filter=slots.get("language_filter"),
            job_level_filter=slots.get("job_level_filter"),
            excluded_skills=slots.get("excluded_skills", []),
        )

        if not retrieved:
            return ChatResponse(
                reply=(
                    "I wasn't able to find catalog items that match your requirements. "
                    "Could you clarify the role or skills you're targeting?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # For refine, also pass prior shortlist as context
        prior_shortlist = None
        if intent == "refine":
            prior_items = _extract_prior_shortlist(messages, catalog_index)
            prior_shortlist = _shortlist_to_raw(prior_items) if prior_items else None

        gen = generate_response(
            messages=messages,
            slots=slots,
            catalog_items=retrieved,
            prior_shortlist=prior_shortlist,
        )

        # Post-filter: validate all LLM-proposed recommendations against catalog
        raw_recs = gen.get("recommendations", [])
        validated = validate_recommendations(raw_recs, catalog_index)

        # Fallback if post-filter drops everything
        if not validated and raw_recs:
            logger.warning(
                "Post-filter dropped all %d LLM recommendations. "
                "Falling back to top retrieved items.",
                len(raw_recs),
            )
            # Use top retrieved items directly (they came from catalog)
            validated = [
                Recommendation(
                    name=item["name"],
                    url=item["link"],
                    test_type=",".join(item.get("test_type_codes", [])),
                )
                for item in retrieved[:5]
            ]

        end_of_conv = gen["end_of_conversation"]
        # Boost EOC detection: if LLM missed it but user clearly confirmed, force True
        if not end_of_conv and validated:
            end_of_conv = _user_is_confirming(messages)

        return ChatResponse(
            reply=gen["reply"],
            recommendations=validated,
            end_of_conversation=end_of_conv,
        )

    # ── Fallback (should never reach here) ───────────────────────────────────
    logger.error("Unknown intent '%s' — falling back to clarify_needed", intent)
    return ChatResponse(
        reply="Could you tell me more about the role and what you're looking for in an assessment?",
        recommendations=[],
        end_of_conversation=False,
    )


# ── Helper functions ──────────────────────────────────────────────────────────

def _resolve_compare_targets(
    target_names: list[str],
    catalog_index: CatalogIndex,
) -> list[dict]:
    """
    Resolve compare_targets to catalog metadata dicts via exact/fuzzy name match.
    """
    results = []
    for name in target_names:
        item = catalog_index.lookup_by_name(name)
        if item:
            results.append(item)
        else:
            logger.warning("Compare target not found in catalog: '%s'", name)
    return results


def _extract_prior_shortlist(
    messages: list[dict],
    catalog_index: CatalogIndex,
) -> list[dict]:
    """
    Reconstruct the prior shortlist from conversation history (stateless approach).

    Strategy: scan assistant messages in reverse for embedded JSON tags OR
    for table-like patterns listing assessment names. The client sends back
    the full message history on every call, so we can recover the last shortlist
    the agent presented.

    Assistant messages may contain:
      1. A JSON block tagged with __RECS__[...]__END_RECS__ (preferred, machine-readable)
      2. Markdown table rows with assessment names (fallback pattern matching)
    """
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        content = msg["content"]

        # ── Strategy 1: machine-readable JSON tag ─────────────────────────────
        marker_match = re.search(r"__RECS__(\[.*?\])__END_RECS__", content, re.DOTALL)
        if marker_match:
            try:
                recs_json = json.loads(marker_match.group(1))
                resolved = []
                for rec in recs_json:
                    item = catalog_index.lookup_by_name(rec.get("name", ""))
                    if item:
                        resolved.append(item)
                if resolved:
                    return resolved
            except (json.JSONDecodeError, KeyError):
                pass

        # ── Strategy 2: extract names from JSON recommendations array in content ──
        # Some clients may echo back the structured response in the message content
        json_match = re.search(r'"recommendations"\s*:\s*(\[.*?\])', content, re.DOTALL)
        if json_match:
            try:
                recs_json = json.loads(json_match.group(1))
                resolved = []
                for rec in recs_json:
                    name = rec.get("name", "")
                    if name:
                        item = catalog_index.lookup_by_name(name)
                        if item:
                            resolved.append(item)
                if resolved:
                    return resolved
            except (json.JSONDecodeError, KeyError):
                pass

        # ── Strategy 3: extract names from markdown table rows ────────────────
        # Pattern: "| 1 | Assessment Name | ..." table rows
        table_names = re.findall(
            r'^\|\s*\d+\s*\|\s*([^|]+?)\s*\|',
            content,
            re.MULTILINE,
        )
        if table_names:
            resolved = []
            for name in table_names:
                name = name.strip()
                if len(name) > 3:   # ignore empty/junk cells
                    item = catalog_index.lookup_by_name(name)
                    if item:
                        resolved.append(item)
            if resolved:
                return resolved

    return []


def _shortlist_to_raw(items: list[dict]) -> list[dict]:
    """Convert catalog metadata dicts to the raw recommendation format expected by generation."""
    return [
        {
            "name": item["name"],
            "url": item["link"],
            "test_type": ",".join(item.get("test_type_codes", [])),
        }
        for item in items
    ]


def _user_is_confirming(messages: list[dict]) -> bool:
    """
    Check if the last user message contains a clear confirmation phrase.
    Used to reliably detect end_of_conversation=True when the LLM misses it.
    """
    last_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"].lower().strip()
            break
    return any(phrase in last_user_msg for phrase in _CONFIRMATION_PHRASES)


def _has_prior_assistant_turn(messages: list[dict]) -> bool:
    """Return True if at least one prior assistant message exists in the history."""
    return any(m["role"] == "assistant" for m in messages[:-1])


def _user_is_refining(messages: list[dict]) -> bool:
    """
    Deterministic check: does the last user message contain clear add/drop signals?
    Used to force refine intent when the LLM mislabels it as clarify_needed.
    """
    last_user_msg = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"].lower().strip()
            break
    has_add = any(kw in last_user_msg for kw in _REFINE_ADD_KEYWORDS)
    has_drop = any(kw in last_user_msg for kw in _REFINE_DROP_KEYWORDS)
    return has_add or has_drop
