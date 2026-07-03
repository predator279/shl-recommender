"""
generation.py — Second LLM call: generate the agent reply + shortlist
given the retrieved catalog context (§9, step 3).

Used for: recommend_new, refine, compare, meta_question_on_list,
          clarify_needed (simple cases), out_of_scope.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.config import PROMPTS_DIR
from app.llm_client import call_llm

logger = logging.getLogger(__name__)

# Load generation prompt template once at import
_GENERATION_TEMPLATE = (PROMPTS_DIR / "generation_prompt.txt").read_text(encoding="utf-8")


def _format_conversation(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _catalog_items_to_context(items: list[dict]) -> str:
    """Serialize retrieved catalog items to a compact, LLM-readable format."""
    if not items:
        return "No matching items retrieved."
    rows = []
    for i, item in enumerate(items, 1):
        langs = ", ".join(list(item.get("languages", set()))[:4])
        if len(item.get("languages", set())) > 4:
            langs += f" (+{len(item.get('languages', set())) - 4} more)"
        duration = item.get("duration_minutes")
        dur_str = f"{duration} minutes" if duration else "—"
        rows.append(
            f"{i}. Name: {item['name']}\n"
            f"   URL: {item['link']}\n"
            f"   Type codes: {','.join(item.get('test_type_codes', []))}\n"
            f"   Duration: {dur_str}\n"
            f"   Job levels: {', '.join(list(item.get('job_levels', set()))[:4])}\n"
            f"   Languages: {langs or 'unconfirmed'}\n"
            f"   Description: {item.get('description', '')[:300]}"
        )
    return "\n\n".join(rows)


def _shortlist_to_context(shortlist: list[dict]) -> str:
    if not shortlist:
        return "[]"
    return json.dumps(shortlist, indent=2)


def generate_response(
    messages: list[dict],
    slots: dict[str, Any],
    catalog_items: list[dict],
    prior_shortlist: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """
    Generate the agent's reply + shortlist.

    Returns a dict with keys: reply, recommendations (list[dict]), end_of_conversation (bool).
    """
    conversation_text = _format_conversation(messages)
    catalog_text = _catalog_items_to_context(catalog_items)
    prior_text = _shortlist_to_context(prior_shortlist or [])
    slots_text = json.dumps(slots, indent=2)

    # Use chained .replace() instead of .format() so JSON braces in the
    # prompt template are not misinterpreted as format variables.
    user_message = (
        _GENERATION_TEMPLATE
        .replace("{conversation_history}", conversation_text)
        .replace("{intent}", slots["intent"])
        .replace("{slots_json}", slots_text)
        .replace("{catalog_items_json}", catalog_text)
        .replace("{prior_shortlist_json}", prior_text)
    )

    system_prompt_path = PROMPTS_DIR / "system_prompt.txt"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    raw = call_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=0.3,
        max_tokens=1500,
    )

    return _validate_generation_output(raw, slots["intent"], prior_shortlist)


def _validate_generation_output(
    raw: dict,
    intent: str,
    prior_shortlist: Optional[list[dict]],
) -> dict[str, Any]:
    """
    Validate and normalise the generation output.
    Ensures recommendations is always a list, never null.
    """
    reply = str(raw.get("reply") or "I'm sorry, I wasn't able to generate a response. Please try again.")

    recs_raw = raw.get("recommendations")
    if recs_raw is None or not isinstance(recs_raw, list):
        recs_raw = []

    # For meta_question_on_list, if LLM returned [] but we have prior shortlist, re-attach it
    if intent == "meta_question_on_list" and not recs_raw and prior_shortlist:
        recs_raw = prior_shortlist

    end_of_conv = bool(raw.get("end_of_conversation", False))

    return {
        "reply": reply,
        "recommendations": recs_raw,   # will be post-filtered in main.py
        "end_of_conversation": end_of_conv,
    }
