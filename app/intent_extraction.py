"""
intent_extraction.py — Single structured LLM call that reads the full
conversation history and returns intent + all slots (§5).

Scope/injection detection is folded into this same call
(out_of_scope is one of the intent values).
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.config import PROMPTS_DIR
from app.llm_client import call_llm

logger = logging.getLogger(__name__)

# Load extraction prompt template once at import
_EXTRACTION_TEMPLATE = (PROMPTS_DIR / "extraction_prompt.txt").read_text(encoding="utf-8")


def _format_conversation(messages: list[dict]) -> str:
    """Format message list into readable conversation string."""
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_intent_and_slots(messages: list[dict]) -> dict[str, Any]:
    """
    Given the full conversation history (list of {role, content} dicts),
    return a slots dict with intent + all extracted fields.

    Returns a safe default (clarify_needed with empty slots) on failure.
    """
    conversation_text = _format_conversation(messages)
    # Use simple replacement instead of .format() to avoid KeyError on JSON
    # curly braces {  }  that appear in the prompt template's JSON examples.
    user_message = _EXTRACTION_TEMPLATE.replace(
        "{conversation_history}", conversation_text
    )

    # Read system prompt for grounding context
    system_prompt_path = PROMPTS_DIR / "system_prompt.txt"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")

    raw = call_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=0.1,    # low temp for deterministic structured output
        max_tokens=512,
    )

    slots = _validate_slots(raw)
    logger.info("Extracted intent: %s | missing_info: %s", slots["intent"], slots["missing_info"])
    return slots


def _validate_slots(raw: dict) -> dict[str, Any]:
    """
    Validate and normalise the LLM-returned slots dict.
    Fill in safe defaults for any missing/invalid fields.
    """
    VALID_INTENTS = {
        "clarify_needed", "recommend_new", "refine",
        "compare", "meta_question_on_list", "out_of_scope",
    }
    VALID_OOS_REASONS = {
        "legal_advice", "general_hiring_advice", "prompt_injection", "unrelated_topic", None,
    }

    intent = raw.get("intent", "clarify_needed")
    if intent not in VALID_INTENTS:
        logger.warning("Invalid intent '%s', defaulting to clarify_needed", intent)
        intent = "clarify_needed"

    oos_reason = raw.get("out_of_scope_reason")
    if oos_reason not in VALID_OOS_REASONS:
        oos_reason = None

    return {
        "intent":              intent,
        "role_context":        str(raw.get("role_context") or ""),
        "required_skills":     _ensure_list(raw.get("required_skills")),
        "excluded_skills":     _ensure_list(raw.get("excluded_skills")),
        "test_type_filter":    _ensure_list(raw.get("test_type_filter")) or None,
        "language_filter":     raw.get("language_filter") or None,
        "job_level_filter":    _ensure_list(raw.get("job_level_filter")) or None,
        "compare_targets":     _ensure_list(raw.get("compare_targets")) or None,
        "missing_info":        _ensure_list(raw.get("missing_info")),
        "out_of_scope_reason": oos_reason,
    }


def _ensure_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    if val is None:
        return []
    return [val]
