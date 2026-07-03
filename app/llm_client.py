"""
llm_client.py — Groq primary + Google Gemini fallback LLM client.

Strategy (§10):
- Primary: Groq llama-3.3-70b-versatile in JSON mode.
- Fallback: Google Gemini Flash on Groq 429/timeout/error.
  Retry-with-immediate-fallback (NOT retry-with-backoff) to respect 30s timeout.

Provider-agnostic prompt format — plain JSON schema instructions, no
Groq-specific tool syntax.
"""

import json
import logging
import time
from typing import Any, Optional

import groq
from google import genai
from google.genai import types as genai_types

from app.config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    GROQ_MODEL,
    GEMINI_MODEL,
)

logger = logging.getLogger(__name__)

# ── Initialise clients once ───────────────────────────────────────────────────
_groq_client: Optional[groq.Groq] = None
_gemini_client: Optional[genai.Client] = None


def _get_groq_client() -> groq.Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _groq_client = groq.Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _call_groq(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Call Groq in JSON mode. Returns raw response string."""
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=25,   # leave headroom under 30s API timeout
    )
    return response.choices[0].message.content or "{}"


def _call_gemini(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Call Gemini Flash as fallback using the new google-genai SDK.
    Returns raw response string.
    """
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    return response.text or "{}"


def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """
    Unified LLM call with immediate Groq → Gemini fallback.
    Always returns a parsed dict (falls back to {} on total failure).
    """
    # ── Try Groq first ────────────────────────────────────────────────────────
    try:
        start = time.monotonic()
        raw = _call_groq(system_prompt, user_message, temperature, max_tokens)
        elapsed = time.monotonic() - start
        logger.info("Groq call completed in %.2fs", elapsed)
        return _parse_json(raw, provider="groq")
    except groq.RateLimitError as e:
        logger.warning("Groq rate-limited (%s). Falling back to Gemini immediately.", e)
    except groq.APITimeoutError as e:
        logger.warning("Groq timed out (%s). Falling back to Gemini immediately.", e)
    except Exception as e:
        logger.warning("Groq call failed (%s: %s). Falling back to Gemini.", type(e).__name__, e)

    # ── Immediate Gemini fallback ─────────────────────────────────────────────
    try:
        start = time.monotonic()
        raw = _call_gemini(system_prompt, user_message, temperature, max_tokens)
        elapsed = time.monotonic() - start
        logger.info("Gemini fallback call completed in %.2fs", elapsed)
        return _parse_json(raw, provider="gemini")
    except Exception as e:
        logger.error("Gemini fallback also failed: %s: %s", type(e).__name__, e)
        return {}


def _parse_json(raw: str, provider: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = raw.strip()
    # Strip ```json ... ``` fences that some models add despite instructions
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("[%s] Failed to parse JSON response: %s\nRaw: %s", provider, e, raw[:500])
        return {}
