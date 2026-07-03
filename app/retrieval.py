"""
retrieval.py — Retrieval + ranking engine (§3).

Embeds query, performs FAISS cosine search, applies hard filters,
soft-boosts, and returns a naturally-sized shortlist.

The SentenceTransformer model is loaded once at module import (startup).
"""

import logging
import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    MAX_RECOMMENDATIONS,
    JOB_LEVEL_BOOST,
    KEYWORD_BOOST,
)
from app.catalog_index import CatalogIndex

logger = logging.getLogger(__name__)

# ── Load embedding model at startup ──────────────────────────────────────────
logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
_embedder = SentenceTransformer(EMBEDDING_MODEL)
logger.info("Embedding model loaded.")


def _embed(text: str) -> np.ndarray:
    """Encode text and L2-normalise so inner-product == cosine similarity."""
    vec = _embedder.encode(text, normalize_embeddings=True)
    return vec.astype("float32")


def _keyword_overlap(skills: list[str], text: str) -> float:
    """Fraction of required skills mentioned in item name+description."""
    if not skills:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for s in skills if s.lower() in text_lower)
    return hits / len(skills)


def _job_level_match(filter_levels: Optional[list[str]], item_levels: set) -> float:
    """1.0 if any requested job level is in item_levels, else 0.0."""
    if not filter_levels or not item_levels:
        return 0.0
    return 1.0 if any(lvl in item_levels for lvl in filter_levels) else 0.0


def _passes_language_filter(
    language_filter: Optional[str],
    item_languages: set,
) -> bool:
    """
    Hard language filter (§3):
    - If no filter → always pass.
    - Items with empty language set → ambiguous, DO NOT exclude (pass through).
    - Items with non-empty language set → exclude only if requested language absent.
    """
    if not language_filter:
        return True
    if not item_languages:   # unknown languages → keep, flag in context
        return True
    # Case-insensitive substring match (handles "English (USA)" vs "English")
    lf_lower = language_filter.lower()
    return any(lf_lower in lang.lower() for lang in item_languages)


def _passes_test_type_filter(
    test_type_filter: Optional[list[str]],
    item_codes: list[str],
) -> bool:
    """Hard test-type filter: item must have at least one of the requested codes."""
    if not test_type_filter:
        return True
    return any(code in item_codes for code in test_type_filter)


def build_query_text(role_context: str, required_skills: list[str]) -> str:
    """Build the embedding query string from extracted slots."""
    skills_str = ", ".join(required_skills) if required_skills else ""
    parts = [role_context]
    if skills_str:
        parts.append(f"Skills: {skills_str}.")
    return " ".join(parts).strip()


def retrieve(
    catalog_index: CatalogIndex,
    role_context: str,
    required_skills: list[str],
    test_type_filter: Optional[list[str]] = None,
    language_filter: Optional[str] = None,
    job_level_filter: Optional[list[str]] = None,
    excluded_skills: Optional[list[str]] = None,
) -> list[dict]:
    """
    Full retrieval pipeline (§3):
    1. Embed query
    2. FAISS cosine search (top RETRIEVAL_TOP_K)
    3. Hard filters (test_type, language)
    4. Soft boosts (job_level_match + keyword_overlap)
    5. Sort by final score, return naturally-sized list (≤ MAX_RECOMMENDATIONS)
    """
    query_text = build_query_text(role_context, required_skills)
    logger.info("Retrieval query: %s", query_text[:120])

    query_vec = _embed(query_text)
    candidates = catalog_index.search(query_vec, RETRIEVAL_TOP_K)

    # ── Hard filters ──────────────────────────────────────────────────────────
    filtered = []
    for item in candidates:
        if not _passes_test_type_filter(test_type_filter, item.get("test_type_codes", [])):
            continue
        if not _passes_language_filter(language_filter, item.get("languages", set())):
            continue
        filtered.append(item)

    # ── Soft boosts ───────────────────────────────────────────────────────────
    scored = []
    for item in filtered:
        base = item["_cosine_sim"]
        boost = (
            JOB_LEVEL_BOOST * _job_level_match(job_level_filter, item.get("job_levels", set()))
            + KEYWORD_BOOST * _keyword_overlap(
                required_skills,
                item["name"] + " " + item.get("description", ""),
            )
        )
        final_score = base + boost
        scored.append((final_score, item))

    # Sort descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Natural shortlist (1–10, don't pad artificially) ─────────────────────
    # Drop items that are clearly below a relevance threshold
    SCORE_THRESHOLD = 0.20
    natural = [item for score, item in scored if score >= SCORE_THRESHOLD]

    if not natural and scored:
        # If nothing clears threshold, take top-3 as a best-effort
        natural = [item for _, item in scored[:3]]

    result = natural[:MAX_RECOMMENDATIONS]
    logger.info("Retrieval: %d candidates → %d filtered → %d returned", len(candidates), len(filtered), len(result))
    return result
