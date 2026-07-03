"""
postfilter.py — Mandatory anti-hallucination step (§8).

Every recommended item from the LLM is verified against the catalog index.
URLs and test_type codes are ALWAYS pulled from the index — never trusted
from LLM output.
"""

import logging
from typing import Optional

from app.catalog_index import CatalogIndex
from app.schemas import Recommendation

logger = logging.getLogger(__name__)


def validate_recommendations(
    llm_output: list[dict],
    catalog_index: CatalogIndex,
    max_results: int = 10,
) -> list[Recommendation]:
    """
    For each LLM-proposed recommendation dict (must have at least 'name'),
    look it up in the catalog index. Replace URL and test_type with catalog
    ground-truth values. Drop items that don't match.

    Returns a validated list of Recommendation objects (max 10).
    """
    validated: list[Recommendation] = []

    for rec in llm_output:
        name_hint = rec.get("name", "").strip()
        if not name_hint:
            continue

        match = catalog_index.lookup_by_name(name_hint)
        if match:
            test_type_str = ",".join(match.get("test_type_codes", []))
            validated.append(
                Recommendation(
                    name=match["name"],                     # canonical name from index
                    url=match["link"],                      # ALWAYS from index, never LLM
                    test_type=test_type_str or "K",         # fallback "K" if derivation failed
                )
            )
        else:
            logger.warning(
                "POST-FILTER DROP: LLM recommended '%s' — not found in catalog.",
                name_hint,
            )

    truncated = validated[:max_results]
    logger.info(
        "Post-filter: %d in → %d validated → %d returned",
        len(llm_output), len(validated), len(truncated),
    )
    return truncated
