"""
catalog_index.py — Loads the prebuilt FAISS index + metadata sidecar at
module-import time. Provides name lookup (exact + fuzzy) used by the
post-filter and compare-intent path.

IMPORTANT: models are loaded once at process startup, never lazily.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rapidfuzz import process as fuzz_process, fuzz

from app.config import FAISS_INDEX_PATH, METADATA_PATH

logger = logging.getLogger(__name__)


class CatalogIndex:
    """Thin wrapper around the prebuilt FAISS index and metadata sidecar."""

    def __init__(self, faiss_path: Path, metadata_path: Path) -> None:
        logger.info("Loading FAISS index from %s", faiss_path)
        self.index: faiss.Index = faiss.read_index(str(faiss_path))

        logger.info("Loading metadata sidecar from %s", metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_metadata: list[dict] = json.load(f)

        # Convert list fields to sets for O(1) membership testing in retrieval
        self.metadata: list[dict] = []
        for item in raw_metadata:
            item["job_levels"] = set(item.get("job_levels", []))
            item["languages"] = set(item.get("languages", []))
            self.metadata.append(item)

        # Build a name → metadata dict for O(1) exact lookup
        self._name_map: dict[str, dict] = {
            item["name"].lower(): item for item in self.metadata
        }
        # Sorted list of names for fuzzy matching
        self._names: list[str] = list(self._name_map.keys())

        logger.info("CatalogIndex ready: %d entries", len(self.metadata))

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """
        Return top_k metadata dicts by cosine similarity.
        query_vector must already be L2-normalised (done in retrieval.py).
        """
        query_vector = query_vector.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["_cosine_sim"] = float(dist)   # inner-product == cosine for normalised vecs
            results.append(item)
        return results

    # ── Name lookup (used by post-filter + compare) ────────────────────────────

    def lookup_by_name(self, name: str) -> Optional[dict]:
        """
        Exact match first (case-insensitive), then fuzzy fallback.
        Returns the metadata dict or None if score < threshold.
        """
        key = name.strip().lower()

        # Exact
        if key in self._name_map:
            return self._name_map[key]

        # Fuzzy (RapidFuzz token_sort_ratio — handles word-order variants)
        if not self._names:
            return None
        result = fuzz_process.extractOne(
            key,
            self._names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=80,         # reject weak matches
        )
        if result:
            matched_key, score, _ = result
            logger.debug("Fuzzy match: '%s' → '%s' (score=%d)", name, matched_key, score)
            return self._name_map[matched_key]

        logger.warning("No catalog match for name: '%s'", name)
        return None

    def get_all(self) -> list[dict]:
        return self.metadata


# ── Singleton loaded at module import (startup) ──────────────────────────────
_catalog_index: Optional[CatalogIndex] = None


def get_catalog_index() -> CatalogIndex:
    global _catalog_index
    if _catalog_index is None:
        _catalog_index = CatalogIndex(FAISS_INDEX_PATH, METADATA_PATH)
    return _catalog_index
