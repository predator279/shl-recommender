"""
build_index.py — ONE-TIME offline script to build and serialize the FAISS
vector index + metadata sidecar from shl_product_catalogue.json.

Run this from the project root before building the Docker image:
    python scripts/build_index.py

Outputs:
    data/catalog_index.faiss   — FAISS flat IP index (normalised → cosine sim)
    data/catalog_metadata.json — Metadata sidecar for all 377 entries

Do NOT run this at server startup. The pre-built index is baked into the
Docker image and loaded at process startup by catalog_index.py.
"""

import json
import re
import sys
from pathlib import Path

# Add project root to path so we can import app.config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    CATALOGUE_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL,
    KEY_TO_CODE,
)


# ── Utility functions ─────────────────────────────────────────────────────────

def derive_codes(keys: list[str]) -> list[str]:
    """Map 'keys' list to sorted single-letter test_type codes."""
    return sorted({KEY_TO_CODE[k] for k in keys if k in KEY_TO_CODE})


def parse_duration_minutes(duration_raw: str) -> int | None:
    """Extract integer minute count from raw duration string."""
    if not duration_raw:
        return None
    m = re.search(r"(\d+)", duration_raw)
    return int(m.group(1)) if m else None


def build_embedding_text(item: dict) -> str:
    """
    Build the text that gets embedded for semantic similarity search (§3).
    Deliberately excludes languages (hard-filter fact, not similarity signal).
    """
    job_levels_str = ", ".join(item.get("job_levels", [])) or "not specified"
    return (
        f"{item['name']}. "
        f"{item['description']} "
        f"Category: {', '.join(item.get('keys', []))}. "
        f"Typical job levels: {job_levels_str}."
    )


# ── Main build logic ──────────────────────────────────────────────────────────

def build_index() -> None:
    print(f"Loading catalogue from: {CATALOGUE_PATH}")
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        catalogue: list[dict] = json.load(f)

    # Filter to status == "ok" (all 377 should be, but be explicit)
    catalogue = [item for item in catalogue if item.get("status") == "ok"]
    print(f"Loaded {len(catalogue)} catalogue entries.")

    # ── Build embedding texts and metadata ───────────────────────────────────
    embedding_texts: list[str] = []
    metadata_records: list[dict] = []

    for item in catalogue:
        embedding_texts.append(build_embedding_text(item))

        # Parse duration
        duration_minutes = parse_duration_minutes(item.get("duration_raw", ""))

        # Derive test type codes
        test_type_codes = derive_codes(item.get("keys", []))

        # Store metadata (sets serialized as lists for JSON; reload as sets in catalog_index.py)
        metadata_records.append({
            "entity_id":        item["entity_id"],
            "name":             item["name"],
            "link":             item["link"],
            "test_type_codes":  test_type_codes,
            "job_levels":       list(item.get("job_levels", [])),
            "languages":        list(item.get("languages", [])),   # empty list = unknown
            "duration_minutes": duration_minutes,
            "duration":         item.get("duration", ""),
            "description":      item.get("description", ""),
            "keys":             item.get("keys", []),
            "remote":           item.get("remote", ""),
            "adaptive":         item.get("adaptive", ""),
        })

    # ── Load embedding model and encode ──────────────────────────────────────
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Encoding {len(embedding_texts)} texts...")
    embeddings = model.encode(
        embedding_texts,
        normalize_embeddings=True,   # L2-normalise → inner product == cosine similarity
        batch_size=64,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype("float32")
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Build FAISS flat IP index ─────────────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product (= cosine for normalised vecs)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"Saved FAISS index: {FAISS_INDEX_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata sidecar: {METADATA_PATH}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    no_duration = sum(1 for m in metadata_records if m["duration_minutes"] is None)
    no_languages = sum(1 for m in metadata_records if not m["languages"])
    print(f"\nData quality summary:")
    print(f"  Total entries: {len(metadata_records)}")
    print(f"  No duration:   {no_duration} (will appear as '—' in responses)")
    print(f"  No languages:  {no_languages} (treated as 'unconfirmed', never excluded)")
    print("\nIndex build complete.")


if __name__ == "__main__":
    build_index()
