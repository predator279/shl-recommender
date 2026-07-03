"""
config.py — Environment variables, model names, path constants.
Loaded once at import time; all other modules import from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (local dev); in Docker, env vars come from the runtime
load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ── Model names ───────────────────────────────────────────────────────────────
GROQ_MODEL: str = "llama-3.3-70b-versatile"
GEMINI_MODEL: str = "gemini-flash-lite-latest"      # confirmed free-tier working fallback

# Embedding model — local, no API call
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── Data paths ────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"

CATALOGUE_PATH: Path = DATA_DIR / "shl_product_catalogue.json"
FAISS_INDEX_PATH: Path = DATA_DIR / "catalog_index.faiss"
METADATA_PATH: Path = DATA_DIR / "catalog_metadata.json"

PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"

# ── Retrieval constants ───────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 20          # candidates fetched before filtering/ranking
MAX_RECOMMENDATIONS: int = 10

# Soft-boost weights (§3)
JOB_LEVEL_BOOST: float = 0.15
KEYWORD_BOOST: float = 0.10

# ── Conversation limits ───────────────────────────────────────────────────────
MAX_EXCHANGES: int = 8             # 1 exchange = 1 user + 1 assistant message pair

# ── key → single-letter test_type code (§2) ───────────────────────────────────
KEY_TO_CODE: dict[str, str] = {
    "Ability & Aptitude":           "A",
    "Biodata & Situational Judgment": "B",
    "Competencies":                 "C",
    "Development & 360":            "D",
    "Assessment Exercises":         "E",
    "Knowledge & Skills":           "K",
    "Personality & Behavior":       "P",
    "Simulations":                  "S",
}
