"""
test_hallucination_check.py — Verify that every URL returned by the agent
exists verbatim in the catalog index (§13.2).

This test directly validates the post-filter's primary job: no invented URLs.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import os
import httpx

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")

# Load catalog URLs once
METADATA_PATH = PROJECT_ROOT / "data" / "catalog_metadata.json"


def _load_catalog_urls() -> set[str]:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {item["link"] for item in metadata}


CATALOG_URLS = _load_catalog_urls()


def _make_chat_request(messages: list[dict]) -> dict:
    with httpx.Client(timeout=35) as client:
        resp = client.post(f"{BASE_URL}/chat", json={"messages": messages})
    assert resp.status_code == 200
    return resp.json()


def _check_urls_in_catalog(response: dict, context: str) -> list[str]:
    """Return list of URLs that are NOT in the catalog."""
    bad_urls = []
    for rec in response.get("recommendations", []):
        url = rec.get("url", "")
        if url and url not in CATALOG_URLS:
            bad_urls.append(url)
    return bad_urls


# ── Hallucination tests ───────────────────────────────────────────────────────

CONVERSATION_SCENARIOS = [
    (
        "java_engineer",
        [
            {"role": "user", "content": "Hiring a senior Java developer with Spring and SQL experience."},
            {"role": "assistant", "content": "What seniority level?"},
            {"role": "user", "content": "Senior IC, 6 years experience."},
        ],
    ),
    (
        "contact_centre",
        [
            {"role": "user", "content": "Screening entry-level contact centre agents, US English."},
        ],
    ),
    (
        "graduate_analysts",
        [
            {"role": "user", "content": "Hiring graduate financial analysts. Need numerical reasoning and finance knowledge."},
        ],
    ),
    (
        "safety_critical",
        [
            {"role": "user", "content": "Hiring plant operators for a chemical facility. Safety is top priority."},
        ],
    ),
    (
        "admin_assistants",
        [
            {"role": "user", "content": "Need to quickly screen admin assistants for Excel and Word skills."},
        ],
    ),
]


@pytest.mark.parametrize("name,messages", CONVERSATION_SCENARIOS)
def test_no_hallucinated_urls(name: str, messages: list[dict]) -> None:
    """Assert all returned URLs exist verbatim in the catalog."""
    resp = _make_chat_request(messages)
    bad_urls = _check_urls_in_catalog(resp, name)
    assert not bad_urls, (
        f"[{name}] Hallucinated URLs found (not in catalog):\n"
        + "\n".join(f"  - {u}" for u in bad_urls)
    )


def test_no_hallucinated_urls_rust_engineer():
    """C2 scenario: Rust engineer with catalog gap — no fabricated URLs allowed."""
    messages = [
        {"role": "user", "content": "Hiring a senior Rust engineer for high-performance networking."},
        {"role": "assistant", "content": "SHL doesn't have a Rust-specific test. Closest fits: Smart Interview Live Coding, Linux Programming. Want me to build a shortlist?"},
        {"role": "user", "content": "Yes, go ahead. Add cognitive test too."},
    ]
    resp = _make_chat_request(messages)
    bad_urls = _check_urls_in_catalog(resp, "rust_engineer")
    assert not bad_urls, f"Hallucinated URLs: {bad_urls}"


def test_test_type_codes_valid():
    """All test_type values must be valid code strings (A, B, C, D, E, K, P, S or comma-joined)."""
    valid_codes = {"A", "B", "C", "D", "E", "K", "P", "S"}
    messages = [
        {"role": "user", "content": "Hiring a mid-level data analyst. Need SQL and numerical reasoning."},
    ]
    resp = _make_chat_request(messages)
    for rec in resp.get("recommendations", []):
        test_type = rec.get("test_type", "")
        codes = [c.strip() for c in test_type.split(",") if c.strip()]
        for code in codes:
            assert code in valid_codes, f"Invalid test_type code '{code}' in recommendation {rec['name']}"
