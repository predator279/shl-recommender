"""
test_schema_compliance.py — Verify that every API response in the 10 reference
conversations is schema-compliant (§13.1).

Tests:
- recommendations is always a list (never null)
- 0 <= len(recommendations) <= 10
- end_of_conversation is boolean
- Every item has name, url, test_type (non-empty strings)
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import os
import httpx

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")


def test_health_endpoint():
    """Health check returns 200 with status ok."""
    with httpx.Client() as client:
        resp = client.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def _make_chat_request(messages: list[dict]) -> dict:
    """Helper: POST to /chat and return parsed response."""
    with httpx.Client(timeout=35) as client:
        resp = client.post(f"{BASE_URL}/chat", json={"messages": messages})
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    return resp.json()


def _assert_schema_compliance(response: dict, context: str = "") -> None:
    """Assert all required fields are present and valid types."""
    assert "reply" in response, f"Missing 'reply' in response. Context: {context}"
    assert isinstance(response["reply"], str), f"'reply' must be str. Context: {context}"
    assert len(response["reply"]) > 0, f"'reply' must not be empty. Context: {context}"

    assert "recommendations" in response, f"Missing 'recommendations'. Context: {context}"
    assert isinstance(response["recommendations"], list), \
        f"'recommendations' must be a list (never null). Context: {context}"
    assert 0 <= len(response["recommendations"]) <= 10, \
        f"recommendations length must be 0-10, got {len(response['recommendations'])}. Context: {context}"

    assert "end_of_conversation" in response, f"Missing 'end_of_conversation'. Context: {context}"
    assert isinstance(response["end_of_conversation"], bool), \
        f"'end_of_conversation' must be bool. Context: {context}"

    for i, rec in enumerate(response["recommendations"]):
        assert "name" in rec and rec["name"], \
            f"Recommendation[{i}] missing 'name'. Context: {context}"
        assert "url" in rec and rec["url"], \
            f"Recommendation[{i}] missing 'url'. Context: {context}"
        assert "test_type" in rec and rec["test_type"], \
            f"Recommendation[{i}] missing 'test_type'. Context: {context}"


# ── Replay test conversations ──────────────────────────────────────────────────

def test_single_vague_turn():
    """Vague opening → clarify_needed → recommendations must be []."""
    messages = [{"role": "user", "content": "I need an assessment."}]
    resp = _make_chat_request(messages)
    _assert_schema_compliance(resp, "vague opening")
    assert resp["recommendations"] == [], \
        "Vague opening must return empty recommendations"
    assert resp["end_of_conversation"] is False


def test_two_turn_conversation():
    """Two turns: vague → clarification → still schema-compliant."""
    messages = [
        {"role": "user", "content": "We need a solution for senior leadership."},
        {"role": "assistant", "content": "Happy to help narrow that down. Who is this meant for?"},
        {"role": "user", "content": "CXOs and directors with 15+ years experience."},
    ]
    resp = _make_chat_request(messages)
    _assert_schema_compliance(resp, "two-turn conversation")


def test_full_java_engineer_conversation():
    """Replay core of C9: Java engineer JD → backend-leaning → senior IC."""
    messages = [
        {
            "role": "user",
            "content": (
                "Here's the JD for an engineer we need to fill. Can you recommend an assessment battery?\n"
                '"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, '
                'Angular, SQL/relational databases, AWS deployment, and Docker."'
            ),
        },
        {
            "role": "assistant",
            "content": "Is this backend-leaning or full-stack? I need to know the primary focus.",
        },
        {"role": "user", "content": "Backend-leaning. Core Java and Spring are day-one priorities."},
        {
            "role": "assistant",
            "content": "Is the seniority closer to a senior IC or a tech lead?",
        },
        {"role": "user", "content": "Senior IC. They lead design on their own services."},
    ]
    resp = _make_chat_request(messages)
    _assert_schema_compliance(resp, "java engineer conversation")
    # After 3 clarifying turns, we expect recommendations now
    assert len(resp["recommendations"]) > 0, "Should have recommendations for senior IC Java role"


def test_out_of_scope_refusal():
    """Out-of-scope legal question → refusal, recommendations=[], end_of_conversation=False."""
    messages = [
        {
            "role": "user",
            "content": "Am I legally required under HIPAA to test all staff who touch patient records?",
        }
    ]
    resp = _make_chat_request(messages)
    _assert_schema_compliance(resp, "out-of-scope legal question")
    assert resp["recommendations"] == [], "Out-of-scope must return empty recommendations"
    assert resp["end_of_conversation"] is False, "Out-of-scope must not end conversation"


def test_end_of_conversation_true():
    """User confirms shortlist → end_of_conversation must be True."""
    messages = [
        {"role": "user", "content": "Hiring graduate financial analysts. Need numerical reasoning."},
        {
            "role": "assistant",
            "content": "Here are my recommendations... [shortlist]",
        },
        {"role": "user", "content": "That covers it. Confirmed."},
    ]
    resp = _make_chat_request(messages)
    _assert_schema_compliance(resp, "confirmation turn")
    # end_of_conversation should be True
    assert resp["end_of_conversation"] is True, \
        "User confirmation must set end_of_conversation to True"
