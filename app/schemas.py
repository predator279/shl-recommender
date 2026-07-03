"""
schemas.py — Pydantic models for request/response validation.
The API schema is non-negotiable (matches assignment spec §6 exactly).
"""

from typing import List
from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages list must not be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str   # e.g. "K", "K,S", "P"


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]   # ALWAYS a list, NEVER null (§6)
    end_of_conversation: bool


class HealthResponse(BaseModel):
    status: str
