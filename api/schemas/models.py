from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    persona: Literal["scientist", "doctor", "faq"]


class ChatResponse(BaseModel):
    answer: str
    persona: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_store_ready: bool
