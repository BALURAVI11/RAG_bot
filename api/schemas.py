"""Pydantic schemas for API request and response validation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to ask the knowledge assistant",
        examples=["What is RAG and how does it work?"],
    )


class SourceCitation(BaseModel):
    source: str
    topic: Optional[str] = None
    page: Optional[Any] = None
    preview: str


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer grounded in retrieved documents")
    sources: List[SourceCitation] = Field(default_factory=list)
    retrieved_chunks: int = Field(..., description="Number of document chunks used")
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")


class IngestResponse(BaseModel):
    status: str
    message: str


class ResetResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    query_count: int
    vector_db: str
    llm_provider: str
    embedding_provider: str
    reranker_enabled: bool
