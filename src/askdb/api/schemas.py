"""Request and response shapes.

Every response carries its own cost and latency. That is deliberate: the
tradeoff between answer quality and what it cost to produce is the thing this
project is about, and a number that has to be reconstructed from logs
afterwards usually is not reconstructed at all.
"""

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    database: str
    evidence: str = Field(default="", max_length=2000)
    samples: int | None = Field(default=None, ge=1, le=8)


class TraceStep(BaseModel):
    kind: str
    detail: str
    model: str | None = None
    tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class Meta(BaseModel):
    input_tokens: int
    output_tokens: int
    model_calls: int
    repairs: int
    latency_ms: float
    agreement: float | None = None


class AskResponse(BaseModel):
    question: str
    database: str

    # answer / clarify / refuse. A query can run perfectly and still not be
    # offered, if the confidence assessment declined to stand behind it.
    decision: str
    confidence: float
    reasons: list[str] = []

    sql: str | None = None
    columns: list[str] = []
    rows: list[list[Any]] = []
    truncated: bool = False
    error: str | None = None

    tables_considered: list[str] = []
    trace: list[TraceStep] = []
    meta: Meta


class DatabaseSummary(BaseModel):
    name: str
    tables: int
    rows: int


class HealthResponse(BaseModel):
    status: str
    databases: int
    model: str
