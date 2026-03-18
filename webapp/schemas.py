"""
ModernBERT-RGAT | Pydantic Schemas
====================================
Request/Response models for the FastAPI backend.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body for the /api/analyze endpoint."""
    text: str = Field(..., min_length=1, max_length=2000, description="Restaurant review text")
    model_year: str = Field(
        default="best",
        description="Model to use: 'best', '2014', '2015', or '2016'"
    )


class AspectResult(BaseModel):
    """A single extracted aspect with sentiment."""
    aspect: str
    sentiment: str
    confidence: float
    start: int
    end: int


class AnalyzeResponse(BaseModel):
    """Response body from the /api/analyze endpoint."""
    text: str
    model_used: str
    aspects: List[AspectResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response body from the /api/health endpoint."""
    status: str
    models_loaded: List[str]
    device: str
