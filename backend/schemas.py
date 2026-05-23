"""Pydantic response models for the Neural Compressor API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """One row of the head-to-head comparison."""

    name: str
    compressed_bytes: int
    ratio: float
    bits_per_byte: float
    compress_seconds: float
    decompress_seconds: float
    ok: bool


class InputSummary(BaseModel):
    size_bytes: int = Field(..., ge=0)
    preview: str = Field(..., description="UTF-8 preview of the input (replacement on errors)")
    preview_truncated: bool


class SurpriseSeries(BaseModel):
    bits: list[float] = Field(default_factory=list)
    bytes_: list[int] = Field(default_factory=list, alias="bytes")

    model_config = {"populate_by_name": True}


class CompressionResponse(BaseModel):
    input: InputSummary
    results: list[ToolResult]
    surprise: SurpriseSeries


class InfoResponse(BaseModel):
    model_loaded: bool
    n_parameters: int | None = None
    context_length: int | None = None
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    fingerprint_hex: str | None = None
    eval_bits_per_byte: float | None = None
    max_upload_bytes: int


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
