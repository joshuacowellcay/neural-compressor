"""Configuration dataclasses for the model, coder, and training loop.

Configs are loaded from and saved to JSON (see :mod:`ncomp.training.io`). They
also travel inside the model checkpoint so that compression and decompression
are guaranteed to use exactly the same architecture and coding parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Architecture of the small causal Transformer."""

    model_config = ConfigDict(extra="forbid")

    vocab_size: int = Field(default=256, ge=2, le=65536)
    context_length: int = Field(default=64, ge=1, le=4096)
    d_model: int = Field(default=128, ge=8)
    n_layers: int = Field(default=4, ge=1)
    n_heads: int = Field(default=4, ge=1)
    d_ff: int = Field(default=256, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    tie_weights: bool = True

    def validate_heads(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


class CodingConfig(BaseModel):
    """Integer-arithmetic coder settings shared by encoder and decoder."""

    model_config = ConfigDict(extra="forbid")

    prob_total: int = Field(default=16384, ge=256)
    min_count: int = Field(default=1, ge=1)

    def validate_for_vocab(self, vocab_size: int) -> None:
        if self.prob_total < vocab_size * self.min_count:
            raise ValueError(
                f"prob_total ({self.prob_total}) must be at least "
                f"vocab_size * min_count ({vocab_size * self.min_count})"
            )


class TrainingConfig(BaseModel):
    """Hyperparameters for the training loop."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    batch_size: int = Field(default=64, ge=1)
    n_steps: int = Field(default=3000, ge=1)
    warmup_steps: int = Field(default=200, ge=0)
    learning_rate: float = Field(default=6e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = Field(default=1.0, gt=0)
    eval_every: int = Field(default=200, ge=1)
    eval_batches: int = Field(default=32, ge=1)
    train_fraction: float = Field(default=0.8, gt=0, lt=1)


class FullConfig(BaseModel):
    """Top-level config bundle persisted with each checkpoint."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    coding: CodingConfig = Field(default_factory=CodingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> FullConfig:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        cfg = cls.model_validate(data)
        cfg.model.validate_heads()
        cfg.coding.validate_for_vocab(cfg.model.vocab_size)
        return cfg

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
            f.write("\n")
