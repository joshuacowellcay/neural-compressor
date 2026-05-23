"""Saving and loading checkpoints.

A checkpoint stores both the model weights and the full configuration so that
a downstream caller (the compressor or decompressor) can rebuild the exact
architecture and coding parameters without any external state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import FullConfig
from .transformer import CausalTransformer


def save_checkpoint(
    path: str | Path,
    model: CausalTransformer,
    config: FullConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "config": config.model_dump(),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[CausalTransformer, FullConfig, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = FullConfig.model_validate(payload["config"])
    config.model.validate_heads()
    model = CausalTransformer(config.model)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    extra = payload.get("extra", {}) or {}
    return model, config, extra
