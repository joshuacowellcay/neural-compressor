"""Backend state: a lazily-loaded compressor with thread-safe access.

The compressor takes a few seconds to load (it builds a Transformer and reads
the checkpoint from disk). We load it once at app startup and keep it for the
process lifetime; concurrent requests share the same in-memory instance under
a lock because PyTorch models are not generally thread-safe for inference.
"""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from ncomp.benchmark import run_all
from ncomp.model.checkpoint import load_checkpoint
from ncomp.pipeline import NeuralCompressor

from .schemas import (
    CompressionResponse,
    InfoResponse,
    InputSummary,
    SurpriseSeries,
    ToolResult,
)

_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "checkpoint.pt"
_DEFAULT_MAX_UPLOAD = 4 * 1024


@dataclass
class BackendSettings:
    checkpoint_path: Path = _DEFAULT_CHECKPOINT
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD

    @classmethod
    def from_env(cls) -> BackendSettings:
        s = cls()
        if env_ckpt := os.environ.get("NCOMP_MODEL_PATH"):
            s.checkpoint_path = Path(env_ckpt)
        if env_max := os.environ.get("NCOMP_MAX_UPLOAD_BYTES"):
            with contextlib.suppress(ValueError):
                s.max_upload_bytes = int(env_max)
        return s


class AppState:
    """Holds the loaded model and config; thread-safe for inference calls."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or BackendSettings.from_env()
        self.compressor: NeuralCompressor | None = None
        self.eval_bpb: float | None = None
        self.n_parameters: int | None = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.compressor is not None

    def load(self) -> None:
        if self.compressor is not None:
            return
        if not self.settings.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.settings.checkpoint_path}")
        model, cfg, extra = load_checkpoint(self.settings.checkpoint_path)
        self.compressor = NeuralCompressor(model, cfg)
        self.eval_bpb = extra.get("best_eval_bpb")
        self.n_parameters = extra.get("n_params")

    def info(self) -> InfoResponse:
        if self.compressor is None:
            return InfoResponse(model_loaded=False, max_upload_bytes=self.settings.max_upload_bytes)
        cfg = self.compressor.config.model
        return InfoResponse(
            model_loaded=True,
            n_parameters=self.n_parameters,
            context_length=cfg.context_length,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            fingerprint_hex=self.compressor.fingerprint.hex(),
            eval_bits_per_byte=self.eval_bpb,
            max_upload_bytes=self.settings.max_upload_bytes,
        )

    def compress_with_comparison(self, data: bytes) -> CompressionResponse:
        if self.compressor is None:
            raise RuntimeError("model not loaded")
        with self._lock:
            results = run_all(data, self.compressor)
            compressor_result = self.compressor.compress(data, capture_surprise=True)

        preview_len = min(200, len(data))
        preview = data[:preview_len].decode("utf-8", errors="replace")

        rows = [
            ToolResult(
                name=r.name,
                compressed_bytes=r.compressed_bytes,
                ratio=r.ratio,
                bits_per_byte=r.bits_per_byte,
                compress_seconds=r.compress_seconds,
                decompress_seconds=r.decompress_seconds,
                ok=r.ok,
            )
            for r in results
        ]

        surprise = SurpriseSeries(
            bits=compressor_result.surprise_bits or [],
            bytes_=list(data),
        )

        return CompressionResponse(
            input=InputSummary(
                size_bytes=len(data),
                preview=preview,
                preview_truncated=preview_len < len(data),
            ),
            results=rows,
            surprise=surprise,
        )

    def decompress(self, payload: bytes) -> bytes:
        if self.compressor is None:
            raise RuntimeError("model not loaded")
        with self._lock:
            return self.compressor.decompress(payload)
