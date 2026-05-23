"""Corpus loading and batch sampling for the training loop.

The corpus is read as raw bytes (no text decoding), then split into a training
prefix and a held-out suffix using ``train_fraction``. The split is chosen at
the nearest chapter boundary (a run of the form ``CHAPTER ...``) when one
exists nearby; otherwise we fall back to the exact byte offset. This keeps
the held-out text completely unseen by the model during training, which is
what makes the benchmark honest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_CHAPTER_PATTERN = re.compile(rb"\n\nCHAPTER\s+[IVXLC]+\.", re.IGNORECASE)


@dataclass(frozen=True)
class CorpusSplit:
    """A training/test split of a bytes corpus."""

    train: bytes
    test: bytes

    def __post_init__(self) -> None:
        if len(self.train) == 0:
            raise ValueError("train split must not be empty")
        if len(self.test) == 0:
            raise ValueError("test split must not be empty")


def load_corpus(path: str | Path, train_fraction: float = 0.8) -> CorpusSplit:
    """Load a byte corpus and split it into train / held-out test.

    The split picks the chapter boundary (or arbitrary byte boundary, if none)
    nearest the requested fraction.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    data = Path(path).read_bytes()
    target = int(len(data) * train_fraction)

    best = None
    best_dist = len(data)
    for match in _CHAPTER_PATTERN.finditer(data):
        pos = match.start() + 2
        dist = abs(pos - target)
        if dist < best_dist:
            best = pos
            best_dist = dist
    split_pos = best if best is not None else target

    train = data[:split_pos]
    test = data[split_pos:]
    return CorpusSplit(train=train, test=test)


def sample_batch(
    data: bytes,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of ``(input, target)`` pairs by drawing random windows."""
    if len(data) <= context_length + 1:
        raise ValueError(
            f"corpus too small ({len(data)} bytes) for context_length+1={context_length + 1}"
        )
    max_start = len(data) - context_length - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    arr = np.frombuffer(data, dtype=np.uint8)
    inputs = np.stack([arr[s : s + context_length] for s in starts])
    targets = np.stack([arr[s + 1 : s + context_length + 1] for s in starts])
    return (
        torch.from_numpy(inputs.astype(np.int64)),
        torch.from_numpy(targets.astype(np.int64)),
    )
