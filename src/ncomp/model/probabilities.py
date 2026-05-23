"""Deterministic probability quantisation.

The neural model outputs float probabilities. The arithmetic coder operates on
integer frequencies. The encoder and decoder must agree on the integer
frequencies bit-for-bit, otherwise the round-trip fails. This module provides
a deterministic quantiser that converts a float probability vector to a vector
of integer counts summing to a fixed ``prob_total`` with every count at least
``min_count`` (so no symbol has zero probability and the coder can always
represent every symbol).

Algorithm
---------
1. Convert the input to ``float64`` for stability.
2. Allocate ``min_count`` to every symbol up front, scale the remaining pool
   ``(total - vocab * min_count)`` by the probabilities, take the floor.
3. The integer sum is now ``<= total``. Distribute the residual one-by-one to
   the symbols with the largest fractional parts (ties broken by symbol
   index, smaller index first). After this step the counts sum exactly to
   ``total`` and every count is at least ``min_count``.

The algorithm is implemented in NumPy with stable sorts and integer
arithmetic, so for any given probability vector it produces the same counts
on any machine running the same NumPy version.
"""

from __future__ import annotations

import numpy as np
import torch


def quantize_probabilities(
    probs: torch.Tensor | np.ndarray,
    total: int,
    min_count: int = 1,
) -> np.ndarray:
    """Quantise a probability vector to integer counts summing to ``total``."""
    if isinstance(probs, torch.Tensor):
        arr = probs.detach().to(torch.float64).cpu().numpy()
    else:
        arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D probabilities, got shape {arr.shape}")

    vocab = int(arr.shape[0])
    if total < vocab * min_count:
        raise ValueError(f"total={total} too small for vocab={vocab} with min_count={min_count}")

    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= 0.0 or not np.isfinite(s):
        arr = np.full(vocab, 1.0 / vocab, dtype=np.float64)
    else:
        arr = arr / s

    pool = total - vocab * min_count
    scaled = arr * pool
    floors = np.floor(scaled).astype(np.int64)
    counts = floors + int(min_count)
    deficit = int(total - counts.sum())

    if deficit > 0:
        frac = scaled - floors
        order = np.lexsort((np.arange(vocab), -frac))
        counts[order[:deficit]] += 1
    elif deficit < 0:
        raise AssertionError(f"quantise produced surplus deficit={deficit} (should not happen)")

    if int(counts.sum()) != total:
        raise AssertionError(f"counts sum to {counts.sum()}, expected {total}")
    if int(counts.min()) < min_count:
        raise AssertionError(f"min count {counts.min()} below required min_count {min_count}")
    return counts


def cdf_from_counts(counts: np.ndarray) -> np.ndarray:
    """Return the length-``(vocab+1)`` cumulative frequency table for ``counts``."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("counts must be 1-D")
    cdf = np.empty(len(counts) + 1, dtype=np.int64)
    cdf[0] = 0
    np.cumsum(counts, out=cdf[1:])
    return cdf


def probabilities_to_cdf(
    probs: torch.Tensor | np.ndarray,
    total: int,
    min_count: int = 1,
) -> np.ndarray:
    """Convenience: quantise probs and return the CDF in one step."""
    return cdf_from_counts(quantize_probabilities(probs, total, min_count))
