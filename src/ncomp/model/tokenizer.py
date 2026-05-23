"""Byte-level tokeniser.

The model operates directly on raw bytes (vocabulary size 256), so encoding
and decoding are trivial and total information is preserved exactly. This
keeps the compressor general (it can handle arbitrary binary input, not just
text) and removes any ambiguity that a learned tokeniser would introduce.
"""

from __future__ import annotations

VOCAB_SIZE = 256


def encode(data: bytes) -> list[int]:
    """Convert a bytes object to a list of integer tokens in ``[0, 255]``."""
    return list(data)


def decode(tokens: list[int]) -> bytes:
    """Inverse of :func:`encode`."""
    if not all(0 <= t < VOCAB_SIZE for t in tokens):
        raise ValueError("token out of range for byte vocabulary")
    return bytes(tokens)
