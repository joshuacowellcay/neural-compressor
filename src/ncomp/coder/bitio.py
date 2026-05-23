"""Bit-level I/O over byte buffers.

The arithmetic coder works one bit at a time. These helpers buffer bits into
whole bytes for output and yield bits from bytes on input. The reader keeps
returning ``0`` after the underlying stream is exhausted, which the arithmetic
decoder relies on as it drains the last few bits past end-of-stream.
"""

from __future__ import annotations

from collections.abc import Iterable


class BitWriter:
    """Accumulate individual bits and flush them as a ``bytes`` payload."""

    __slots__ = ("_buffer", "_current", "_n_bits")

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current = 0
        self._n_bits = 0

    def write_bit(self, bit: int) -> None:
        if bit not in (0, 1):
            raise ValueError(f"bit must be 0 or 1, got {bit!r}")
        self._current = (self._current << 1) | bit
        self._n_bits += 1
        if self._n_bits == 8:
            self._buffer.append(self._current)
            self._current = 0
            self._n_bits = 0

    def write_bits(self, bits: Iterable[int]) -> None:
        for b in bits:
            self.write_bit(b)

    def flush(self) -> bytes:
        """Pad with zero bits to the next byte boundary and return all output."""
        if self._n_bits:
            self._current <<= 8 - self._n_bits
            self._buffer.append(self._current)
            self._current = 0
            self._n_bits = 0
        out = bytes(self._buffer)
        self._buffer = bytearray()
        return out


class BitReader:
    """Yield bits from a ``bytes`` payload. Returns 0 after exhaustion."""

    __slots__ = ("_data", "_byte_pos", "_bit_pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0

    def read_bit(self) -> int:
        if self._byte_pos >= len(self._data):
            return 0
        byte = self._data[self._byte_pos]
        bit = (byte >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit
