"""Integer-arithmetic encoder and decoder.

This is a textbook 32-bit-precision arithmetic coder with E1/E2/E3 (underflow)
scaling. The coding path uses only Python integers, no floating point, so the
encoder and decoder agree exactly on every interval split. As long as the
caller passes byte-identical integer frequency tables to both sides, the
round-trip is bit-exact.

Why integer arithmetic
----------------------
Arithmetic coding compresses a stream by progressively narrowing a fractional
interval. A floating-point implementation would let small rounding differences
between encode and decode push the interval over a boundary and silently
corrupt the bitstream. With pure integer arithmetic and a fixed precision,
every interval split is reproducible byte-for-byte on any machine.

Coder API
---------
The coder is symbol-agnostic. The caller supplies a cumulative frequency table
(CDF) at every step:

* ``encode(low_count, high_count, total_count)`` narrows the interval to the
  symbol's slot in the CDF.
* ``decode_target(total_count)`` returns a value in ``[0, total_count)`` that
  identifies which symbol's slot contains the current interval; the caller
  binary-searches the CDF to recover the symbol, then calls ``decode`` with
  that symbol's slot to keep the encoder and decoder in lockstep.

The :func:`encode_with_cdf` and :func:`decode_with_cdf` helpers take care of
the slot lookup when you have a full CDF array to hand.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from .bitio import BitReader, BitWriter

PRECISION = 32
FULL = 1 << PRECISION
HALF = FULL >> 1
QUARTER = FULL >> 2
THREE_QUARTERS = HALF + QUARTER
MAX_VAL = FULL - 1

MAX_TOTAL = 1 << (PRECISION - 2)


class ArithmeticEncoder:
    """Encode a stream of symbols into a bitstream using 32-bit precision."""

    __slots__ = ("_writer", "_low", "_high", "_pending_bits", "_finished")

    def __init__(self, writer: BitWriter) -> None:
        self._writer = writer
        self._low = 0
        self._high = MAX_VAL
        self._pending_bits = 0
        self._finished = False

    def encode(self, low_count: int, high_count: int, total_count: int) -> None:
        """Encode one symbol whose CDF slot is ``[low_count, high_count)``.

        ``total_count`` is the sum of all symbol frequencies, equal to the last
        entry of the CDF array. The slot describes the fraction of the interval
        the symbol owns: ``[low_count, high_count) / total_count``.
        """
        if self._finished:
            raise RuntimeError("encoder is already finished")
        if not 0 <= low_count < high_count <= total_count:
            raise ValueError(f"invalid slot: low={low_count} high={high_count} total={total_count}")
        if total_count > MAX_TOTAL:
            raise ValueError(
                f"total_count {total_count} exceeds MAX_TOTAL ({MAX_TOTAL}); "
                "reduce probability precision."
            )

        cur_range = self._high - self._low + 1
        self._high = self._low + (cur_range * high_count) // total_count - 1
        self._low = self._low + (cur_range * low_count) // total_count

        while True:
            if self._high < HALF:
                self._emit_bit_and_pending(0)
            elif self._low >= HALF:
                self._emit_bit_and_pending(1)
                self._low -= HALF
                self._high -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTERS:
                self._pending_bits += 1
                self._low -= QUARTER
                self._high -= QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1

    def finish(self) -> None:
        """Flush the remaining interval information; idempotent."""
        if self._finished:
            return
        self._pending_bits += 1
        if self._low < QUARTER:
            self._emit_bit_and_pending(0)
        else:
            self._emit_bit_and_pending(1)
        self._finished = True

    def _emit_bit_and_pending(self, bit: int) -> None:
        self._writer.write_bit(bit)
        opp = 1 - bit
        for _ in range(self._pending_bits):
            self._writer.write_bit(opp)
        self._pending_bits = 0


class ArithmeticDecoder:
    """Decode a bitstream produced by :class:`ArithmeticEncoder`."""

    __slots__ = ("_reader", "_low", "_high", "_code")

    def __init__(self, reader: BitReader) -> None:
        self._reader = reader
        self._low = 0
        self._high = MAX_VAL
        self._code = 0
        for _ in range(PRECISION):
            self._code = (self._code << 1) | reader.read_bit()

    def decode_target(self, total_count: int) -> int:
        """Return an integer in ``[0, total_count)`` indicating the next slot."""
        if total_count > MAX_TOTAL:
            raise ValueError(
                f"total_count {total_count} exceeds MAX_TOTAL ({MAX_TOTAL}); "
                "reduce probability precision."
            )
        cur_range = self._high - self._low + 1
        return ((self._code - self._low + 1) * total_count - 1) // cur_range

    def decode(self, low_count: int, high_count: int, total_count: int) -> None:
        """Advance the decoder, given the slot the caller just resolved."""
        if not 0 <= low_count < high_count <= total_count:
            raise ValueError(f"invalid slot: low={low_count} high={high_count} total={total_count}")
        cur_range = self._high - self._low + 1
        self._high = self._low + (cur_range * high_count) // total_count - 1
        self._low = self._low + (cur_range * low_count) // total_count

        while True:
            if self._high < HALF:
                pass
            elif self._low >= HALF:
                self._code -= HALF
                self._low -= HALF
                self._high -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTERS:
                self._code -= QUARTER
                self._low -= QUARTER
                self._high -= QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1
            self._code = (self._code << 1) | self._reader.read_bit()


def encode_with_cdf(encoder: ArithmeticEncoder, symbol: int, cdf: Sequence[int]) -> None:
    """Encode ``symbol`` whose slot is ``[cdf[symbol], cdf[symbol+1])``.

    ``cdf`` is a monotonically non-decreasing integer sequence with
    ``cdf[0] == 0`` and ``cdf[-1]`` equal to the total frequency. Each gap must
    be strictly positive (no zero-frequency symbols), otherwise that symbol
    cannot be coded at all.
    """
    encoder.encode(int(cdf[symbol]), int(cdf[symbol + 1]), int(cdf[-1]))


def decode_with_cdf(decoder: ArithmeticDecoder, cdf: Sequence[int]) -> int:
    """Decode the next symbol given the ``cdf`` for this step.

    Returns the index ``s`` such that ``cdf[s] <= target < cdf[s+1]``.
    """
    total = int(cdf[-1])
    target = decoder.decode_target(total)
    symbol = bisect_right(cdf, target) - 1
    if symbol < 0 or symbol + 1 >= len(cdf) or not cdf[symbol] <= target < cdf[symbol + 1]:
        raise ValueError(f"CDF lookup failed for target={target}, cdf bounds invalid")
    decoder.decode(int(cdf[symbol]), int(cdf[symbol + 1]), total)
    return symbol
