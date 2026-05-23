"""Tests for the integer arithmetic coder.

These tests pin the most important properties of the coder:

* Round-trip exactness on randomised streams under several probability models,
  including pathological ones (extremely skewed, uniform, two-symbol).
* The compressed size is close to the Shannon entropy of the source, within a
  small constant overhead due to bit alignment and the finish step.
* Edge cases: empty stream, single symbol, very long streams of one symbol.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from ncomp.coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    BitReader,
    BitWriter,
    decode_with_cdf,
    encode_with_cdf,
)


def _cdf_from_counts(counts: Sequence[int]) -> list[int]:
    cdf = [0]
    for c in counts:
        cdf.append(cdf[-1] + c)
    return cdf


def _round_trip(symbols: Sequence[int], counts: Sequence[int]) -> tuple[bytes, list[int]]:
    cdf = _cdf_from_counts(counts)
    writer = BitWriter()
    enc = ArithmeticEncoder(writer)
    for s in symbols:
        encode_with_cdf(enc, s, cdf)
    enc.finish()
    payload = writer.flush()

    reader = BitReader(payload)
    dec = ArithmeticDecoder(reader)
    recovered = [decode_with_cdf(dec, cdf) for _ in range(len(symbols))]
    return payload, recovered


def test_round_trip_uniform_bytes() -> None:
    rng = random.Random(0)
    symbols = [rng.randrange(256) for _ in range(2_000)]
    counts = [1] * 256
    payload, recovered = _round_trip(symbols, counts)
    assert recovered == symbols
    assert len(payload) > 0


def test_round_trip_skewed_distribution() -> None:
    rng = random.Random(1)
    counts = [1] * 8
    counts[0] = 1000
    cdf = _cdf_from_counts(counts)
    total = cdf[-1]
    symbols: list[int] = []
    for _ in range(4_000):
        target = rng.randrange(total)
        for s in range(len(counts)):
            if cdf[s] <= target < cdf[s + 1]:
                symbols.append(s)
                break
    payload, recovered = _round_trip(symbols, counts)
    assert recovered == symbols
    expected_bits = sum(-math.log2(counts[s] / total) for s in symbols)
    actual_bits = len(payload) * 8
    overhead = actual_bits - expected_bits
    assert overhead >= 0
    assert (
        overhead < 0.10 * expected_bits + 64
    ), f"actual_bits={actual_bits} expected={expected_bits} overhead={overhead}"


def test_round_trip_two_symbol_extremely_biased() -> None:
    counts = [999, 1]
    symbols = [0] * 10_000 + [1] * 10
    rng = random.Random(2)
    rng.shuffle(symbols)
    payload, recovered = _round_trip(symbols, counts)
    assert recovered == symbols
    total = sum(counts)
    expected_bits = sum(-math.log2(counts[s] / total) for s in symbols)
    actual_bits = len(payload) * 8
    assert actual_bits < expected_bits * 1.05 + 64


def test_round_trip_empty_stream() -> None:
    counts = [1, 1, 1]
    payload, recovered = _round_trip([], counts)
    assert recovered == []
    assert len(payload) <= 8


def test_round_trip_single_symbol() -> None:
    counts = [3, 1]
    payload, recovered = _round_trip([0], counts)
    assert recovered == [0]
    assert len(payload) >= 1


def test_round_trip_long_constant_run() -> None:
    counts = [1] * 256
    counts[42] = 10_000
    symbols = [42] * 20_000
    payload, recovered = _round_trip(symbols, counts)
    assert recovered == symbols
    p = counts[42] / sum(counts)
    expected_bits = -math.log2(p) * len(symbols)
    assert len(payload) * 8 < expected_bits * 1.05 + 64


def test_round_trip_many_random_streams() -> None:
    rng = random.Random(123)
    for trial in range(20):
        vocab = rng.randint(2, 50)
        counts = [rng.randint(1, 50) for _ in range(vocab)]
        n = rng.randint(0, 800)
        symbols = [rng.randrange(vocab) for _ in range(n)]
        _, recovered = _round_trip(symbols, counts)
        assert recovered == symbols, f"trial {trial}: vocab={vocab} n={n}"


def test_encode_rejects_invalid_slots() -> None:
    writer = BitWriter()
    enc = ArithmeticEncoder(writer)
    with pytest.raises(ValueError):
        enc.encode(5, 3, 10)
    with pytest.raises(ValueError):
        enc.encode(-1, 1, 10)
    with pytest.raises(ValueError):
        enc.encode(0, 11, 10)


def test_encode_finish_is_idempotent() -> None:
    writer = BitWriter()
    enc = ArithmeticEncoder(writer)
    enc.encode(0, 1, 2)
    enc.finish()
    enc.finish()
    payload = writer.flush()
    reader = BitReader(payload)
    dec = ArithmeticDecoder(reader)
    cdf = [0, 1, 2]
    assert decode_with_cdf(dec, cdf) == 0


def test_bitwriter_flush_pads_with_zeros() -> None:
    w = BitWriter()
    for b in (1, 0, 1):
        w.write_bit(b)
    out = w.flush()
    assert out == bytes([0b10100000])
    assert w.flush() == b""


def test_bitreader_returns_zero_past_eof() -> None:
    r = BitReader(b"\x00")
    for _ in range(8):
        assert r.read_bit() == 0
    for _ in range(20):
        assert r.read_bit() == 0


def test_bitwriter_rejects_invalid_bit() -> None:
    w = BitWriter()
    with pytest.raises(ValueError):
        w.write_bit(2)


def test_round_trip_after_writer_flush_extra_zeros() -> None:
    """The decoder treats trailing zero bits as zero, so extra padding is harmless."""
    rng = random.Random(7)
    counts = [3, 2, 1, 4]
    cdf = _cdf_from_counts(counts)
    symbols = [rng.randrange(4) for _ in range(50)]
    writer = BitWriter()
    enc = ArithmeticEncoder(writer)
    for s in symbols:
        encode_with_cdf(enc, s, cdf)
    enc.finish()
    payload = writer.flush()
    padded = payload + b"\x00" * 16
    reader = BitReader(padded)
    dec = ArithmeticDecoder(reader)
    recovered = [decode_with_cdf(dec, cdf) for _ in range(len(symbols))]
    assert recovered == symbols
