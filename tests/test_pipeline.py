"""Round-trip and edge-case tests for the full compression pipeline.

These tests do not require a *trained* model: an untrained Transformer is
fine for proving the round-trip is exact, because what matters here is that
encoder and decoder see the same probabilities. The benchmark tests in
``test_benchmark.py`` (and the manual benchmark report) cover compression
ratio with a trained checkpoint.
"""

from __future__ import annotations

import random

import pytest
import torch

from ncomp.model.config import CodingConfig, FullConfig, ModelConfig, TrainingConfig
from ncomp.model.transformer import CausalTransformer
from ncomp.pipeline import (
    HEADER_SIZE,
    MAGIC,
    NeuralCompressor,
    pack_header,
    parse_header,
)


def _tiny_compressor() -> NeuralCompressor:
    cfg = FullConfig(
        model=ModelConfig(
            vocab_size=256,
            context_length=8,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
            tie_weights=True,
        ),
        coding=CodingConfig(prob_total=1024, min_count=1),
        training=TrainingConfig(),
    )
    torch.manual_seed(0)
    model = CausalTransformer(cfg.model)
    return NeuralCompressor(model, cfg)


def _assert_round_trip(comp: NeuralCompressor, data: bytes) -> bytes:
    result = comp.compress(data)
    decoded = comp.decompress(result.payload)
    assert decoded == data, f"round-trip mismatch on {len(data)} bytes"
    return result.payload


def test_round_trip_empty() -> None:
    comp = _tiny_compressor()
    payload = _assert_round_trip(comp, b"")
    assert len(payload) == HEADER_SIZE


def test_round_trip_single_byte() -> None:
    comp = _tiny_compressor()
    for b in (0, 1, 65, 127, 200, 255):
        _assert_round_trip(comp, bytes([b]))


def test_round_trip_long_constant_run() -> None:
    comp = _tiny_compressor()
    _assert_round_trip(comp, b"A" * 2_000)
    _assert_round_trip(comp, b"\x00" * 1_500)
    _assert_round_trip(comp, b"\xff" * 1_500)


def test_round_trip_short_text() -> None:
    comp = _tiny_compressor()
    _assert_round_trip(comp, b"hello, world!\n")
    _assert_round_trip(comp, b"To be, or not to be.")


def test_round_trip_arbitrary_binary_bytes() -> None:
    comp = _tiny_compressor()
    data = bytes(range(256)) + bytes(range(255, -1, -1))
    _assert_round_trip(comp, data)


def test_round_trip_random_streams() -> None:
    comp = _tiny_compressor()
    rng = random.Random(0)
    for trial in range(8):
        n = rng.randint(1, 300)
        data = bytes(rng.randrange(256) for _ in range(n))
        _assert_round_trip(comp, data), f"failed trial {trial}"


def test_round_trip_input_longer_than_context() -> None:
    comp = _tiny_compressor()
    assert comp.config.model.context_length == 8
    rng = random.Random(2)
    data = bytes(rng.randrange(256) for _ in range(500))
    _assert_round_trip(comp, data)


def test_header_round_trip() -> None:
    fp = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    raw = pack_header(123_456, fp)
    header, payload = parse_header(raw + b"abc")
    assert header.uncompressed_length == 123_456
    assert header.fingerprint == fp
    assert payload == b"abc"


def test_parse_header_rejects_bad_magic() -> None:
    with pytest.raises(ValueError):
        parse_header(b"XXXX" + b"\x01" + b"\x00" * 16)


def test_parse_header_rejects_short_buffer() -> None:
    with pytest.raises(ValueError):
        parse_header(b"\x00")


def test_decompress_rejects_wrong_model() -> None:
    comp_a = _tiny_compressor()
    cfg = FullConfig(
        model=ModelConfig(
            vocab_size=256,
            context_length=8,
            d_model=16,
            n_layers=1,
            n_heads=2,
            d_ff=32,
            dropout=0.0,
        ),
        coding=CodingConfig(prob_total=1024, min_count=1),
    )
    torch.manual_seed(99)
    other_model = CausalTransformer(cfg.model)
    comp_b = NeuralCompressor(other_model, cfg)

    payload = comp_a.compress(b"hello").payload
    with pytest.raises(ValueError):
        comp_b.decompress(payload)


def test_compress_returns_well_formed_header() -> None:
    comp = _tiny_compressor()
    data = b"abc123"
    payload = comp.compress(data).payload
    assert payload[:4] == MAGIC
    header, _ = parse_header(payload)
    assert header.uncompressed_length == len(data)
    assert header.fingerprint == comp.fingerprint


def test_capture_surprise_returns_per_byte_bits() -> None:
    comp = _tiny_compressor()
    data = b"hello"
    result = comp.compress(data, capture_surprise=True)
    assert result.surprise_bits is not None
    assert len(result.surprise_bits) == len(data)
    assert all(s >= 0.0 for s in result.surprise_bits)


def test_round_trip_is_deterministic() -> None:
    comp = _tiny_compressor()
    data = b"deterministic" * 10
    p1 = comp.compress(data).payload
    p2 = comp.compress(data).payload
    assert p1 == p2


def test_progress_callback_called() -> None:
    comp = _tiny_compressor()
    seen: list[tuple[int, int]] = []
    comp.compress(b"abcdefghijklmnop" * 5, progress=lambda done, total: seen.append((done, total)))
    assert seen
    last = seen[-1]
    assert last[0] == last[1]
