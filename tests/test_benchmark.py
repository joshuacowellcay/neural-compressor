"""Tests for the benchmark runner."""

from __future__ import annotations

import torch

from ncomp.benchmark import (
    benchmark_bzip2,
    benchmark_gzip,
    benchmark_neural,
    benchmark_xz,
    format_table,
    run_all,
)
from ncomp.model.config import CodingConfig, FullConfig, ModelConfig
from ncomp.model.transformer import CausalTransformer
from ncomp.pipeline import NeuralCompressor


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
        ),
        coding=CodingConfig(prob_total=1024, min_count=1),
    )
    torch.manual_seed(0)
    return NeuralCompressor(CausalTransformer(cfg.model), cfg)


def test_gzip_round_trip() -> None:
    data = b"hello world" * 50
    r = benchmark_gzip(data)
    assert r.ok
    assert r.input_bytes == len(data)
    assert r.compressed_bytes > 0


def test_bzip2_round_trip() -> None:
    data = b"abcdefghij" * 50
    r = benchmark_bzip2(data)
    assert r.ok


def test_xz_round_trip() -> None:
    data = b"foo bar baz" * 50
    r = benchmark_xz(data)
    assert r.ok


def test_neural_round_trip() -> None:
    comp = _tiny_compressor()
    data = b"some bytes" * 30
    r = benchmark_neural(data, comp)
    assert r.ok
    assert r.input_bytes == len(data)


def test_run_all_returns_four_results() -> None:
    comp = _tiny_compressor()
    data = b"x" * 200
    rs = run_all(data, comp)
    assert [r.name for r in rs] == ["ncomp", "gzip -9", "bzip2 -9", "xz -9"]
    assert all(r.ok for r in rs)


def test_format_table_contains_all_names() -> None:
    comp = _tiny_compressor()
    data = b"y" * 200
    rs = run_all(data, comp)
    table = format_table(rs)
    for name in [r.name for r in rs]:
        assert name in table
    assert "bits/byte" in table


def test_result_derived_fields() -> None:
    data = b"a" * 1000
    r = benchmark_gzip(data)
    assert abs(r.bits_per_byte - (r.compressed_bytes * 8 / r.input_bytes)) < 1e-9
    assert r.saved_bytes == r.input_bytes - r.compressed_bytes
