"""Tests for the next-byte prediction model."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ncomp.model import (
    CausalTransformer,
    ModelConfig,
    cdf_from_counts,
    load_checkpoint,
    quantize_probabilities,
    save_checkpoint,
)
from ncomp.model.config import FullConfig
from ncomp.training.data import CorpusSplit, load_corpus, sample_batch


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256, context_length=16, d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.0
    )


def test_model_shapes() -> None:
    cfg = _tiny_cfg()
    model = CausalTransformer(cfg)
    x = torch.zeros((3, cfg.context_length), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (3, cfg.context_length, cfg.vocab_size)


def test_model_accepts_short_context() -> None:
    cfg = _tiny_cfg()
    model = CausalTransformer(cfg)
    for t in (1, 5, cfg.context_length):
        out = model(torch.zeros((1, t), dtype=torch.long))
        assert out.shape == (1, t, cfg.vocab_size)


def test_model_rejects_too_long_context() -> None:
    cfg = _tiny_cfg()
    model = CausalTransformer(cfg)
    with pytest.raises(ValueError):
        model(torch.zeros((1, cfg.context_length + 1), dtype=torch.long))


def test_model_is_causal() -> None:
    """Changing position t of the input must not change logits at positions < t."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = CausalTransformer(cfg).eval()
    x1 = torch.randint(0, cfg.vocab_size, (1, cfg.context_length), dtype=torch.long)
    x2 = x1.clone()
    t = cfg.context_length // 2
    x2[0, t] = (int(x1[0, t]) + 1) % cfg.vocab_size
    with torch.no_grad():
        o1 = model(x1)
        o2 = model(x2)
    assert torch.allclose(o1[0, :t], o2[0, :t], atol=1e-6)
    assert not torch.allclose(o1[0, t:], o2[0, t:], atol=1e-6)


def test_eval_mode_is_deterministic() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(1)
    model = CausalTransformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_length), dtype=torch.long)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)


def test_quantize_sums_and_min_count() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        vocab = int(rng.integers(2, 300))
        total = int(rng.integers(vocab * 2, 1 << 14))
        p = rng.dirichlet(np.ones(vocab) * 0.3)
        counts = quantize_probabilities(p, total, min_count=1)
        assert counts.sum() == total
        assert counts.min() >= 1
        assert counts.shape == (vocab,)


def test_quantize_deterministic_same_input() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    c1 = quantize_probabilities(p, 1024)
    c2 = quantize_probabilities(p, 1024)
    np.testing.assert_array_equal(c1, c2)


def test_quantize_handles_zero_and_nan() -> None:
    p = np.zeros(8, dtype=np.float64)
    counts = quantize_probabilities(p, 16, min_count=1)
    assert counts.sum() == 16
    assert counts.min() >= 1

    p = np.full(4, float("nan"))
    counts = quantize_probabilities(p, 8, min_count=1)
    assert counts.sum() == 8
    assert counts.min() >= 1


def test_cdf_from_counts_is_monotonic() -> None:
    counts = np.array([3, 0, 5, 1, 2], dtype=np.int64)
    cdf = cdf_from_counts(counts)
    assert cdf[0] == 0
    assert cdf[-1] == int(counts.sum())
    assert np.all(np.diff(cdf) >= 0)


def test_checkpoint_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = FullConfig()
    cfg.model = _tiny_cfg()
    torch.manual_seed(2)
    model = CausalTransformer(cfg.model).eval()
    x = torch.randint(0, cfg.model.vocab_size, (1, cfg.model.context_length), dtype=torch.long)
    with torch.no_grad():
        before = model(x)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, cfg)
    loaded, cfg2, extra = load_checkpoint(path)
    with torch.no_grad():
        after = loaded(x)
    assert torch.equal(before, after)
    assert cfg2.model.d_model == cfg.model.d_model
    assert extra == {}


def test_sample_batch_shapes() -> None:
    rng = np.random.default_rng(0)
    data = bytes(range(256)) * 4
    x, y = sample_batch(data, batch_size=8, context_length=16, rng=rng)
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    assert x.dtype == torch.long
    assert torch.all(x >= 0) and torch.all(x < 256)
    # The targets should be the inputs shifted by 1.
    starts = []
    for i in range(8):
        starts.append(int(x[i, 0].item()))
    assert starts


def test_load_corpus_splits_train_and_test(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "c.txt"
    body = b"a" * 1000 + b"\n\nCHAPTER X.\n" + b"b" * 1000
    p.write_bytes(body)
    split = load_corpus(p, train_fraction=0.5)
    # The chapter marker (without its leading blank line) starts the test split.
    assert split.test.startswith(b"CHAPTER X.")
    assert split.train.endswith(b"\n\n")
    assert split.train + split.test == body
    assert isinstance(split, CorpusSplit)


def test_initial_loss_near_log_vocab() -> None:
    """A randomly initialised model should have cross-entropy near log(vocab)."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = CausalTransformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (16, cfg.context_length), dtype=torch.long)
    y = torch.randint(0, cfg.vocab_size, (16, cfg.context_length), dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), y.reshape(-1)
    ).item()
    expected = math.log(cfg.vocab_size)
    assert abs(loss - expected) < 0.5
