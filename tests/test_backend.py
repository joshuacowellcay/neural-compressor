"""Integration tests for the FastAPI backend."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from backend.app import create_app
from backend.state import AppState, BackendSettings
from fastapi.testclient import TestClient

from ncomp.model.checkpoint import save_checkpoint
from ncomp.model.config import CodingConfig, FullConfig, ModelConfig, TrainingConfig
from ncomp.model.transformer import CausalTransformer


def _write_tiny_checkpoint(path: Path) -> None:
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
        training=TrainingConfig(),
    )
    torch.manual_seed(7)
    model = CausalTransformer(cfg.model)
    save_checkpoint(
        path, model, cfg, extra={"best_eval_bpb": 4.2, "n_params": model.num_parameters()}
    )


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    ckpt = tmp_path / "ckpt.pt"
    _write_tiny_checkpoint(ckpt)
    state = AppState(BackendSettings(checkpoint_path=ckpt, max_upload_bytes=1024))
    app = create_app(state=state)
    with TestClient(app) as c:
        yield c


def test_health_reports_loaded(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model_loaded"] is True


def test_info_returns_model_metadata(client: TestClient) -> None:
    r = client.get("/api/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["context_length"] == 8
    assert body["d_model"] == 16
    assert isinstance(body["fingerprint_hex"], str)
    assert body["max_upload_bytes"] == 1024


def test_compress_returns_comparison_and_surprise(client: TestClient) -> None:
    data = b"To be, or not to be: that is the question." * 4
    files = {"file": ("sample.txt", io.BytesIO(data), "application/octet-stream")}
    r = client.post("/api/compress", files=files)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["input"]["size_bytes"] == len(data)
    assert "To be" in body["input"]["preview"]
    names = [row["name"] for row in body["results"]]
    assert names == ["ncomp", "gzip -9", "bzip2 -9", "xz -9"]
    for row in body["results"]:
        assert row["compressed_bytes"] > 0
        assert row["bits_per_byte"] > 0
        assert row["ok"] is True
    assert len(body["surprise"]["bits"]) == len(data)
    assert body["surprise"]["bytes"][0] == data[0]


def test_compress_rejects_empty_upload(client: TestClient) -> None:
    files = {"file": ("empty.txt", io.BytesIO(b""), "application/octet-stream")}
    r = client.post("/api/compress", files=files)
    assert r.status_code == 400


def test_compress_rejects_too_large_upload(client: TestClient) -> None:
    data = b"x" * 2048  # cap is 1024 in the fixture
    files = {"file": ("big.txt", io.BytesIO(data), "application/octet-stream")}
    r = client.post("/api/compress", files=files)
    assert r.status_code == 413


def test_decompress_round_trip(client: TestClient) -> None:
    data = b"Round-trip me please." * 10
    files = {"file": ("a.txt", io.BytesIO(data), "application/octet-stream")}
    c = client.post("/api/compress", files=files)
    assert c.status_code == 200

    # Re-compress on the server-side (the API does not return the bitstream itself),
    # so to test decompress we feed it a payload built by re-running the compressor
    # via a second endpoint call: we ask the app to round-trip a tiny stream we
    # built directly through the underlying state to keep this self-contained.
    state: AppState = client.app.state.ncomp  # type: ignore[attr-defined]
    payload = state.compressor.compress(data).payload  # type: ignore[union-attr]

    files = {"file": ("a.ncz", io.BytesIO(payload), "application/octet-stream")}
    r = client.post("/api/decompress", files=files)
    assert r.status_code == 200
    assert r.content == data
    assert r.headers["X-Original-Length"] == str(len(data))


def test_decompress_rejects_wrong_model(client: TestClient) -> None:
    files = {
        "file": ("bad.ncz", io.BytesIO(b"NCMP\x01" + b"\x00" * 16), "application/octet-stream")
    }
    r = client.post("/api/decompress", files=files)
    assert r.status_code == 400
