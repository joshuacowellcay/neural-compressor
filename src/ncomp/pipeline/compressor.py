"""Neural-network-driven arithmetic compression pipeline.

This module wires the trained next-byte prediction model to the integer
arithmetic coder. Both the compressor and decompressor follow the same loop
in lockstep:

1. Look at the most recent ``context_length`` bytes already encoded /
   decoded.
2. Run the model on that context to get logits for the next byte.
3. Quantise the resulting probability vector to integer counts (the
   :mod:`ncomp.model.probabilities` module).
4. Use the resulting CDF with the arithmetic encoder / decoder.

Because the model is in ``eval()`` mode, runs in ``float32`` on CPU, and is
loaded from the same checkpoint on both sides, encoder and decoder produce
byte-identical CDFs at every step. Combined with the integer arithmetic
coder, the round-trip is exact for any input.

The compressor handles three edge cases explicitly:

* Empty input: writes a header with length zero and an empty payload.
* Inputs shorter than ``context_length``: feeds whatever bytes are available
  so far as context (the model is causal, so this works at any length from 1
  upwards).
* Inputs longer than ``context_length``: slides the context window so the
  model always sees the most recent ``context_length`` bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from ..coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    BitReader,
    BitWriter,
    decode_with_cdf,
    encode_with_cdf,
)
from ..model.config import CodingConfig, FullConfig
from ..model.probabilities import cdf_from_counts, quantize_probabilities
from ..model.transformer import CausalTransformer
from .file_format import compute_model_fingerprint, pack_header, parse_header


@dataclass(frozen=True)
class CompressionResult:
    """Outputs of a single ``compress`` call.

    ``payload`` is the full ``.ncz`` byte string (header + bitstream). The
    optional ``surprise_bits`` list reports ``-log2(p(byte))`` for each
    encoded byte; this drives the per-byte heatmap shown in the frontend.
    """

    payload: bytes
    surprise_bits: list[float] | None


def _uniform_cdf(coding: CodingConfig, vocab_size: int) -> np.ndarray:
    """CDF for a uniform distribution, used for the very first byte."""
    counts = quantize_probabilities(
        np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64),
        coding.prob_total,
        coding.min_count,
    )
    return cdf_from_counts(counts)


def _context_for(data: bytes, end_pos: int, ctx_len: int) -> bytes:
    return data[max(0, end_pos - ctx_len) : end_pos]


class NeuralCompressor:
    """Compress and decompress bytes through a neural probability model."""

    def __init__(self, model: CausalTransformer, config: FullConfig) -> None:
        self.model = model
        self.model.eval()
        self.config = config
        config.model.validate_heads()
        config.coding.validate_for_vocab(config.model.vocab_size)
        state = model.state_dict()
        self._fingerprint = compute_model_fingerprint(config.model_dump(), state)
        self._uniform_cdf = _uniform_cdf(config.coding, config.model.vocab_size)

    @property
    def fingerprint(self) -> bytes:
        return self._fingerprint

    def _cdf_for_context(self, context: bytes) -> np.ndarray:
        if len(context) == 0:
            return self._uniform_cdf
        x = torch.tensor([list(context)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        counts = quantize_probabilities(
            probs, self.config.coding.prob_total, self.config.coding.min_count
        )
        return cdf_from_counts(counts)

    def compress(
        self,
        data: bytes,
        capture_surprise: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> CompressionResult:
        """Compress ``data`` and return its ``.ncz`` payload."""
        ctx_len = self.config.model.context_length
        n = len(data)

        surprise: list[float] | None = [] if capture_surprise else None
        if n == 0:
            header = pack_header(0, self._fingerprint)
            return CompressionResult(payload=header, surprise_bits=surprise)

        writer = BitWriter()
        encoder = ArithmeticEncoder(writer)
        for t in range(n):
            ctx = _context_for(data, t, ctx_len)
            cdf = self._cdf_for_context(ctx)
            byte = data[t]
            encode_with_cdf(encoder, byte, cdf)
            if surprise is not None:
                count = int(cdf[byte + 1] - cdf[byte])
                p = count / int(cdf[-1])
                surprise.append(-float(np.log2(p)))
            if progress is not None and (t % 256 == 0 or t == n - 1):
                progress(t + 1, n)

        encoder.finish()
        bitstream = writer.flush()
        header = pack_header(n, self._fingerprint)
        return CompressionResult(payload=header + bitstream, surprise_bits=surprise)

    def decompress(
        self,
        payload: bytes,
        progress: Callable[[int, int], None] | None = None,
    ) -> bytes:
        """Decompress an ``.ncz`` payload back to its original bytes."""
        header, body = parse_header(payload)
        if header.fingerprint != self._fingerprint:
            raise ValueError(
                "model fingerprint mismatch: payload was produced by a different model"
            )

        ctx_len = self.config.model.context_length
        n = header.uncompressed_length

        if n == 0:
            return b""

        reader = BitReader(body)
        decoder = ArithmeticDecoder(reader)
        out = bytearray()

        for t in range(n):
            ctx = bytes(out[max(0, t - ctx_len) : t])
            cdf = self._cdf_for_context(ctx)
            byte = decode_with_cdf(decoder, cdf)
            out.append(byte)
            if progress is not None and (t % 256 == 0 or t == n - 1):
                progress(t + 1, n)

        return bytes(out)


def compute_surprise(
    compressor: NeuralCompressor,
    data: bytes,
    positions: Iterable[int] | None = None,
) -> list[float]:
    """Return ``-log2(p)`` for each byte of ``data`` under ``compressor``.

    Faster than calling :meth:`NeuralCompressor.compress` when only the
    surprise values are needed, because it skips bit writing.
    """
    out: list[float] = []
    ctx_len = compressor.config.model.context_length
    indices = range(len(data)) if positions is None else positions
    for t in indices:
        ctx = _context_for(data, t, ctx_len)
        cdf = compressor._cdf_for_context(ctx)  # noqa: SLF001 (intentional internal call)
        byte = data[t]
        count = int(cdf[byte + 1] - cdf[byte])
        p = count / int(cdf[-1])
        out.append(-float(np.log2(p)))
    return out
