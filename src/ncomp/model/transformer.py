"""Small causal Transformer used as the next-byte probability model.

The architecture is intentionally compact: a few layers, small ``d_model``,
learned token and positional embeddings, pre-norm residual blocks, GELU MLP,
optional weight-tying between the embedding and output projection. It is
chosen to be CPU-trainable in tens of minutes while still beating gzip's
bits-per-character on natural-language text.

Determinism
-----------
The forward pass uses only standard PyTorch ops in ``float32`` on CPU and is
called with ``model.eval()`` and ``torch.no_grad()`` during compression and
decompression, so a given input produces the same logits on a given machine.
The compressor never depends on cross-machine bit-exact equality of float
operations: encoder and decoder are expected to run on the same host, and the
probability quantiser in :mod:`ncomp.model.probabilities` turns the floats
into integer counts before the coder sees them.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (upper triangle masked)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate_heads()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:t, :t], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    """Standard two-layer MLP with GELU activation."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CausalTransformer(nn.Module):
    """Next-token prediction model.

    Output has shape ``(batch, time, vocab_size)``. Position ``t`` of the
    output is the unnormalised logits for the token following position ``t``
    of the input.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if t > self.cfg.context_length:
            raise ValueError(f"input length {t} exceeds context_length {self.cfg.context_length}")
        pos = torch.arange(t, device=idx.device, dtype=torch.long)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
