"""Training loop for the next-byte prediction model.

Cross-entropy in nats is converted to bits-per-character so that the training
log reads in the same units the compressor is judged on: lower bits per
character means fewer bits to encode the data, which is what arithmetic
coding will achieve given these probabilities. Training cross-entropy in
bits is an upper bound on the achievable compressed size in bits per byte.

The loop is deliberately simple: random window sampling, AdamW, linear
warmup followed by cosine decay, gradient clipping, and periodic evaluation
on the held-out split. The best held-out bits-per-byte checkpoint is
returned so that the compressor uses the most generalisable model.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..model.checkpoint import save_checkpoint
from ..model.config import FullConfig
from ..model.transformer import CausalTransformer
from .data import CorpusSplit, sample_batch

LN2 = math.log(2.0)


@dataclass
class TrainStep:
    step: int
    train_bpb: float
    eval_bpb: float | None
    lr: float


@dataclass
class TrainResult:
    best_eval_bpb: float
    best_step: int
    elapsed_seconds: float
    history: list[TrainStep] = field(default_factory=list)


def _lr_schedule(step: int, n_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, n_steps - warmup)
    progress = min(1.0, progress)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for group in optim.param_groups:
        group["lr"] = lr


@torch.no_grad()
def _eval_bpb(
    model: CausalTransformer,
    data: bytes,
    batch_size: int,
    n_batches: int,
    context_length: int,
    rng: np.random.Generator,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for _ in range(n_batches):
        x, y = sample_batch(data, batch_size, context_length, rng)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += int(y.numel())
    model.train()
    nats_per_byte = total_loss / max(1, total_tokens)
    return nats_per_byte / LN2


def train(
    corpus: CorpusSplit,
    config: FullConfig,
    checkpoint_path: str | Path,
    log: Callable[[TrainStep], None] | None = None,
) -> TrainResult:
    """Train ``CausalTransformer`` on ``corpus`` and persist the best checkpoint."""
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    rng_train = np.random.default_rng(config.training.seed)
    rng_eval = np.random.default_rng(config.training.seed + 1)

    model = CausalTransformer(config.model)
    model.train()
    n_params = model.num_parameters()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )

    best_bpb = float("inf")
    best_step = 0
    history: list[TrainStep] = []
    start = time.perf_counter()

    for step in range(1, config.training.n_steps + 1):
        lr = _lr_schedule(
            step - 1,
            config.training.n_steps,
            config.training.warmup_steps,
            config.training.learning_rate,
        )
        _set_lr(optim, lr)

        x, y = sample_batch(
            corpus.train,
            config.training.batch_size,
            config.model.context_length,
            rng_train,
        )
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optim.step()

        eval_bpb: float | None = None
        do_eval = step % config.training.eval_every == 0 or step == config.training.n_steps
        if do_eval:
            eval_bpb = _eval_bpb(
                model,
                corpus.test,
                config.training.batch_size,
                config.training.eval_batches,
                config.model.context_length,
                rng_eval,
            )
            if eval_bpb < best_bpb:
                best_bpb = eval_bpb
                best_step = step
                save_checkpoint(
                    checkpoint_path,
                    model,
                    config,
                    extra={
                        "best_step": step,
                        "best_eval_bpb": best_bpb,
                        "n_params": n_params,
                    },
                )

        train_bpb = float(loss.item()) / LN2
        record = TrainStep(step=step, train_bpb=train_bpb, eval_bpb=eval_bpb, lr=lr)
        history.append(record)
        if log is not None:
            log(record)

    elapsed = time.perf_counter() - start
    return TrainResult(
        best_eval_bpb=best_bpb,
        best_step=best_step,
        elapsed_seconds=elapsed,
        history=history,
    )
