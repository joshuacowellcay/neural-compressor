#!/usr/bin/env python3
"""Train the next-byte prediction model and save the best checkpoint.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/small.json --corpus corpus/pride_and_prejudice.txt \
        --checkpoint models/checkpoint.pt

Defaults are configured for the small CPU-friendly model in
``configs/small.json``. Training prints train and held-out bits per byte
periodically; the best held-out checkpoint is written to ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the package importable when running from a fresh clone without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ncomp.model.config import FullConfig  # noqa: E402
from ncomp.training.data import load_corpus  # noqa: E402
from ncomp.training.loop import TrainStep, train  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Neural Compressor model")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "small.json",
        help="Path to JSON config",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "corpus" / "pride_and_prejudice.txt",
        help="Path to the corpus file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoint.pt",
        help="Where to write the best checkpoint",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override the number of training steps from the config",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce stdout output",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2
    if not args.corpus.exists():
        print(f"Corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    config = FullConfig.from_json(args.config)
    if args.n_steps is not None:
        config.training.n_steps = args.n_steps

    corpus = load_corpus(args.corpus, config.training.train_fraction)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    last_print = time.perf_counter()

    def log(step: TrainStep) -> None:
        nonlocal last_print
        now = time.perf_counter()
        if args.quiet:
            return
        is_eval = step.eval_bpb is not None
        is_last = step.step == config.training.n_steps
        if is_eval or is_last or (now - last_print) > 10:
            eval_str = f"eval={step.eval_bpb:.4f}" if is_eval else "eval=-"
            print(
                f"step {step.step:5d}/{config.training.n_steps} "
                f"train={step.train_bpb:.4f} {eval_str} lr={step.lr:.2e}",
                flush=True,
            )
            last_print = now

    if not args.quiet:
        print(
            f"corpus train={len(corpus.train):,}B test={len(corpus.test):,}B  "
            f"steps={config.training.n_steps} batch={config.training.batch_size} "
            f"ctx={config.model.context_length}",
            flush=True,
        )

    result = train(corpus, config, args.checkpoint, log=log)

    print(
        f"best eval bits/byte = {result.best_eval_bpb:.4f} at step {result.best_step} "
        f"(elapsed {result.elapsed_seconds:.1f}s)"
    )
    print(f"checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
