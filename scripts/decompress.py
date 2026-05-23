#!/usr/bin/env python3
"""Decompress a ``.ncz`` file produced by ``scripts/compress.py``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ncomp.model.checkpoint import load_checkpoint  # noqa: E402
from ncomp.pipeline import NeuralCompressor  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompress a Neural Compressor .ncz file")
    parser.add_argument("input", type=Path, help="The .ncz file to decompress")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to stripping the .ncz suffix)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoint.pt",
        help="Path to the model checkpoint (must match the one used to compress)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    if args.output is not None:
        output = args.output
    elif args.input.suffix == ".ncz":
        output = args.input.with_suffix("")
    else:
        output = args.input.with_suffix(args.input.suffix + ".out")

    payload = args.input.read_bytes()
    model, cfg, _ = load_checkpoint(args.checkpoint)
    compressor = NeuralCompressor(model, cfg)

    last_print = 0.0

    def progress(done: int, total: int) -> None:
        nonlocal last_print
        if args.quiet:
            return
        now = time.perf_counter()
        if now - last_print > 0.5 or done == total:
            pct = done / max(1, total) * 100.0
            print(
                f"\r  decompressing... {done:>8} / {total} bytes ({pct:5.1f}%)",
                end="",
                flush=True,
            )
            last_print = now

    t0 = time.perf_counter()
    data = compressor.decompress(payload, progress=None if args.quiet else progress)
    elapsed = time.perf_counter() - t0
    if not args.quiet:
        print()

    output.write_bytes(data)
    print(
        f"input  : {args.input}  {len(payload):,} bytes\n"
        f"output : {output}  {len(data):,} bytes\n"
        f"elapsed: {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
