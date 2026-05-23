#!/usr/bin/env python3
"""Compress a file using the trained neural compressor.

Usage:
    python scripts/compress.py input.txt -o input.txt.ncz
    python scripts/compress.py input.txt    # writes input.txt.ncz next to the input
"""

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
    parser = argparse.ArgumentParser(description="Compress a file with Neural Compressor")
    parser.add_argument("input", type=Path, help="Input file to compress")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to <input>.ncz)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoint.pt",
        help="Path to the model checkpoint",
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

    output = (
        args.output
        if args.output is not None
        else args.input.with_suffix(args.input.suffix + ".ncz")
    )

    data = args.input.read_bytes()
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
            print(f"\r  compressing... {done:>8} / {total} bytes ({pct:5.1f}%)", end="", flush=True)
            last_print = now

    t0 = time.perf_counter()
    result = compressor.compress(data, progress=None if args.quiet else progress)
    elapsed = time.perf_counter() - t0
    if not args.quiet:
        print()

    output.write_bytes(result.payload)

    ratio = len(result.payload) / max(1, len(data))
    bpc = len(result.payload) * 8 / max(1, len(data))
    print(
        f"input  : {args.input}  {len(data):,} bytes\n"
        f"output : {output}  {len(result.payload):,} bytes\n"
        f"ratio  : {ratio:.3f}  ({bpc:.3f} bits/byte)  in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
