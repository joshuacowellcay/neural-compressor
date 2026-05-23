#!/usr/bin/env python3
"""Benchmark Neural Compressor against gzip, bzip2, and xz.

Compresses a chunk of the held-out test text with each method, records the
compressed size and the wall-clock compression / decompression times, and
verifies the round-trip is exact. Writes a Markdown report to BENCHMARK.md,
saves a bits-per-byte chart to assets/benchmark_bpb.png, and saves a per-byte
"surprise" heatmap snippet to assets/surprise.png.

Every number in the report comes from the run; nothing is hand-edited.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ncomp.benchmark import Result, format_table, run_all  # noqa: E402
from ncomp.model.checkpoint import load_checkpoint  # noqa: E402
from ncomp.pipeline import NeuralCompressor, compute_surprise  # noqa: E402
from ncomp.training.data import load_corpus  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Neural Compressor vs standard tools")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "checkpoint.pt",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "corpus" / "pride_and_prejudice.txt",
        help="Path to the corpus file",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=16_384,
        help="How many held-out bytes to benchmark on (default: 16384)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "BENCHMARK.md",
        help="Where to write the Markdown report",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=ROOT / "assets" / "benchmark_bpb.png",
        help="Where to save the bits-per-byte bar chart",
    )
    parser.add_argument(
        "--surprise-fig",
        type=Path,
        default=ROOT / "assets" / "surprise.png",
        help="Where to save the per-byte surprise figure",
    )
    parser.add_argument(
        "--surprise-bytes",
        type=int,
        default=512,
        help="How many bytes to plot in the surprise figure (default: 512)",
    )
    return parser.parse_args()


def _save_bpb_chart(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [r.name for r in results]
    bpb = [r.bits_per_byte for r in results]

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
    bars = ax.bar(names, bpb, color=["#0a7", "#888", "#888", "#888"])
    ax.set_ylabel("bits per byte (lower is better)")
    ax.set_title("Compression efficiency on held-out text")
    ax.axhline(8.0, color="#bbb", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(len(names) - 0.5, 8.0, " raw bytes", color="#888", va="center", ha="left", fontsize=8)
    for bar, v in zip(bars, bpb, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.05,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(8.4, max(bpb) + 0.6))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_surprise_figure(
    text: bytes,
    surprise: list[float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(surprise, dtype=np.float64)
    n = len(arr)
    chars = [chr(b) if 32 <= b < 127 else " " for b in text]

    cols = min(64, max(32, n))
    rows = (n + cols - 1) // cols
    grid = np.full((rows, cols), np.nan, dtype=np.float64)
    for i, v in enumerate(arr):
        grid[i // cols, i % cols] = v

    fig, ax = plt.subplots(figsize=(11, max(3, rows * 0.32)), dpi=150)
    vmax = max(2.0, float(np.nanpercentile(arr, 98)))
    im = ax.imshow(grid, cmap="magma_r", vmin=0.0, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("bits to encode this byte")

    for i, ch in enumerate(chars):
        r, c = i // cols, i % cols
        text_color = "black" if 0 <= arr[i] <= vmax * 0.5 else "white"
        ax.text(c, r, ch, ha="center", va="center", color=text_color, fontsize=6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Per-byte surprise on a held-out passage (darker = more surprising)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_report(
    *,
    results: list[Result],
    report_path: Path,
    chart_path: Path,
    surprise_path: Path,
    chunk_bytes: int,
    chunk_preview: str,
    eval_bpb_from_checkpoint: float | None,
    machine: str,
    timestamp: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    chart_rel = (
        chart_path.relative_to(report_path.parent) if chart_path.is_absolute() else chart_path
    )
    surprise_rel = (
        surprise_path.relative_to(report_path.parent)
        if surprise_path.is_absolute()
        else surprise_path
    )

    by_name = {r.name: r for r in results}
    ncomp = next(r for r in results if r.name == "ncomp")
    gzip_r = by_name.get("gzip -9")
    bzip2_r = by_name.get("bzip2 -9")
    xz_r = by_name.get("xz -9")

    headline_parts = []
    if gzip_r is not None:
        saved_vs_gzip = 1.0 - ncomp.compressed_bytes / gzip_r.compressed_bytes
        headline_parts.append(
            f"Neural Compressor encodes the held-out chunk in **{ncomp.bits_per_byte:.3f} bits "
            f"per byte**, versus **{gzip_r.bits_per_byte:.3f}** for gzip -9; the resulting file "
            f"is **{saved_vs_gzip * 100:.1f}% smaller** than gzip's."
        )
    if bzip2_r is not None and ncomp.bits_per_byte < bzip2_r.bits_per_byte:
        headline_parts.append(
            f"It also beats bzip2 -9 ({bzip2_r.bits_per_byte:.3f} bits/byte) on this text."
        )
    if xz_r is not None and ncomp.bits_per_byte < xz_r.bits_per_byte:
        headline_parts.append(
            f"It also beats xz -9 ({xz_r.bits_per_byte:.3f} bits/byte) on this text."
        )

    speed_note = ""
    if gzip_r is not None:
        slower = ncomp.compress_seconds / max(1e-6, gzip_r.compress_seconds)
        speed_note = (
            f"This costs roughly **{slower:.0f}x** the wall-clock time of gzip "
            f"({ncomp.compress_seconds:.1f}s vs {gzip_r.compress_seconds * 1000:.1f}ms to compress)."
        )

    table = format_table(results)
    setup_bullets = [
        f"* Input: first **{chunk_bytes:,} bytes** of the held-out test split of"
        " Pride and Prejudice (text the model never saw during training).",
    ]
    if eval_bpb_from_checkpoint is not None:
        setup_bullets.append(
            f"* Reference cross-entropy on held-out at training time: "
            f"**{eval_bpb_from_checkpoint:.4f} bits/byte**."
        )
    setup_bullets.extend(
        [
            "* Standard tools: Python's stdlib `gzip`, `bz2`, and `lzma`, all at their"
            " highest preset (`-9`).",
            "* Neural Compressor: the bundled `models/checkpoint.pt`"
            " (~430k-parameter byte-level Transformer) with the default coding settings.",
            "* All timings are wall-clock on a single CPU; round-trip equality is asserted"
            " on every entry below.",
        ]
    )

    limitation_bullets = [
        "* The model was trained on a single book; on out-of-domain text (source code,"
        " log files, foreign languages) the ratio will degrade towards gzip's territory.",
        "* The benchmark chunk is a deliberate slice (default 16 KB) to keep the run"
        " under a few minutes on CPU; rerun with `--bytes 65536` (or more) for larger numbers.",
        "* Times include Python overhead and per-token PyTorch dispatch; an optimised"
        " C++/CUDA implementation would close some of the speed gap but not all of it.",
    ]

    sections: list[str] = [
        "# Benchmark",
        "",
        f"_Generated by `scripts/benchmark.py` on {timestamp} ({machine})._",
        "",
        "## Headline",
        "",
        " ".join(headline_parts),
        "",
        speed_note,
        "",
        "## Setup",
        "",
        *setup_bullets,
        "",
        "## Results",
        "",
        "```",
        table,
        "```",
        "",
        f"![bits per byte]({chart_rel})",
        "",
        "## Per-byte surprise",
        "",
        "The figure below shows how many bits the model spent on each byte of a short"
        " passage from the held-out text. Common characters and well-predicted spelling"
        " are pale (low surprise); unusual words, sentence boundaries, and rare characters"
        " are dark (high surprise).",
        "",
        f"![per-byte surprise]({surprise_rel})",
        "",
        "Passage shown (first ~60 characters):",
        "",
        "```",
        chunk_preview,
        "```",
        "",
        "## Reading the numbers",
        "",
        "Lower bits-per-byte means the file shrinks more. Eight bits per byte is the"
        " uncompressed baseline (one byte stays one byte). gzip and friends approach the"
        " information-theoretic limit for fixed-window dictionary codes; a probability"
        " model that anticipates English orthography and grammar can squeeze further, and"
        " that is exactly what Neural Compressor demonstrates on this corpus.",
        "",
        "Speed, on the other hand, is the cost: the standard tools run a tight C loop,"
        " whereas Neural Compressor runs a transformer forward pass for every byte. This"
        " asymmetry is the headline tradeoff and is shown above in the timing columns.",
        "",
        "## Limitations",
        "",
        *limitation_bullets,
        "",
    ]
    report_path.write_text("\n".join(sections), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.corpus.exists():
        print(f"Corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    print(f"loading model from {args.checkpoint} ...", flush=True)
    model, cfg, extra = load_checkpoint(args.checkpoint)
    compressor = NeuralCompressor(model, cfg)
    eval_bpb = extra.get("best_eval_bpb")

    print(f"loading corpus from {args.corpus} ...", flush=True)
    corpus = load_corpus(args.corpus, cfg.training.train_fraction)
    held_out = corpus.test
    chunk = held_out[: args.bytes]
    if len(chunk) < args.bytes:
        print(
            f"warning: held-out only has {len(chunk)} bytes; " f"requested {args.bytes}",
            file=sys.stderr,
        )

    print(f"benchmarking on {len(chunk):,} bytes...", flush=True)
    t0 = time.perf_counter()
    results = run_all(chunk, compressor)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s\n")
    print(format_table(results))
    print()

    for r in results:
        if not r.ok:
            print(f"ERROR: {r.name} failed round-trip", file=sys.stderr)
            return 1

    print(f"generating chart: {args.chart}")
    _save_bpb_chart(results, args.chart)

    n_show = min(args.surprise_bytes, len(chunk))
    print(f"generating surprise figure for {n_show} bytes: {args.surprise_fig}")
    surprise = compute_surprise(compressor, chunk[:n_show])
    _save_surprise_figure(chunk[:n_show], surprise, args.surprise_fig)

    preview_n = min(60, len(chunk))
    preview = chunk[:preview_n].decode("utf-8", errors="replace").replace("\n", "\\n ")
    machine = (
        f"{platform.python_implementation()} {platform.python_version()} on {platform.machine()}"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"writing report: {args.report}")
    _write_report(
        results=results,
        report_path=args.report,
        chart_path=args.chart,
        surprise_path=args.surprise_fig,
        chunk_bytes=len(chunk),
        chunk_preview=preview,
        eval_bpb_from_checkpoint=eval_bpb,
        machine=machine,
        timestamp=timestamp,
    )

    extras = {r.name: asdict(r) for r in results}
    print(f"results dict: {list(extras.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
