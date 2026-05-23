"""Benchmark Neural Compressor against gzip, bzip2, and xz.

Each method compresses and decompresses the same bytes; we record:

* compressed size (bytes)
* compression ratio (compressed / original)
* bits per byte ( = 8 * ratio )
* compression time and decompression time, both measured with
  :func:`time.perf_counter`
* a flag indicating the round-trip is exact

Round-trip equality is asserted explicitly; if any compressor fails it, the
result is marked ``ok=False``. The neural compressor must of course pass,
and the standard tools are losslesss by construction, so all four are
expected to be ``ok=True`` on every input.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import time
from dataclasses import dataclass

from ..pipeline import NeuralCompressor


@dataclass(frozen=True)
class Result:
    name: str
    input_bytes: int
    compressed_bytes: int
    compress_seconds: float
    decompress_seconds: float
    ok: bool

    @property
    def ratio(self) -> float:
        return self.compressed_bytes / max(1, self.input_bytes)

    @property
    def bits_per_byte(self) -> float:
        return self.compressed_bytes * 8 / max(1, self.input_bytes)

    @property
    def saved_bytes(self) -> int:
        return self.input_bytes - self.compressed_bytes

    @property
    def compress_mb_per_s(self) -> float:
        return self.input_bytes / 1_048_576 / max(1e-9, self.compress_seconds)

    @property
    def decompress_mb_per_s(self) -> float:
        return self.input_bytes / 1_048_576 / max(1e-9, self.decompress_seconds)


def _time_call(fn):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def benchmark_gzip(data: bytes, level: int = 9) -> Result:
    compressed, c = _time_call(lambda: gzip.compress(data, compresslevel=level, mtime=0))
    restored, d = _time_call(lambda: gzip.decompress(compressed))
    return Result(
        name=f"gzip -{level}",
        input_bytes=len(data),
        compressed_bytes=len(compressed),
        compress_seconds=c,
        decompress_seconds=d,
        ok=restored == data,
    )


def benchmark_bzip2(data: bytes, level: int = 9) -> Result:
    compressed, c = _time_call(lambda: bz2.compress(data, compresslevel=level))
    restored, d = _time_call(lambda: bz2.decompress(compressed))
    return Result(
        name=f"bzip2 -{level}",
        input_bytes=len(data),
        compressed_bytes=len(compressed),
        compress_seconds=c,
        decompress_seconds=d,
        ok=restored == data,
    )


def benchmark_xz(data: bytes, preset: int = 9) -> Result:
    compressed, c = _time_call(lambda: lzma.compress(data, preset=preset))
    restored, d = _time_call(lambda: lzma.decompress(compressed))
    return Result(
        name=f"xz -{preset}",
        input_bytes=len(data),
        compressed_bytes=len(compressed),
        compress_seconds=c,
        decompress_seconds=d,
        ok=restored == data,
    )


def benchmark_neural(data: bytes, compressor: NeuralCompressor) -> Result:
    compressed_result, c = _time_call(lambda: compressor.compress(data))
    compressed = compressed_result.payload
    restored, d = _time_call(lambda: compressor.decompress(compressed))
    return Result(
        name="ncomp",
        input_bytes=len(data),
        compressed_bytes=len(compressed),
        compress_seconds=c,
        decompress_seconds=d,
        ok=restored == data,
    )


def run_all(data: bytes, compressor: NeuralCompressor) -> list[Result]:
    """Run every benchmark on ``data`` and return their results in stable order."""
    return [
        benchmark_neural(data, compressor),
        benchmark_gzip(data),
        benchmark_bzip2(data),
        benchmark_xz(data),
    ]


def format_table(results: list[Result]) -> str:
    """Pretty-print a comparison table for stdout / BENCHMARK.md."""
    rows = [
        ("tool", "compressed", "ratio", "bits/byte", "compress", "decompress", "ok"),
        ("----", "----------", "-----", "---------", "--------", "----------", "--"),
    ]
    for r in results:
        rows.append(
            (
                r.name,
                f"{r.compressed_bytes:,}",
                f"{r.ratio:.4f}",
                f"{r.bits_per_byte:.4f}",
                f"{r.compress_seconds:.2f}s",
                f"{r.decompress_seconds:.2f}s",
                "yes" if r.ok else "NO",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)
