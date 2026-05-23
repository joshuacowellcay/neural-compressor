"""Honest benchmark of Neural Compressor against gzip, bzip2, and xz."""

from .runner import (
    Result,
    benchmark_bzip2,
    benchmark_gzip,
    benchmark_neural,
    benchmark_xz,
    format_table,
    run_all,
)

__all__ = [
    "Result",
    "benchmark_bzip2",
    "benchmark_gzip",
    "benchmark_neural",
    "benchmark_xz",
    "format_table",
    "run_all",
]
