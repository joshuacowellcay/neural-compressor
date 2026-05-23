"""End-to-end compression/decompression pipeline and file format."""

from .compressor import CompressionResult, NeuralCompressor, compute_surprise
from .file_format import (
    FINGERPRINT_BYTES,
    HEADER_SIZE,
    MAGIC,
    VERSION,
    FileHeader,
    compute_model_fingerprint,
    pack_header,
    parse_header,
)

__all__ = [
    "CompressionResult",
    "FINGERPRINT_BYTES",
    "FileHeader",
    "HEADER_SIZE",
    "MAGIC",
    "NeuralCompressor",
    "VERSION",
    "compute_model_fingerprint",
    "compute_surprise",
    "pack_header",
    "parse_header",
]
