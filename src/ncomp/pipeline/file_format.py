"""On-disk format for ``.ncz`` compressed files.

The format is intentionally minimal: a magic number, a version byte, the
original (uncompressed) length, an 8-byte model fingerprint, and the raw
bitstream payload from the arithmetic encoder.

The fingerprint is a truncated SHA-256 of the canonical model config plus the
serialised state dict. It lets the decompressor refuse to decode a payload
that was produced by a different model, which is the only way a wrong model
manifests (it would otherwise produce different probabilities at every step
and silently corrupt the output).
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import dataclass
from typing import Any

import torch

MAGIC = b"NCMP"
VERSION = 1
FINGERPRINT_BYTES = 8
HEADER_FORMAT = ">4sBQ"  # magic, version, uncompressed length
HEADER_BASE_SIZE = struct.calcsize(HEADER_FORMAT)
HEADER_SIZE = HEADER_BASE_SIZE + FINGERPRINT_BYTES


@dataclass(frozen=True)
class FileHeader:
    """Parsed contents of a ``.ncz`` header."""

    version: int
    uncompressed_length: int
    fingerprint: bytes

    def __post_init__(self) -> None:
        if self.version != VERSION:
            raise ValueError(f"unsupported version: {self.version}")
        if self.uncompressed_length < 0:
            raise ValueError(f"negative length: {self.uncompressed_length}")
        if len(self.fingerprint) != FINGERPRINT_BYTES:
            raise ValueError(
                f"fingerprint must be {FINGERPRINT_BYTES} bytes, got {len(self.fingerprint)}"
            )


def pack_header(uncompressed_length: int, fingerprint: bytes) -> bytes:
    """Serialise a header."""
    if uncompressed_length < 0:
        raise ValueError("uncompressed_length must be non-negative")
    if len(fingerprint) != FINGERPRINT_BYTES:
        raise ValueError(f"fingerprint must be {FINGERPRINT_BYTES} bytes")
    return struct.pack(HEADER_FORMAT, MAGIC, VERSION, uncompressed_length) + fingerprint


def parse_header(buf: bytes) -> tuple[FileHeader, bytes]:
    """Parse a ``.ncz`` header from ``buf`` and return ``(header, payload)``."""
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"buffer too short ({len(buf)} bytes) for header")
    magic, version, length = struct.unpack(HEADER_FORMAT, buf[:HEADER_BASE_SIZE])
    if magic != MAGIC:
        raise ValueError(f"bad magic: expected {MAGIC!r}, got {magic!r}")
    fingerprint = bytes(buf[HEADER_BASE_SIZE:HEADER_SIZE])
    return (
        FileHeader(version=version, uncompressed_length=length, fingerprint=fingerprint),
        buf[HEADER_SIZE:],
    )


def compute_model_fingerprint(config_dict: dict[str, Any], state_dict: dict[str, Any]) -> bytes:
    """Compute an 8-byte fingerprint of a model so encoder and decoder can verify a match."""
    digest = hashlib.sha256()
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest.update(b"config:" + canonical)
    digest.update(b"\nstate:")
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            buf = io.BytesIO()
            torch.save(tensor.detach().cpu(), buf)
            tensor_bytes = buf.getvalue()
        else:
            tensor_bytes = repr(tensor).encode("utf-8")
        digest.update(key.encode("utf-8") + b":" + tensor_bytes + b"\n")
    return digest.digest()[:FINGERPRINT_BYTES]
