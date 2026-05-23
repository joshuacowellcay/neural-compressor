#!/usr/bin/env python3
"""Run the Neural Compressor FastAPI backend with uvicorn.

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000

Environment variables (all optional):
    NCOMP_MODEL_PATH         path to the model checkpoint
    NCOMP_MAX_UPLOAD_BYTES   override the upload size cap
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
BACKEND_PARENT = str(ROOT)
if BACKEND_PARENT not in sys.path:
    sys.path.insert(0, BACKEND_PARENT)

import uvicorn  # noqa: E402
from backend.app import create_app  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Neural Compressor backend")
    parser.add_argument(
        "--host",
        default=os.environ.get("NCOMP_BACKEND_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("NCOMP_BACKEND_PORT", "8000")),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
