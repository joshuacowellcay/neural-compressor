"""FastAPI application factory for the Neural Compressor backend."""

from __future__ import annotations

from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import CompressionResponse, HealthResponse, InfoResponse
from .state import AppState, BackendSettings


def create_app(state: AppState | None = None) -> FastAPI:
    state = state or AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        # Tolerate a missing checkpoint so /api/health can still report the state.
        with suppress(FileNotFoundError):
            state.load()
        yield

    app = FastAPI(title="Neural Compressor API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.ncomp = state  # type: ignore[attr-defined]

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(ok=True, model_loaded=state.is_loaded())

    @app.get("/api/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        return state.info()

    @app.post("/api/compress", response_model=CompressionResponse)
    async def compress(file: UploadFile = File(...)) -> CompressionResponse:
        if not state.is_loaded():
            raise HTTPException(503, "model is not loaded")
        data = await file.read()
        if len(data) == 0:
            raise HTTPException(400, "empty upload")
        if len(data) > state.settings.max_upload_bytes:
            raise HTTPException(
                413,
                f"file too large: {len(data)} bytes (max {state.settings.max_upload_bytes})",
            )
        return state.compress_with_comparison(data)

    @app.post("/api/decompress")
    async def decompress(file: UploadFile = File(...)) -> Response:
        if not state.is_loaded():
            raise HTTPException(503, "model is not loaded")
        data = await file.read()
        if len(data) == 0:
            raise HTTPException(400, "empty upload")
        if len(data) > state.settings.max_upload_bytes * 4:
            raise HTTPException(
                413,
                f"compressed payload too large: {len(data)} bytes",
            )
        try:
            out = state.decompress(data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return Response(
            content=out,
            media_type="application/octet-stream",
            headers={"X-Original-Length": str(len(out))},
        )

    return app


__all__ = ["create_app", "AppState", "BackendSettings"]
