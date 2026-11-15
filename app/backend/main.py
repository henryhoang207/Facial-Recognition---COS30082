"""FastAPI app factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import rest, stream
from .config import REPO_ROOT
from .context import AppContext

FRONTEND_DIST = REPO_ROOT / "app" / "frontend" / "dist"


def create_app(config_path: str | Path | None = None, ctx: AppContext | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # nạp model một lần khi server khởi động
        app.state.ctx = ctx or AppContext.from_file(config_path)
        yield

    app = FastAPI(title="FaceGuard AI", version="0.1.0", lifespan=lifespan)
    # cho phép Vite dev server (npm run dev) gọi API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(rest.router)
    app.include_router(stream.router)
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app
