# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import (
    descriptors,
    examples,
    fs,
    help,
    home,
    mcp_servers,
    model_config,
    plugins,
    processors,
    providers,
    run,
    schema,
    sessions,
    skills,
    tools,
    validate,
    vendors,
)


def create_app(serve_static: bool = True) -> FastAPI:
    """Create and configure the Harness Lab FastAPI app.

    Args:
        serve_static: If True, mount ``frontend/dist/`` as static files at ``/``.
                      Set False in dev mode (Vite dev server handles static content).
    """
    try:
        from harnessx.core.config_store import register_harnessx_configs

        register_harnessx_configs()
    except Exception:
        pass

    app = FastAPI(
        title="Harness Lab",
        description="Zero-code harness composition and comparison UI",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes — all prefixed with /api
    for mod in (
        schema,
        descriptors,
        examples,
        run,
        sessions,
        validate,
        providers,
        vendors,
        model_config,
        tools,
        skills,
        fs,
        mcp_servers,
        plugins,
        processors,
        home,
        help,
    ):
        app.include_router(mod.router, prefix="/api")

    if serve_static:
        _mount_frontend(app)

    @app.on_event("shutdown")
    async def _shutdown_active_runs() -> None:
        await run.shutdown_active_runs()

    return app


def _resolve_dist_file(dist: Path, full_path: str) -> Path | None:
    """Resolve a request path to a concrete static file under frontend dist.

    Returns None when the requested path does not exist, points to a directory,
    or escapes the dist root via path traversal.
    """
    rel = full_path.lstrip("/")
    if not rel:
        return None
    dist_root = dist.resolve()
    candidate = (dist_root / rel).resolve()
    try:
        candidate.relative_to(dist_root)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _mount_frontend(app: FastAPI) -> None:
    dist = Path(__file__).parents[2] / "frontend" / "dist"
    if not dist.exists():
        return  # silently skip if frontend hasn't been built yet

    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        """Serve static dist files first; fallback to index.html for SPA routes."""
        static_file = _resolve_dist_file(dist, full_path)
        if static_file is not None:
            return FileResponse(static_file)
        return FileResponse(dist / "index.html")
