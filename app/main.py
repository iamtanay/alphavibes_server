"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry point.

Configuration:
  • CORS: allows the Next.js frontend to call this API
  • Logging: structured, level controlled by LOG_LEVEL env var
  • Lifespan: warms up Redis connection on startup
  • Middleware: request timing, error logging

Environment variables (all optional with sensible defaults):
  REDIS_URL              Redis connection string (e.g. redis://localhost:6379)
                         Leave unset to run without cache (fine for development)
  ALLOWED_ORIGINS        Comma-separated CORS origins (default: localhost:3000)
  LOG_LEVEL              Logging level (default: INFO)
  RATE_LIMIT             Analyses per session window (default: 3)
  RATE_WINDOW_SECONDS    Rate limit window in seconds (default: 18000 = 5h)
  PORT                   Server port (default: 8000)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers.api import router

# ── Logging setup ─────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("alphavibes")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up the Redis connection so the first request isn't slow.
    Shutdown: nothing to clean up (Redis connections are async and GC'd).
    """
    logger.info("AlphaVibes backend starting up…")

    # Warm up Redis (no-op if REDIS_URL not set)
    from app.services.cache import _get_redis
    await _get_redis()

    logger.info("AlphaVibes backend ready.")
    yield
    logger.info("AlphaVibes backend shutting down.")


# ── App factory ───────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="AlphaVibes API",
        description="Stock analysis backend for AlphaVibes by Accrion",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    raw_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000",
    )
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    # In production, add your Vercel URL:
    # e.g. ALLOWED_ORIGINS=https://alphavibes.vercel.app
    vercel_url = os.getenv("VERCEL_URL", "")
    if vercel_url:
        allowed_origins.append(f"https://{vercel_url}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        """Add X-Process-Time header to all responses for performance monitoring."""
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled error in request: %s %s", request.method, request.url)
            response = JSONResponse(
                status_code=500,
                content={"error": "internal_error", "message": "An unexpected error occurred."},
            )
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.3f}s"

        # Log slow requests
        if elapsed > 10:
            logger.warning(
                "Slow request: %s %s took %.1fs",
                request.method, request.url.path, elapsed,
            )
        return response

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Global exception handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "An unexpected error occurred."},
        )

    return app


# ── Module-level app instance ─────────────────────────────────────────────
app = create_app()
