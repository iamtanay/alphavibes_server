"""
app/services/cache.py
─────────────────────────────────────────────────────────────────────────────
Redis cache layer.

Design:
  • Redis is OPTIONAL. If no REDIS_URL is set, the cache silently no-ops
    and every request hits yfinance directly. This means the app works
    perfectly on a laptop with zero infrastructure.
  • When Redis IS available (e.g. Upstash free tier on Railway), responses
    are cached to protect against yfinance rate-limiting and speed up
    repeated requests dramatically.
  • All values are stored as JSON strings.
  • Cache keys follow the pattern: av:{endpoint}:{param}

TTLs:
  • Full analysis:      15 minutes  (ANALYSIS_TTL)
  • Quote only:         5 minutes   (QUOTE_TTL)
  • Market overview:    5 minutes   (MARKET_TTL)
  • Screener:           24 hours    (SCREENER_TTL)
  • Search results:     1 hour      (SEARCH_TTL)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── TTLs (seconds) ────────────────────────────────────────────────────────
ANALYSIS_TTL = 15 * 60   # 15 minutes
QUOTE_TTL    = 5  * 60   # 5 minutes
MARKET_TTL   = 5  * 60   # 5 minutes
SCREENER_TTL = 24 * 60 * 60  # 24 hours
SEARCH_TTL   = 60 * 60   # 1 hour

# ── Redis client (lazy init) ──────────────────────────────────────────────
_redis = None
_redis_available = False


async def _get_redis():
    """Lazy-initialise Redis connection. Returns None if unavailable."""
    global _redis, _redis_available

    if _redis is not None:
        return _redis if _redis_available else None

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        logger.info("No REDIS_URL set — running without cache (OK for development)")
        _redis_available = False
        _redis = "no-op"
        return None

    try:
        import redis.asyncio as aioredis  # type: ignore
        client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        # Test connection
        await client.ping()
        _redis = client
        _redis_available = True
        logger.info("Redis connected: %s", redis_url[:30] + "...")
        return client
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — running without cache", exc)
        _redis = "no-op"
        _redis_available = False
        return None


# ── Public API ────────────────────────────────────────────────────────────

async def cache_get(key: str) -> dict | list | None:
    """
    Get a cached value by key.
    Returns parsed dict/list, or None if not found / cache unavailable.
    """
    client = await _get_redis()
    if client is None:
        return None

    try:
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Cache GET failed for %s: %s", key, exc)
        return None


async def cache_set(key: str, value: Any, ttl: int = ANALYSIS_TTL) -> None:
    """
    Store a value in cache with a TTL.
    Silently fails if cache is unavailable.
    """
    client = await _get_redis()
    if client is None:
        return

    try:
        serialised = json.dumps(value, default=str)
        await client.setex(key, ttl, serialised)
    except Exception as exc:
        logger.warning("Cache SET failed for %s: %s", key, exc)


async def cache_delete(key: str) -> None:
    """Delete a cache entry. Used for manual cache invalidation."""
    client = await _get_redis()
    if client is None:
        return

    try:
        await client.delete(key)
    except Exception as exc:
        logger.warning("Cache DELETE failed for %s: %s", key, exc)


async def cache_exists(key: str) -> bool:
    """Check if a key exists in cache."""
    client = await _get_redis()
    if client is None:
        return False

    try:
        return bool(await client.exists(key))
    except Exception:
        return False


# ── Convenience key builders ──────────────────────────────────────────────

def analysis_key(ticker: str) -> str:
    return f"av:analysis:{ticker.upper()}"

def quote_key(ticker: str) -> str:
    return f"av:quote:{ticker.upper()}"

def market_key() -> str:
    return "av:market:overview"

def screener_key(params_hash: str = "") -> str:
    return f"av:screener:{params_hash or 'all'}"

def search_key(query: str) -> str:
    return f"av:search:{query.lower()[:50]}"
