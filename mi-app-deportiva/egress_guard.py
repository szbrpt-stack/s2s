"""Runtime egress guard for heavy historical reports.

Preserves the existing response contracts while preventing repeated 5k-row
historical reads from PostgreSQL. The first request still uses the legacy
implementation; subsequent requests within the TTL reuse the in-process result.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import main

TTL_SECONDS = max(300, min(int(main.os.getenv("REPORT_CACHE_TTL_SECONDS", "3600")), 21600))
_MAX_DAILY_KEYS = 32
_lock = threading.Lock()
_cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
_installed = False


def _cached(key: tuple[Any, ...], producer: Callable[[], Any]) -> Any:
    now = time.monotonic()
    with _lock:
        item = _cache.get(key)
        if item and item[0] > now:
            return item[1]
        if item:
            _cache.pop(key, None)
    value = producer()
    with _lock:
        _cache[key] = (now + TTL_SECONDS, value)
        # Keep bounded memory even if callers request many date combinations.
        daily_keys = [k for k in _cache if k and k[0] == "daily"]
        if len(daily_keys) > _MAX_DAILY_KEYS:
            for old_key in daily_keys[:-_MAX_DAILY_KEYS]:
                _cache.pop(old_key, None)
    return value


def install() -> None:
    global _installed
    if _installed:
        return
    original_scorecard = main.db_model_scorecard
    original_daily = main.db_daily_performance

    def scorecard_cached() -> dict[str, Any]:
        return _cached(("scorecard", main.MODEL_VERSION), original_scorecard)

    def daily_cached(date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
        return _cached(("daily", date_from, date_to), lambda: original_daily(date_from, date_to))

    main.db_model_scorecard = scorecard_cached
    main.db_daily_performance = daily_cached
    _installed = True
    main.log.info("Egress guard active report_cache_ttl=%ss", TTL_SECONDS)
