from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import main

# Provider statistics for a just-finished fixture can appear with some delay.
# Persist a negative result, but allow a controlled recheck later instead of
# re-querying the same IDs on every catalog refresh/cold start.
NEGATIVE_ADVANCED_RECHECK_SECONDS = max(
    1800, int(main.os.getenv("NEGATIVE_ADVANCED_RECHECK_SECONDS", "21600"))
)

_original_hydrate = main.hydrate_finished_advanced


def _negative_is_fresh(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or payload.get("availability") != "UNAVAILABLE":
        return False
    checked_at = main.parse_dt(payload.get("checked_at"))
    if checked_at is None:
        return False
    return datetime.now(timezone.utc) - checked_at.astimezone(timezone.utc) < timedelta(
        seconds=NEGATIVE_ADVANCED_RECHECK_SECONDS
    )


async def hydrate_finished_advanced_incremental(
    client: main.httpx.AsyncClient,
    catalog: list[dict[str, Any]],
) -> int:
    finished_ids = {
        int((row.get("fixture") or {}).get("id") or 0)
        for row in catalog
        if main.fixture_state(row.get("fixture") or {})["is_finished"]
    }
    finished_ids.discard(0)
    if not finished_ids:
        return 0

    stored = await main.asyncio.to_thread(main.db_load_advanced_stats, list(finished_ids))

    # Supported rows never need hydration again. Negative rows are retried only
    # after the configured recheck window, so a provider delay can still heal.
    missing: list[int] = []
    for fixture_id in sorted(finished_ids):
        payload = stored.get(fixture_id)
        if payload is None:
            missing.append(fixture_id)
            continue
        if _negative_is_fresh(payload):
            continue
        if isinstance(payload, dict) and payload.get("availability") == "UNAVAILABLE":
            missing.append(fixture_id)

    if not missing:
        main.log.info(
            "Warehouse incremental: finished=%s provider_fetch=0 persisted=%s",
            len(finished_ids), len(stored),
        )
        return 0

    now = datetime.now(timezone.utc).isoformat()
    persist: dict[int, dict[str, Any]] = {}
    supported = 0
    unavailable = 0

    for batch in main.chunks(missing, 20):
        rows = await main.provider_get(client, "/fixtures", {"ids": "-".join(map(str, batch))})
        by_id = {
            int((row.get("fixture") or {}).get("id") or 0): row
            for row in (rows if isinstance(rows, list) else [])
            if int((row.get("fixture") or {}).get("id") or 0)
        }
        for fixture_id in batch:
            row = by_id.get(fixture_id)
            blocks = main.fixture_advanced_blocks(row) if row else {}
            if blocks:
                persist[fixture_id] = {
                    "teams": blocks,
                    "availability": "SUPPORTED",
                    "checked_at": now,
                }
                supported += 1
            else:
                persist[fixture_id] = {
                    "teams": {},
                    "availability": "UNAVAILABLE",
                    "checked_at": now,
                }
                unavailable += 1

    await main.asyncio.to_thread(main.db_save_advanced_stats, persist)
    main.log.info(
        "Warehouse incremental: finished=%s provider_fetch=%s supported=%s unavailable_cached=%s recheck_seconds=%s",
        len(finished_ids), len(missing), supported, unavailable, NEGATIVE_ADVANCED_RECHECK_SECONDS,
    )
    return supported


def install() -> None:
    main.hydrate_finished_advanced = hydrate_finished_advanced_incremental
    main.log.info(
        "Hydratacion incremental activa negative_recheck_seconds=%s",
        NEGATIVE_ADVANCED_RECHECK_SECONDS,
    )
