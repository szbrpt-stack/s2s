from __future__ import annotations

import asyncio
from typing import Any

import httpx
import main

# Smaller sets persist sooner and cap transient memory. Provider pressure remains
# governed by the existing global rate limiter and HTTP concurrency controls.
PROGRESSIVE_BATCH_SIZE = max(4, min(int(main.os.getenv("PROGRESSIVE_BATCH_SIZE", "8")), 24))


async def analyze_catalog_progressive(
    fixtures: list[dict[str, Any]], fixture_date: str,
    snapshots: dict[str, dict[str, Any]] | None = None,
    errors: dict[str, str] | None = None,
    progress: dict[str, Any] | None = None,
    save_coverage: bool = True,
) -> None:
    snapshot_state = snapshots if snapshots is not None else main._snapshots
    error_state = errors if errors is not None else main._analysis_errors
    progress_state = progress if progress is not None else main._progress

    candidates: list[dict[str, Any]] = []
    for fixture in fixtures:
        shell = main.base_shell(fixture)
        snapshot = snapshot_state.get(shell["fixture_id"])
        outdated = bool(snapshot and snapshot.get("model_version") != main.MODEL_VERSION)
        if shell["is_upcoming"] and (snapshot is None or outdated):
            candidates.append(fixture)
    candidates.sort(key=lambda fixture: main.fixture_state(fixture.get("fixture") or {})["timestamp"])

    total = len(candidates)
    work_total = max(total * 3, 1)
    progress_state.update(
        phase="PROGRESSIVE_ANALYSIS", done=0, total=work_total,
        overall_done=0.0, overall_total=float(work_total),
        started_at=main.datetime.now(main.UTC).isoformat(),
        fixtures_done=0, fixtures_total=total,
    )
    if not candidates:
        progress_state.update(phase="COMPLETE", done=1, total=1, overall_done=1.0, overall_total=1.0,
                              fixtures_done=0, fixtures_total=0)
        if save_coverage:
            await asyncio.to_thread(main.db_save_coverage_manifest, main.coverage_payload())
        return

    progress_lock = asyncio.Lock()

    async def advance(amount: float = 1.0) -> None:
        async with progress_lock:
            value = min(float(work_total), float(progress_state.get("overall_done", 0.0)) + amount)
            progress_state["overall_done"] = value
            progress_state["done"] = min(work_total, int(value))

    timeout = httpx.Timeout(connect=10, read=35, write=10, pool=35)
    async with httpx.AsyncClient(base_url=main.BASE_URL, headers={"x-apisports-key": main.API_KEY}, timeout=timeout) as client:
        for start in range(0, total, PROGRESSIVE_BATCH_SIZE):
            batch = candidates[start:start + PROGRESSIVE_BATCH_SIZE]
            historical_ids: set[int] = set()
            progress_state["phase"] = "EVIDENCE_WARMUP"

            async def warm(fixture: dict[str, Any]) -> None:
                try:
                    shell = main.base_shell(fixture)
                    cutoff = main.parse_dt(shell["kickoff_utc"])
                    if not cutoff or main.datetime.now(main.UTC) >= cutoff:
                        return
                    home, away = await asyncio.gather(
                        main.team_profile(client, shell["home_id"], shell["league_id"], shell["season"], cutoff),
                        main.team_profile(client, shell["away_id"], shell["league_id"], shell["season"], cutoff),
                    )
                    historical_ids.update(match["fixture_id"] for match in home["matches"] + away["matches"])
                except Exception as exc:
                    main.log.warning("Precarga progresiva incompleta fixture=%s: %s", main.base_shell(fixture)["fixture_id"], exc)
                finally:
                    await advance()

            for warm_batch in main.chunks(batch, main.ANALYSIS_WORKERS):
                await asyncio.gather(*(warm(fixture) for fixture in warm_batch))

            progress_state["phase"] = "ADVANCED_HISTORY"
            # Keep the helper focused on fetching/cache population. Account for
            # this stage here so /sync has one consistent progress writer.
            await main.preload_fixture_stats(client, historical_ids)
            await advance(float(len(batch)))

            progress_state["phase"] = "MODEL_AND_PERSIST"
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            for fixture in batch:
                queue.put_nowait(fixture)

            async def worker() -> None:
                while True:
                    try:
                        fixture = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    shell = main.base_shell(fixture)
                    fixture_id = shell["fixture_id"]
                    try:
                        result = main.stamp_snapshot(await main.build_analysis(client, fixture))
                        if result["analysis_status"] in {"READY", "ABSTAINED"}:
                            snapshot_state[fixture_id] = result
                            cutoff = result.get("snapshot_cutoff_utc") or main.datetime.now(main.UTC).isoformat()
                            await asyncio.to_thread(main.db_save_snapshot, fixture_date, fixture_id, cutoff, result)
                        error_state.pop(fixture_id, None)
                    except Exception as exc:
                        error_state[fixture_id] = str(exc)
                        main.log.error("Análisis progresivo fallido fixture=%s: %s", fixture_id, exc)
                    finally:
                        progress_state["fixtures_done"] = min(total, int(progress_state.get("fixtures_done", 0)) + 1)
                        await advance()
                        queue.task_done()

            await asyncio.gather(*(worker() for _ in range(min(main.ANALYSIS_WORKERS, len(batch)))))
            main.log.info(
                "Lote progresivo persistido fecha=%s fixtures=%s/%s progreso=%.1f%% snapshots=%s",
                fixture_date, min(start + len(batch), total), total,
                100.0 * float(progress_state["overall_done"]) / work_total, len(snapshot_state),
            )
            await asyncio.sleep(0)

    progress_state.update(
        phase="COMPLETE", done=work_total, total=work_total,
        overall_done=float(work_total), overall_total=float(work_total),
        fixtures_done=total, fixtures_total=total,
    )
    if save_coverage:
        await asyncio.to_thread(main.db_save_coverage_manifest, main.coverage_payload())


def install() -> None:
    main.analyze_catalog = analyze_catalog_progressive
    main.log.info("Procesamiento progresivo activo batch_size=%s", PROGRESSIVE_BATCH_SIZE)
