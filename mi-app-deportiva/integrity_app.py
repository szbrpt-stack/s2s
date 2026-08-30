from __future__ import annotations

from typing import Any

import main
import runtime_recovery

app = runtime_recovery.app


def _empirical_reliability() -> dict[str, Any]:
    if not main.DATABASE_URL or main.psycopg is None:
        return {"available": False, "reason": "Persistent PostgreSQL is not configured"}

    quality_sql = """
    WITH x AS (
      SELECT
        (prediction::jsonb #>> '{evidence_quality,score}')::int AS quality,
        (metrics::jsonb->>'brier_1x2')::float AS brier,
        (metrics::jsonb->>'log_loss_1x2')::float AS logloss,
        (metrics::jsonb->>'correct_1x2')::int AS correct,
        (metrics::jsonb->>'brier_over25')::float AS brier_o25,
        (metrics::jsonb->>'brier_btts')::float AS brier_btts
      FROM prediction_outcomes
      WHERE metrics IS NOT NULL AND prediction IS NOT NULL
    )
    SELECT
      CASE
        WHEN quality >= 90 THEN '90-100'
        WHEN quality >= 75 THEN '75-89'
        WHEN quality >= 60 THEN '60-74'
        ELSE '<60'
      END AS bucket,
      COUNT(*)::int AS n,
      AVG(correct)::float AS accuracy,
      AVG(brier)::float AS brier_1x2,
      AVG(logloss)::float AS logloss_1x2,
      AVG(brier_o25)::float AS brier_over25,
      AVG(brier_btts)::float AS brier_btts
    FROM x GROUP BY 1 ORDER BY MIN(quality)
    """
    sigma_sql = """
    WITH x AS (
      SELECT
        (prediction::jsonb #>> '{sigma_score}')::int AS sigma,
        (metrics::jsonb->>'brier_1x2')::float AS brier,
        (metrics::jsonb->>'log_loss_1x2')::float AS logloss,
        (metrics::jsonb->>'correct_1x2')::int AS correct,
        (metrics::jsonb->>'brier_over25')::float AS brier_o25,
        (metrics::jsonb->>'brier_btts')::float AS brier_btts
      FROM prediction_outcomes
      WHERE metrics IS NOT NULL AND prediction IS NOT NULL
        AND prediction::jsonb ? 'sigma_score'
    )
    SELECT
      CASE
        WHEN sigma >= 80 THEN '80-100'
        WHEN sigma >= 70 THEN '70-79'
        WHEN sigma >= 60 THEN '60-69'
        ELSE '<60'
      END AS bucket,
      COUNT(*)::int AS n,
      AVG(correct)::float AS accuracy,
      AVG(brier)::float AS brier_1x2,
      AVG(logloss)::float AS logloss_1x2,
      AVG(brier_o25)::float AS brier_over25,
      AVG(brier_btts)::float AS brier_btts
    FROM x GROUP BY 1 ORDER BY MIN(sigma)
    """
    overall_sql = """
    SELECT
      COUNT(*)::int AS n,
      AVG((metrics::jsonb->>'correct_1x2')::int)::float AS accuracy,
      AVG((metrics::jsonb->>'brier_1x2')::float)::float AS brier_1x2,
      AVG((metrics::jsonb->>'log_loss_1x2')::float)::float AS logloss_1x2,
      AVG((metrics::jsonb->>'brier_over25')::float)::float AS brier_over25,
      AVG((metrics::jsonb->>'brier_btts')::float)::float AS brier_btts
    FROM prediction_outcomes
    WHERE metrics IS NOT NULL AND prediction IS NOT NULL
    """

    with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
        quality_rows = db.execute(quality_sql).fetchall()
        sigma_rows = db.execute(sigma_sql).fetchall()
        overall = db.execute(overall_sql).fetchone()

    def rows_to_payload(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        return [
            {
                "bucket": row[0], "n": int(row[1]),
                "accuracy_1x2": round(float(row[2]), 6) if row[2] is not None else None,
                "brier_1x2": round(float(row[3]), 6) if row[3] is not None else None,
                "logloss_1x2": round(float(row[4]), 6) if row[4] is not None else None,
                "brier_over25": round(float(row[5]), 6) if row[5] is not None else None,
                "brier_btts": round(float(row[6]), 6) if row[6] is not None else None,
            }
            for row in rows
        ]

    quality = rows_to_payload(quality_rows)
    sigma = rows_to_payload(sigma_rows)

    # Data-quality scores are useful for provenance/completeness, but they should
    # not be presented as empirical confidence unless performance improves
    # monotonically as the score rises.
    ordered_quality = sorted(quality, key=lambda row: {"<60": 0, "60-74": 1, "75-89": 2, "90-100": 3}.get(row["bucket"], 99))
    quality_monotonic_accuracy = all(
        ordered_quality[i]["accuracy_1x2"] <= ordered_quality[i + 1]["accuracy_1x2"]
        for i in range(len(ordered_quality) - 1)
        if ordered_quality[i]["accuracy_1x2"] is not None and ordered_quality[i + 1]["accuracy_1x2"] is not None
    )
    quality_monotonic_brier = all(
        ordered_quality[i]["brier_1x2"] >= ordered_quality[i + 1]["brier_1x2"]
        for i in range(len(ordered_quality) - 1)
        if ordered_quality[i]["brier_1x2"] is not None and ordered_quality[i + 1]["brier_1x2"] is not None
    )

    return {
        "available": True,
        "model_version": main.MODEL_VERSION,
        "overall": {
            "n": int(overall[0]),
            "accuracy_1x2": round(float(overall[1]), 6),
            "brier_1x2": round(float(overall[2]), 6),
            "logloss_1x2": round(float(overall[3]), 6),
            "brier_over25": round(float(overall[4]), 6),
            "brier_btts": round(float(overall[5]), 6),
        },
        "by_data_quality": quality,
        "by_sigma": sigma,
        "diagnostics": {
            "data_quality_monotonic_accuracy": quality_monotonic_accuracy,
            "data_quality_monotonic_brier": quality_monotonic_brier,
            "data_quality_is_empirical_confidence": quality_monotonic_accuracy and quality_monotonic_brier,
            "interpretation": (
                "evidence_quality measures provenance/completeness and must remain separate from empirical reliability"
            ),
        },
        "reporting_policy": {
            "keep_data_quality_separate": True,
            "require_prequential_metrics": True,
            "require_minimum_bucket_sample": 100,
            "do_not_infer_precision_from_quality_score_alone": True,
        },
    }


@app.get("/api/v1/model/empirical-integrity")
async def empirical_integrity() -> dict[str, Any]:
    import asyncio
    return await asyncio.to_thread(_empirical_reliability)
