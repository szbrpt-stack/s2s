from __future__ import annotations

import math
from typing import Any

import main
import runtime_recovery

app = runtime_recovery.app


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float] | None:
    if n <= 0:
        return None
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator
    return {"low": round(max(0.0, center - margin), 6), "high": round(min(1.0, center + margin), 6)}


def _metric_row(row: tuple[Any, ...], label_key: str = "bucket") -> dict[str, Any]:
    n = int(row[1])
    accuracy = round(float(row[2]), 6) if row[2] is not None else None
    successes = int(round(float(row[2]) * n)) if row[2] is not None else 0
    return {
        label_key: row[0],
        "n": n,
        "accuracy_1x2": accuracy,
        "accuracy_1x2_ci95": _wilson_interval(successes, n) if accuracy is not None else None,
        "brier_1x2": round(float(row[3]), 6) if row[3] is not None else None,
        "logloss_1x2": round(float(row[4]), 6) if row[4] is not None else None,
        "brier_over25": round(float(row[5]), 6) if row[5] is not None else None,
        "brier_btts": round(float(row[6]), 6) if row[6] is not None else None,
    }


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
      SUM((metrics::jsonb->>'correct_1x2')::int)::int AS successes,
      AVG((metrics::jsonb->>'correct_1x2')::int)::float AS accuracy,
      AVG((metrics::jsonb->>'brier_1x2')::float)::float AS brier_1x2,
      AVG((metrics::jsonb->>'log_loss_1x2')::float)::float AS logloss_1x2,
      AVG((metrics::jsonb->>'brier_over25')::float)::float AS brier_over25,
      AVG((metrics::jsonb->>'brier_btts')::float)::float AS brier_btts,
      AVG((metrics::jsonb->>'baseline_brier_1x2')::float)::float AS baseline_brier_1x2,
      AVG((metrics::jsonb->>'baseline_log_loss_1x2')::float)::float AS baseline_logloss_1x2
    FROM prediction_outcomes
    WHERE metrics IS NOT NULL AND prediction IS NOT NULL
    """

    cohort_sql = """
    WITH x AS (
      SELECT
        COALESCE(prediction::jsonb #>> '{primary_pick,status}', 'NONE') AS cohort,
        (metrics::jsonb->>'correct_1x2')::int AS correct,
        (metrics::jsonb->>'brier_1x2')::float AS brier,
        (metrics::jsonb->>'log_loss_1x2')::float AS logloss,
        (metrics::jsonb->>'baseline_brier_1x2')::float AS baseline_brier,
        (metrics::jsonb->>'baseline_log_loss_1x2')::float AS baseline_logloss,
        (metrics::jsonb->>'primary_correct')::int AS primary_correct,
        NULLIF(prediction::jsonb->>'sigma_score','')::int AS sigma
      FROM prediction_outcomes
      WHERE metrics IS NOT NULL AND prediction IS NOT NULL
    )
    SELECT cohort,
           COUNT(*)::int AS n,
           SUM(correct)::int AS successes,
           AVG(correct)::float AS accuracy,
           AVG(brier)::float AS brier_1x2,
           AVG(logloss)::float AS logloss_1x2,
           AVG(baseline_brier)::float AS baseline_brier_1x2,
           AVG(baseline_logloss)::float AS baseline_logloss_1x2,
           COUNT(primary_correct)::int AS primary_n,
           SUM(primary_correct)::int AS primary_successes,
           AVG(primary_correct)::float AS primary_accuracy,
           AVG(sigma)::float AS avg_sigma
    FROM x GROUP BY cohort ORDER BY n DESC
    """

    with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
        quality_rows = db.execute(quality_sql).fetchall()
        sigma_rows = db.execute(sigma_sql).fetchall()
        overall = db.execute(overall_sql).fetchone()
        cohort_rows = db.execute(cohort_sql).fetchall()

    quality = [_metric_row(row) for row in quality_rows]
    sigma = [_metric_row(row) for row in sigma_rows]

    cohorts = []
    for row in cohort_rows:
        primary_n = int(row[8])
        primary_successes = int(row[9] or 0)
        cohorts.append({
            "cohort": row[0],
            "n": int(row[1]),
            "accuracy_1x2": round(float(row[3]), 6) if row[3] is not None else None,
            "accuracy_1x2_ci95": _wilson_interval(int(row[2]), int(row[1])),
            "brier_1x2": round(float(row[4]), 6) if row[4] is not None else None,
            "logloss_1x2": round(float(row[5]), 6) if row[5] is not None else None,
            "baseline_brier_1x2": round(float(row[6]), 6) if row[6] is not None else None,
            "baseline_logloss_1x2": round(float(row[7]), 6) if row[7] is not None else None,
            "primary_n": primary_n,
            "primary_accuracy": round(float(row[10]), 6) if row[10] is not None else None,
            "primary_accuracy_ci95": _wilson_interval(primary_successes, primary_n) if primary_n else None,
            "avg_sigma": round(float(row[11]), 6) if row[11] is not None else None,
        })

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

    overall_n = int(overall[0])
    overall_successes = int(overall[1])
    overall_brier = round(float(overall[3]), 6)
    overall_logloss = round(float(overall[4]), 6)
    baseline_brier = round(float(overall[7]), 6) if overall[7] is not None else None
    baseline_logloss = round(float(overall[8]), 6) if overall[8] is not None else None

    return {
        "available": True,
        "model_version": main.MODEL_VERSION,
        "overall": {
            "n": overall_n,
            "accuracy_1x2": round(float(overall[2]), 6),
            "accuracy_1x2_ci95": _wilson_interval(overall_successes, overall_n),
            "brier_1x2": overall_brier,
            "logloss_1x2": overall_logloss,
            "brier_over25": round(float(overall[5]), 6),
            "brier_btts": round(float(overall[6]), 6),
            "baseline_brier_1x2": baseline_brier,
            "baseline_logloss_1x2": baseline_logloss,
            "beats_baseline_brier_1x2": baseline_brier is not None and overall_brier < baseline_brier,
            "beats_baseline_logloss_1x2": baseline_logloss is not None and overall_logloss < baseline_logloss,
        },
        "by_data_quality": quality,
        "by_sigma": sigma,
        "by_selection_cohort": cohorts,
        "diagnostics": {
            "data_quality_monotonic_accuracy": quality_monotonic_accuracy,
            "data_quality_monotonic_brier": quality_monotonic_brier,
            "data_quality_is_empirical_confidence": quality_monotonic_accuracy and quality_monotonic_brier,
            "interpretation": "evidence_quality measures provenance/completeness and must remain separate from empirical reliability",
        },
        "reporting_policy": {
            "keep_data_quality_separate": True,
            "require_prequential_metrics": True,
            "require_minimum_bucket_sample": 100,
            "require_confidence_intervals": True,
            "compare_against_baseline": True,
            "do_not_infer_precision_from_quality_score_alone": True,
            "do_not_promote_parameters_from_this_endpoint": True,
        },
    }


@app.get("/api/v1/model/empirical-integrity")
async def empirical_integrity() -> dict[str, Any]:
    import asyncio
    return await asyncio.to_thread(_empirical_reliability)
