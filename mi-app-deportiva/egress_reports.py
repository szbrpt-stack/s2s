"""PostgreSQL-side report aggregation for S2S.

The scorecard no longer transfers the last 5,000 prediction/outcome JSON rows to
Render. PostgreSQL extracts and aggregates the persisted metrics and returns a
small set of numeric rows. SQLite keeps the legacy implementation.
"""
from __future__ import annotations

from typing import Any

import main

_installed = False


def _summary(row: tuple[Any, ...] | None) -> dict[str, Any]:
    if not row:
        return {"n": 0, "log_loss_1x2": None, "brier_1x2": None, "accuracy_1x2": None,
                "brier_over25": None, "baseline_brier_1x2": None, "baseline_brier_over25": None,
                "brier_improvement_pct": None, "over25_brier_improvement_pct": None,
                "brier_btts": None, "baseline_brier_btts": None, "btts_brier_improvement_pct": None,
                "advanced": {name: {"n": 0, "brier": None, "mae": None, "baseline_brier": None,
                                      "brier_improvement_pct": None} for name in ("corners", "cards", "shots")}}
    n = int(row[0] or 0)
    values = [float(v) if v is not None else None for v in row[1:]]
    (logloss, brier, accuracy, over_brier, baseline_brier, baseline_over,
     btts_brier, baseline_btts, corners_brier, corners_mae, cards_brier,
     cards_mae, shots_brier, shots_mae) = values[:14]
    counts = [int(v or 0) for v in row[15:18]]
    def improvement(base: float | None, value: float | None) -> float | None:
        return round((base - value) / base * 100, 4) if base and value is not None else None
    advanced = {}
    for name, count, field_brier, mae in (
        ("corners", counts[0], corners_brier, corners_mae),
        ("cards", counts[1], cards_brier, cards_mae),
        ("shots", counts[2], shots_brier, shots_mae),
    ):
        advanced[name] = {
            "n": count,
            "brier": round(field_brier, 6) if field_brier is not None else None,
            "mae": round(mae, 6) if mae is not None else None,
            "baseline_brier": 0.25 if count else None,
            "brier_improvement_pct": improvement(0.25, field_brier),
        }
    return {
        "n": n,
        "log_loss_1x2": round(logloss, 6) if logloss is not None else None,
        "brier_1x2": round(brier, 6) if brier is not None else None,
        "accuracy_1x2": round(accuracy, 6) if accuracy is not None else None,
        "brier_over25": round(over_brier, 6) if over_brier is not None else None,
        "baseline_brier_1x2": round(baseline_brier, 6) if baseline_brier is not None else None,
        "baseline_brier_over25": round(baseline_over, 6) if baseline_over is not None else None,
        "brier_improvement_pct": improvement(baseline_brier, brier),
        "over25_brier_improvement_pct": improvement(baseline_over, over_brier),
        "brier_btts": round(btts_brier, 6) if btts_brier is not None else None,
        "baseline_brier_btts": round(baseline_btts, 6) if baseline_btts is not None else None,
        "btts_brier_improvement_pct": improvement(baseline_btts, btts_brier),
        "advanced": advanced,
    }


BASE_CTE = """
WITH recent AS (
    SELECT kickoff_utc, prediction::jsonb AS p, metrics::jsonb AS m,
           actual_home, actual_away, resolved_at
    FROM prediction_outcomes
    ORDER BY resolved_at::timestamptz DESC
    LIMIT 5000
), x AS (
    SELECT *,
      NULLIF(m->>'log_loss_1x2','')::float AS logloss,
      NULLIF(m->>'brier_1x2','')::float AS brier,
      NULLIF(m->>'correct_1x2','')::float AS correct,
      NULLIF(m->>'brier_over25','')::float AS over_brier,
      NULLIF(m->>'baseline_brier_1x2','')::float AS baseline_brier,
      NULLIF(m->>'baseline_brier_over25','')::float AS baseline_over,
      NULLIF(m->>'brier_btts','')::float AS btts_brier,
      NULLIF(m->>'baseline_brier_btts','')::float AS baseline_btts,
      NULLIF(m->>'corners_brier','')::float AS corners_brier,
      NULLIF(m->>'corners_mae','')::float AS corners_mae,
      NULLIF(m->>'cards_brier','')::float AS cards_brier,
      NULLIF(m->>'cards_mae','')::float AS cards_mae,
      NULLIF(m->>'shots_brier','')::float AS shots_brier,
      NULLIF(m->>'shots_mae','')::float AS shots_mae,
      NULLIF(m->>'modal_correct','')::float AS modal_correct,
      COALESCE(NULLIF(p->>'league_id',''),'0')::int AS league_id,
      COALESCE(NULLIF(p->>'league_name',''), NULLIF(p->>'liga',''), 'N/D') AS league_name
    FROM recent
)
"""

SUMMARY_SELECT = """
SELECT COUNT(*)::int,
 AVG(logloss)::float, AVG(brier)::float, AVG(correct)::float, AVG(over_brier)::float,
 AVG(baseline_brier)::float, AVG(baseline_over)::float, AVG(btts_brier)::float, AVG(baseline_btts)::float,
 AVG(corners_brier)::float, AVG(corners_mae)::float, AVG(cards_brier)::float, AVG(cards_mae)::float,
 AVG(shots_brier)::float, AVG(shots_mae)::float,
 COUNT(corners_brier)::int, COUNT(cards_brier)::int, COUNT(shots_brier)::int
FROM x
"""


def scorecard_postgres() -> dict[str, Any]:
    with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
        overall_row = db.execute(BASE_CTE + SUMMARY_SELECT).fetchone()
        overall = _summary(overall_row)
        n = overall["n"]
        if not n:
            return {"status": "INSUFFICIENT", "resolved_predictions": 0, "minimum_for_reporting": 30}

        horizons: dict[str, Any] = {}
        for label, days in (("7d", 7), ("30d", 30), ("90d", 90)):
            row = db.execute(BASE_CTE + SUMMARY_SELECT + " WHERE kickoff_utc::timestamptz >= now() - (%s * interval '1 day')", (days,)).fetchone()
            horizons[label] = _summary(row)

        league_rows = db.execute(BASE_CTE + """
            SELECT league_id, league_name, COUNT(*)::int,
             AVG(logloss)::float, AVG(brier)::float, AVG(correct)::float, AVG(over_brier)::float,
             AVG(baseline_brier)::float, AVG(baseline_over)::float, AVG(btts_brier)::float, AVG(baseline_btts)::float,
             AVG(corners_brier)::float, AVG(corners_mae)::float, AVG(cards_brier)::float, AVG(cards_mae)::float,
             AVG(shots_brier)::float, AVG(shots_mae)::float,
             COUNT(corners_brier)::int, COUNT(cards_brier)::int, COUNT(shots_brier)::int
            FROM x GROUP BY league_id,league_name
        """).fetchall()
        by_league = {f"{r[0]}:{r[1]}": _summary(r[2:]) for r in league_rows}

        modal = db.execute(BASE_CTE + "SELECT AVG(modal_correct)::float FROM x").fetchone()

        def bins(market: str) -> list[dict[str, Any]]:
            if market == "1x2":
                probability = "GREATEST(COALESCE(NULLIF(p->>'p_home','')::float,0),COALESCE(NULLIF(p->>'p_draw','')::float,0),COALESCE(NULLIF(p->>'p_away','')::float,0))/100.0"
                actual = "COALESCE(correct,0)"
            elif market == "over25":
                probability = "NULLIF(m->>'probability_over25','')::float"
                actual = "(actual_home + actual_away >= 3)::int"
            else:
                probability = "NULLIF(m->>'probability_btts','')::float"
                actual = "(actual_home > 0 AND actual_away > 0)::int"
            q = BASE_CTE + f"""
                , b AS (SELECT {probability} AS prob, {actual}::float AS actual FROM x)
                SELECT LEAST(FLOOR(prob*10)::int,9) AS bucket, COUNT(*)::int,
                       AVG(prob)::float, AVG(actual)::float
                FROM b WHERE prob IS NOT NULL
                GROUP BY 1 ORDER BY 1
            """
            return [
                {"from": int(r[0]) / 10, "to": (int(r[0]) + 1) / 10, "n": int(r[1]),
                 "mean_probability": round(float(r[2]), 4), "observed_frequency": round(float(r[3]), 4)}
                for r in db.execute(q).fetchall()
            ]

        calibration_1x2 = bins("1x2")
        calibration_o25 = bins("over25")
        calibration_btts = bins("btts")

    promotion = {
        "1x2": {"eligible": n >= 300 and (overall.get("brier_improvement_pct") or -999) > 0,
                "requirements": {"minimum_n": 300, "positive_brier_improvement": True}},
        "goals_2_5": {"eligible": n >= 300 and (overall.get("over25_brier_improvement_pct") or -999) > 0,
                      "requirements": {"minimum_n": 300, "positive_brier_improvement": True}},
        "btts": {"eligible": n >= 300 and (overall.get("btts_brier_improvement_pct") or -999) > 0,
                 "requirements": {"minimum_n": 300, "positive_brier_improvement": True},
                 "current": {"n": n, "brier": overall.get("brier_btts"),
                             "improvement_pct": overall.get("btts_brier_improvement_pct")}},
    }
    for field in ("corners", "cards", "shots"):
        current = overall["advanced"][field]
        promotion[field] = {
            "eligible": current["n"] >= 100 and (current.get("brier_improvement_pct") or -999) > 0,
            "requirements": {"minimum_n": 100, "positive_brier_improvement": True},
            "current": current,
        }
    return {
        "status": "PRELIMINARY" if n < 100 else "MONITORED",
        "resolved_predictions": n,
        "minimum_for_reporting": 30,
        "overall": overall,
        "horizons": horizons,
        "by_league": by_league,
        "calibration_bins_1x2": calibration_1x2,
        "calibration_bins_over25": calibration_o25,
        "calibration_bins_btts": calibration_btts,
        "promotion_gates": promotion,
        "log_loss_1x2": overall.get("log_loss_1x2"),
        "brier_1x2": overall.get("brier_1x2"),
        "accuracy_1x2": overall.get("accuracy_1x2"),
        "brier_over25": overall.get("brier_over25"),
        "brier_btts": overall.get("brier_btts"),
        "modal_accuracy": round(float(modal[0]), 6) if modal and modal[0] is not None else None,
        "warning": "Métrica prequential descriptiva; promoción nunca automática.",
    }


def install() -> None:
    global _installed
    if _installed:
        return
    if main.DATABASE_URL and main.psycopg is not None:
        main.db_model_scorecard = scorecard_postgres
        main.log.info("PostgreSQL-side scorecard aggregation active")
    _installed = True
