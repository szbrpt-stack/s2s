from __future__ import annotations

import math
from typing import Any

import main
import incremental_catalog

incremental_catalog.install()

import runtime_recovery
import progressive_analysis
import egress_reports
import egress_daily
import egress_guard

progressive_analysis.install()
# Replace legacy 5k-row report reads before wrapping reports with the bounded cache.
egress_reports.install()
egress_daily.install()
egress_guard.install()
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
    n = int(row[1]); accuracy = round(float(row[2]), 6) if row[2] is not None else None; successes = int(round(float(row[2]) * n)) if row[2] is not None else 0
    return {label_key: row[0], "n": n, "accuracy_1x2": accuracy,"accuracy_1x2_ci95": _wilson_interval(successes, n) if accuracy is not None else None,"brier_1x2": round(float(row[3]), 6) if row[3] is not None else None,"logloss_1x2": round(float(row[4]), 6) if row[4] is not None else None,"brier_over25": round(float(row[5]), 6) if row[5] is not None else None,"brier_btts": round(float(row[6]), 6) if row[6] is not None else None}


def _empirical_reliability() -> dict[str, Any]:
    if not main.DATABASE_URL or main.psycopg is None: return {"available": False, "reason": "Persistent PostgreSQL is not configured"}
    active_filter = "model_version = %s AND metrics IS NOT NULL AND prediction IS NOT NULL"
    quality_sql = f"""WITH x AS (SELECT (prediction::jsonb #>> '{{evidence_quality,score}}')::int AS quality,(metrics::jsonb->>'brier_1x2')::float AS brier,(metrics::jsonb->>'log_loss_1x2')::float AS logloss,(metrics::jsonb->>'correct_1x2')::int AS correct,(metrics::jsonb->>'brier_over25')::float AS brier_o25,(metrics::jsonb->>'brier_btts')::float AS brier_btts FROM prediction_outcomes WHERE {active_filter}) SELECT CASE WHEN quality >= 90 THEN '90-100' WHEN quality >= 75 THEN '75-89' WHEN quality >= 60 THEN '60-74' ELSE '<60' END AS bucket,COUNT(*)::int,AVG(correct)::float,AVG(brier)::float,AVG(logloss)::float,AVG(brier_o25)::float,AVG(brier_btts)::float FROM x GROUP BY 1 ORDER BY MIN(quality)"""
    sigma_sql = f"""WITH x AS (SELECT NULLIF(prediction::jsonb #>> '{{sigma_score}}','')::float AS sigma,(metrics::jsonb->>'brier_1x2')::float AS brier,(metrics::jsonb->>'log_loss_1x2')::float AS logloss,(metrics::jsonb->>'correct_1x2')::int AS correct,(metrics::jsonb->>'brier_over25')::float AS brier_o25,(metrics::jsonb->>'brier_btts')::float AS brier_btts FROM prediction_outcomes WHERE {active_filter} AND prediction::jsonb ? 'sigma_score') SELECT CASE WHEN sigma >= 80 THEN '80-100' WHEN sigma >= 70 THEN '70-79' WHEN sigma >= 60 THEN '60-69' ELSE '<60' END AS bucket,COUNT(*)::int,AVG(correct)::float,AVG(brier)::float,AVG(logloss)::float,AVG(brier_o25)::float,AVG(brier_btts)::float FROM x GROUP BY 1 ORDER BY MIN(sigma)"""
    overall_sql = f"""SELECT COUNT(*)::int,SUM((metrics::jsonb->>'correct_1x2')::int)::int,AVG((metrics::jsonb->>'correct_1x2')::int)::float,AVG((metrics::jsonb->>'brier_1x2')::float)::float,AVG((metrics::jsonb->>'log_loss_1x2')::float)::float,AVG((metrics::jsonb->>'brier_over25')::float)::float,AVG((metrics::jsonb->>'brier_btts')::float)::float,AVG((metrics::jsonb->>'baseline_brier_1x2')::float)::float,AVG((metrics::jsonb->>'baseline_log_loss_1x2')::float)::float FROM prediction_outcomes WHERE {active_filter}"""
    cohort_sql = f"""WITH x AS (SELECT COALESCE(prediction::jsonb #>> '{{primary_pick,status}}', 'NONE') AS cohort,(metrics::jsonb->>'correct_1x2')::int AS correct,(metrics::jsonb->>'brier_1x2')::float AS brier,(metrics::jsonb->>'log_loss_1x2')::float AS logloss,(metrics::jsonb->>'baseline_brier_1x2')::float AS baseline_brier,(metrics::jsonb->>'baseline_log_loss_1x2')::float AS baseline_logloss,(metrics::jsonb->>'primary_correct')::int AS primary_correct,NULLIF(prediction::jsonb->>'sigma_score','')::float AS sigma FROM prediction_outcomes WHERE {active_filter}) SELECT cohort,COUNT(*)::int,SUM(correct)::int,AVG(correct)::float,AVG(brier)::float,AVG(logloss)::float,AVG(baseline_brier)::float,AVG(baseline_logloss)::float,COUNT(primary_correct)::int,SUM(primary_correct)::int,AVG(primary_correct)::float,AVG(sigma)::float FROM x GROUP BY cohort ORDER BY COUNT(*) DESC"""
    windows_sql = f"""SELECT COUNT(*) FILTER (WHERE resolved_at::timestamptz >= now() - interval '24 hours')::int,AVG((metrics::jsonb->>'correct_1x2')::int) FILTER (WHERE resolved_at::timestamptz >= now() - interval '24 hours')::float,COUNT(*) FILTER (WHERE resolved_at::timestamptz >= now() - interval '7 days')::int,AVG((metrics::jsonb->>'correct_1x2')::int) FILTER (WHERE resolved_at::timestamptz >= now() - interval '7 days')::float FROM prediction_outcomes WHERE {active_filter}"""
    versions_sql = """SELECT model_version,COUNT(*)::int,AVG((metrics::jsonb->>'correct_1x2')::int)::float,AVG((metrics::jsonb->>'brier_1x2')::float)::float FROM prediction_outcomes WHERE metrics IS NOT NULL AND prediction IS NOT NULL GROUP BY model_version ORDER BY COUNT(*) DESC"""
    params = (main.MODEL_VERSION,)
    with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
        quality_rows=db.execute(quality_sql,params).fetchall(); sigma_rows=db.execute(sigma_sql,params).fetchall(); overall=db.execute(overall_sql,params).fetchone(); cohort_rows=db.execute(cohort_sql,params).fetchall(); windows=db.execute(windows_sql,params).fetchone(); version_rows=db.execute(versions_sql).fetchall()
    if not overall or int(overall[0] or 0)==0: return {"available":False,"reason":"No resolved outcomes for active model","model_version":main.MODEL_VERSION}
    quality=[_metric_row(r) for r in quality_rows]; sigma=[_metric_row(r) for r in sigma_rows]; cohorts=[]; total_current=int(overall[0])
    for r in cohort_rows:
        pn=int(r[8]); ps=int(r[9] or 0); n=int(r[1]); cohorts.append({"cohort":r[0],"n":n,"share":round(n/total_current,6) if total_current else None,"accuracy_1x2":round(float(r[3]),6) if r[3] is not None else None,"accuracy_1x2_ci95":_wilson_interval(int(r[2]),n),"brier_1x2":round(float(r[4]),6) if r[4] is not None else None,"logloss_1x2":round(float(r[5]),6) if r[5] is not None else None,"legacy_primary_n":pn,"legacy_primary_accuracy":round(float(r[10]),6) if r[10] is not None else None,"legacy_primary_accuracy_ci95":_wilson_interval(ps,pn) if pn else None,"avg_sigma":round(float(r[11]),6) if r[11] is not None else None})
    ordered=sorted(quality,key=lambda r:{"<60":0,"60-74":1,"75-89":2,"90-100":3}.get(r["bucket"],99)); mono_acc=all(ordered[i]["accuracy_1x2"]<=ordered[i+1]["accuracy_1x2"] for i in range(len(ordered)-1) if ordered[i]["accuracy_1x2"] is not None and ordered[i+1]["accuracy_1x2"] is not None); mono_brier=all(ordered[i]["brier_1x2"]>=ordered[i+1]["brier_1x2"] for i in range(len(ordered)-1) if ordered[i]["brier_1x2"] is not None and ordered[i+1]["brier_1x2"] is not None)
    n=int(overall[0]); successes=int(overall[1] or 0); brier=round(float(overall[3]),6) if overall[3] is not None else None; logloss=round(float(overall[4]),6) if overall[4] is not None else None; bb=round(float(overall[7]),6) if overall[7] is not None else None; bl=round(float(overall[8]),6) if overall[8] is not None else None; w24n,w24acc,w7n,w7acc=windows
    return {"available":True,"model_version":main.MODEL_VERSION,"headline":{"metric_name":"accuracy_1x2_model","n":n,"accuracy_1x2_model":round(float(overall[2]),6) if overall[2] is not None else None,"accuracy_1x2_ci95":_wilson_interval(successes,n),"window_24h":{"n":int(w24n or 0),"accuracy_1x2_model":round(float(w24acc),6) if w24acc is not None else None},"window_7d":{"n":int(w7n or 0),"accuracy_1x2_model":round(float(w7acc),6) if w7acc is not None else None},"definition":"Exactitud del resultado 1X2 del modelo activo; no mezcla versiones ni cohortes heredadas."},"probabilistic_quality":{"brier_1x2":brier,"logloss_1x2":logloss,"brier_over25":round(float(overall[5]),6) if overall[5] is not None else None,"brier_btts":round(float(overall[6]),6) if overall[6] is not None else None,"baseline_brier_1x2":bb,"baseline_logloss_1x2":bl,"beats_baseline_brier_1x2":bb is not None and brier is not None and brier<bb,"beats_baseline_logloss_1x2":bl is not None and logloss is not None and logloss<bl},"legacy_cohorts":cohorts,"historical_versions":[{"model_version":r[0],"n":int(r[1]),"accuracy_1x2":round(float(r[2]),6) if r[2] is not None else None,"brier_1x2":round(float(r[3]),6) if r[3] is not None else None} for r in version_rows],"by_data_quality":quality,"by_sigma":sigma,"diagnostics":{"data_quality_monotonic_accuracy":mono_acc,"data_quality_monotonic_brier":mono_brier,"data_quality_is_empirical_confidence":mono_acc and mono_brier,"interpretation":"evidence_quality mide procedencia/completitud y permanece separada de la fiabilidad empírica."},"reporting_policy":{"generic_winrate_field_removed":True,"headline_uses_active_model_only":True,"historical_versions_are_diagnostic_only":True,"legacy_primary_metrics_are_not_headline_metrics":True,"keep_data_quality_separate":True,"require_prequential_metrics":True,"require_confidence_intervals":True,"compare_against_baseline":True}}


@app.get("/api/v1/model/empirical-integrity")
async def empirical_integrity() -> dict[str, Any]:
    return await main.asyncio.to_thread(_empirical_reliability)
