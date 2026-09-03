"""PostgreSQL-side daily performance aggregation.

Replaces the legacy 5,000-row prediction_outcomes download with compact SQL
aggregates while preserving the public report shape. SQLite keeps the legacy path.
"""
from __future__ import annotations

from typing import Any

import main

_installed = False


def _wilson(wins: int, total: int) -> list[float] | None:
    return main._wilson_interval(wins, total)


def _filters(date_from: str | None, date_to: str | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if date_from:
        clauses.append("local_day >= %s::date")
        params.append(date_from)
    if date_to:
        clauses.append("local_day <= %s::date")
        params.append(date_to)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


BASE = """
WITH recent AS (
  SELECT fixture_id, kickoff_utc, prediction::jsonb AS p, metrics::jsonb AS m,
         actual_home, actual_away,
         (kickoff_utc::timestamptz AT TIME ZONE 'America/Bogota')::date AS local_day
  FROM prediction_outcomes
  ORDER BY kickoff_utc::timestamptz DESC
  LIMIT 5000
), base AS (
  SELECT *,
    COALESCE(NULLIF(p->>'league_name',''), NULLIF(p->>'liga',''), 'Sin liga') AS league_name,
    COALESCE(NULLIF(p->>'p_home','')::float,0)/100.0 AS ph,
    COALESCE(NULLIF(p->>'p_draw','')::float,0)/100.0 AS pd,
    COALESCE(NULLIF(p->>'p_away','')::float,0)/100.0 AS pa,
    NULLIF(m->>'probability_over25','')::float AS p_over25,
    NULLIF(m->>'probability_btts','')::float AS p_btts,
    NULLIF(m->>'primary_correct','')::int AS primary_correct,
    COALESCE(NULLIF(m->>'primary_market',''),'unknown') AS primary_market
  FROM recent
), verdicts AS (
  SELECT local_day, league_name, '1x2'::text AS market,
         ((CASE WHEN ph>=pd AND ph>=pa THEN 0 WHEN pd>=pa THEN 1 ELSE 2 END) =
          (CASE WHEN actual_home>actual_away THEN 0 WHEN actual_home=actual_away THEN 1 ELSE 2 END)) AS won,
         GREATEST(ph,pd,pa) AS confidence
  FROM base WHERE (ph+pd+pa)>0
  UNION ALL
  SELECT local_day, league_name, 'goals_2_5',
         ((actual_home+actual_away>=3) = (p_over25>=0.5)), GREATEST(p_over25,1-p_over25)
  FROM base WHERE p_over25 IS NOT NULL
  UNION ALL
  SELECT local_day, league_name, 'btts',
         ((actual_home>0 AND actual_away>0) = (p_btts>=0.5)), GREATEST(p_btts,1-p_btts)
  FROM base WHERE p_btts IS NOT NULL
  UNION ALL
  SELECT local_day, league_name, 'corners',
         ((NULLIF(m->>'corners_actual','')::float > NULLIF(p #>> '{metrics,corners_8_5,line}','')::float) =
          (NULLIF(p #>> '{metrics,corners_8_5,probability_over}','')::float >= 0.5)),
         GREATEST(NULLIF(p #>> '{metrics,corners_8_5,probability_over}','')::float,
                  1-NULLIF(p #>> '{metrics,corners_8_5,probability_over}','')::float)
  FROM base WHERE NULLIF(m->>'corners_actual','') IS NOT NULL
    AND NULLIF(p #>> '{metrics,corners_8_5,line}','') IS NOT NULL
    AND NULLIF(p #>> '{metrics,corners_8_5,probability_over}','') IS NOT NULL
  UNION ALL
  SELECT local_day, league_name, 'cards',
         ((NULLIF(m->>'cards_actual','')::float > NULLIF(p #>> '{metrics,cards_4_5,line}','')::float) =
          (NULLIF(p #>> '{metrics,cards_4_5,probability_over}','')::float >= 0.5)),
         GREATEST(NULLIF(p #>> '{metrics,cards_4_5,probability_over}','')::float,
                  1-NULLIF(p #>> '{metrics,cards_4_5,probability_over}','')::float)
  FROM base WHERE NULLIF(m->>'cards_actual','') IS NOT NULL
    AND NULLIF(p #>> '{metrics,cards_4_5,line}','') IS NOT NULL
    AND NULLIF(p #>> '{metrics,cards_4_5,probability_over}','') IS NOT NULL
  UNION ALL
  SELECT local_day, league_name, 'shots',
         ((NULLIF(m->>'shots_actual','')::float > NULLIF(p #>> '{metrics,shots_20_5,line}','')::float) =
          (NULLIF(p #>> '{metrics,shots_20_5,probability_over}','')::float >= 0.5)),
         GREATEST(NULLIF(p #>> '{metrics,shots_20_5,probability_over}','')::float,
                  1-NULLIF(p #>> '{metrics,shots_20_5,probability_over}','')::float)
  FROM base WHERE NULLIF(m->>'shots_actual','') IS NOT NULL
    AND NULLIF(p #>> '{metrics,shots_20_5,line}','') IS NOT NULL
    AND NULLIF(p #>> '{metrics,shots_20_5,probability_over}','') IS NOT NULL
)
"""


def _aggregate_rows(rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    markets: dict[str, Any] = {}
    all_wins = all_total = 0
    for market, wins, total, mean_conf in rows:
        wins, total = int(wins or 0), int(total or 0)
        all_wins += wins; all_total += total
        markets[str(market)] = {
            "wins": wins, "losses": total-wins, "total": total,
            "winrate": round(wins/total,4) if total else None,
            "confidence_interval_95": _wilson(wins,total),
            "mean_model_confidence": round(float(mean_conf),4) if mean_conf is not None else None,
        }
    return {"evaluated_picks": all_total, "wins": all_wins, "losses": all_total-all_wins,
            "winrate": round(all_wins/all_total,4) if all_total else None,
            "confidence_interval_95": _wilson(all_wins,all_total), "markets": markets}


def daily_postgres(date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
    extra, params = _filters(date_from, date_to)
    with main.psycopg.connect(main.DATABASE_URL, connect_timeout=15, prepare_threshold=None) as db:
        fixture_n = db.execute(BASE + "SELECT COUNT(*)::int FROM base WHERE TRUE" + extra, params).fetchone()[0]
        overall_rows = db.execute(BASE + "SELECT market,SUM(won::int)::int,COUNT(*)::int,AVG(confidence)::float FROM verdicts WHERE TRUE" + extra + " GROUP BY market", params).fetchall()
        overall = _aggregate_rows(overall_rows)
        overall["fixtures"] = int(fixture_n or 0)

        primary_rows = db.execute(BASE + "SELECT primary_market,SUM(primary_correct)::int,COUNT(primary_correct)::int FROM base WHERE primary_correct IS NOT NULL" + extra + " GROUP BY primary_market", params).fetchall()
        p_markets: dict[str, Any] = {}
        pw=pt=0
        for market,wins,total in primary_rows:
            wins,total=int(wins or 0),int(total or 0); pw+=wins; pt+=total
            p_markets[str(market)]={"wins":wins,"losses":total-wins,"total":total,"winrate":round(wins/total,4) if total else None}
        overall["primary"]={"wins":pw,"losses":pt-pw,"total":pt,"winrate":round(pw/pt,4) if pt else None,"confidence_interval_95":_wilson(pw,pt),"markets":p_markets}

        conf_rows = db.execute(BASE + """SELECT t.threshold,SUM((v.won)::int)::int,COUNT(v.*)::int
            FROM (VALUES (0.50),(0.55),(0.60),(0.65),(0.70),(0.75),(0.80)) AS t(threshold)
            LEFT JOIN verdicts v ON v.confidence>=t.threshold""" + (" AND " + " AND ".join(c.strip() for c in extra.replace(" AND ","|AND ").split("|") if c.strip()) if extra else "") + " GROUP BY t.threshold ORDER BY t.threshold", params).fetchall()
        confidence_segments=[]
        for th,wins,total in conf_rows:
            wins,total=int(wins or 0),int(total or 0)
            confidence_segments.append({"threshold":float(th),"wins":wins,"losses":total-wins,"total":total,"winrate":round(wins/total,4) if total else None,"confidence_interval_95":_wilson(wins,total)})

        rel_rows = db.execute(BASE + "SELECT LEAST(FLOOR(confidence*10)::int,9),COUNT(*)::int,AVG(confidence)::float,AVG(won::int)::float FROM verdicts WHERE TRUE" + extra + " GROUP BY 1 ORDER BY 1", params).fetchall()
        reliability_bins=[{"from":int(b)/10,"to":(int(b)+1)/10,"total":int(n),"mean_confidence":round(float(mc),4),"observed_winrate":round(float(ow),4)} for b,n,mc,ow in rel_rows]

        daily_raw = db.execute(BASE + "SELECT local_day,market,SUM(won::int)::int,COUNT(*)::int,AVG(confidence)::float FROM verdicts WHERE TRUE" + extra + " GROUP BY local_day,market ORDER BY local_day DESC", params).fetchall()
        daily_map: dict[str,list[tuple[Any,...]]]={}
        for day,market,wins,total,mc in daily_raw: daily_map.setdefault(str(day),[]).append((market,wins,total,mc))
        daily=[]
        for day,rows in daily_map.items():
            agg=_aggregate_rows(rows)
            fn=db.execute(BASE + "SELECT COUNT(*)::int FROM base WHERE local_day=%s::date",(day,)).fetchone()[0]
            agg["fixtures"]=int(fn or 0); daily.append({"date":day,**agg})

        league_raw = db.execute(BASE + "SELECT league_name,market,SUM(won::int)::int,COUNT(*)::int,AVG(confidence)::float FROM verdicts WHERE TRUE" + extra + " GROUP BY league_name,market",params).fetchall()
        league_map: dict[str,list[tuple[Any,...]]]={}
        for league,market,wins,total,mc in league_raw: league_map.setdefault(str(league),[]).append((market,wins,total,mc))
        league_rows=[]
        for league,rows in league_map.items():
            agg=_aggregate_rows(rows); league_rows.append({"league":league,"fixtures":None,**agg,"primary":{"wins":0,"losses":0,"total":0,"winrate":None,"confidence_interval_95":None,"markets":{}}})
        league_rows.sort(key=lambda x:x["evaluated_picks"],reverse=True)

    return {"status":"READY" if fixture_n else "INSUFFICIENT","timezone":str(main.BOGOTA),
            "methodology":"Prequential: solo snapshots READY creados antes del inicio; cada mercado evaluable cuenta como un pick.",
            "roi_status":"UNAVAILABLE_WITHOUT_HISTORICAL_ODDS","overall":overall,
            "confidence_segments":confidence_segments,"reliability_bins":reliability_bins,
            "by_league":league_rows[:30],"daily":daily}


def install() -> None:
    global _installed
    if _installed: return
    if main.DATABASE_URL and main.psycopg is not None:
        main.db_daily_performance = daily_postgres
        main.log.info("PostgreSQL-side daily performance aggregation active")
    _installed=True
