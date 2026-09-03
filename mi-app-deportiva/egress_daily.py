"""PostgreSQL-side daily performance aggregation.

Replaces the legacy 5,000-row prediction_outcomes download with compact SQL
aggregates while preserving the public report shape. SQLite keeps the legacy path.
"""
from __future__ import annotations
from typing import Any
import main
_installed=False

def _wilson(wins:int,total:int)->list[float]|None:return main._wilson_interval(wins,total)
def _filters(date_from:str|None,date_to:str|None)->tuple[str,list[Any]]:
    clauses=[];params=[]
    if date_from:clauses.append("local_day >= %s::date");params.append(date_from)
    if date_to:clauses.append("local_day <= %s::date");params.append(date_to)
    return ((" AND "+" AND ".join(clauses)) if clauses else "",params)
BASE="""
WITH recent AS (SELECT fixture_id,kickoff_utc,prediction::jsonb p,metrics::jsonb m,actual_home,actual_away,(kickoff_utc::timestamptz AT TIME ZONE 'America/Bogota')::date local_day FROM prediction_outcomes ORDER BY kickoff_utc::timestamptz DESC LIMIT 5000),
base AS (SELECT *,COALESCE(NULLIF(p->>'league_name',''),NULLIF(p->>'liga',''),'Sin liga') league_name,COALESCE(NULLIF(p->>'p_home','')::float,0)/100.0 ph,COALESCE(NULLIF(p->>'p_draw','')::float,0)/100.0 pd,COALESCE(NULLIF(p->>'p_away','')::float,0)/100.0 pa,NULLIF(m->>'probability_over25','')::float p_over25,NULLIF(m->>'probability_btts','')::float p_btts,NULLIF(m->>'primary_correct','')::int primary_correct,COALESCE(NULLIF(m->>'primary_market',''),'unknown') primary_market FROM recent),
verdicts AS (
SELECT local_day,league_name,'1x2'::text market,((CASE WHEN ph>=pd AND ph>=pa THEN 0 WHEN pd>=pa THEN 1 ELSE 2 END)=(CASE WHEN actual_home>actual_away THEN 0 WHEN actual_home=actual_away THEN 1 ELSE 2 END)) won,GREATEST(ph,pd,pa) confidence FROM base WHERE ph+pd+pa>0
UNION ALL SELECT local_day,league_name,'goals_2_5',((actual_home+actual_away>=3)=(p_over25>=0.5)),GREATEST(p_over25,1-p_over25) FROM base WHERE p_over25 IS NOT NULL
UNION ALL SELECT local_day,league_name,'btts',((actual_home>0 AND actual_away>0)=(p_btts>=0.5)),GREATEST(p_btts,1-p_btts) FROM base WHERE p_btts IS NOT NULL
UNION ALL SELECT local_day,league_name,'corners',((NULLIF(m->>'corners_actual','')::float>NULLIF(p#>>'{metrics,corners_8_5,line}','')::float)=(NULLIF(p#>>'{metrics,corners_8_5,probability_over}','')::float>=0.5)),GREATEST(NULLIF(p#>>'{metrics,corners_8_5,probability_over}','')::float,1-NULLIF(p#>>'{metrics,corners_8_5,probability_over}','')::float) FROM base WHERE NULLIF(m->>'corners_actual','') IS NOT NULL AND NULLIF(p#>>'{metrics,corners_8_5,line}','') IS NOT NULL AND NULLIF(p#>>'{metrics,corners_8_5,probability_over}','') IS NOT NULL
UNION ALL SELECT local_day,league_name,'cards',((NULLIF(m->>'cards_actual','')::float>NULLIF(p#>>'{metrics,cards_4_5,line}','')::float)=(NULLIF(p#>>'{metrics,cards_4_5,probability_over}','')::float>=0.5)),GREATEST(NULLIF(p#>>'{metrics,cards_4_5,probability_over}','')::float,1-NULLIF(p#>>'{metrics,cards_4_5,probability_over}','')::float) FROM base WHERE NULLIF(m->>'cards_actual','') IS NOT NULL AND NULLIF(p#>>'{metrics,cards_4_5,line}','') IS NOT NULL AND NULLIF(p#>>'{metrics,cards_4_5,probability_over}','') IS NOT NULL
UNION ALL SELECT local_day,league_name,'shots',((NULLIF(m->>'shots_actual','')::float>NULLIF(p#>>'{metrics,shots_20_5,line}','')::float)=(NULLIF(p#>>'{metrics,shots_20_5,probability_over}','')::float>=0.5)),GREATEST(NULLIF(p#>>'{metrics,shots_20_5,probability_over}','')::float,1-NULLIF(p#>>'{metrics,shots_20_5,probability_over}','')::float) FROM base WHERE NULLIF(m->>'shots_actual','') IS NOT NULL AND NULLIF(p#>>'{metrics,shots_20_5,line}','') IS NOT NULL AND NULLIF(p#>>'{metrics,shots_20_5,probability_over}','') IS NOT NULL)
"""
def _aggregate(rows:list[tuple[Any,...]])->dict[str,Any]:
    markets={};w=t=0
    for market,wins,total,mc in rows:
        wins,total=int(wins or 0),int(total or 0);w+=wins;t+=total;markets[str(market)]={"wins":wins,"losses":total-wins,"total":total,"winrate":round(wins/total,4) if total else None,"confidence_interval_95":_wilson(wins,total),"mean_model_confidence":round(float(mc),4) if mc is not None else None}
    return {"evaluated_picks":t,"wins":w,"losses":t-w,"winrate":round(w/t,4) if t else None,"confidence_interval_95":_wilson(w,t),"markets":markets}
def daily_postgres(date_from:str|None=None,date_to:str|None=None)->dict[str,Any]:
    extra,params=_filters(date_from,date_to)
    with main.psycopg.connect(main.DATABASE_URL,connect_timeout=15,prepare_threshold=None) as db:
        fixture_n=int(db.execute(BASE+"SELECT COUNT(*)::int FROM base WHERE TRUE"+extra,params).fetchone()[0] or 0)
        overall=_aggregate(db.execute(BASE+"SELECT market,SUM(won::int)::int,COUNT(*)::int,AVG(confidence)::float FROM verdicts WHERE TRUE"+extra+" GROUP BY market",params).fetchall());overall["fixtures"]=fixture_n
        pr=db.execute(BASE+"SELECT primary_market,SUM(primary_correct)::int,COUNT(*)::int FROM base WHERE primary_correct IS NOT NULL"+extra+" GROUP BY primary_market",params).fetchall();pm={};pw=pt=0
        for market,wins,total in pr:wins,total=int(wins or 0),int(total or 0);pw+=wins;pt+=total;pm[str(market)]={"wins":wins,"losses":total-wins,"total":total,"winrate":round(wins/total,4) if total else None}
        overall["primary"]={"wins":pw,"losses":pt-pw,"total":pt,"winrate":round(pw/pt,4) if pt else None,"confidence_interval_95":_wilson(pw,pt),"markets":pm}
        cr=db.execute(BASE+", filtered AS (SELECT * FROM verdicts WHERE TRUE"+extra+") SELECT t.threshold,SUM((f.won)::int)::int,COUNT(f.*)::int FROM (VALUES (0.50),(0.55),(0.60),(0.65),(0.70),(0.75),(0.80)) t(threshold) LEFT JOIN filtered f ON f.confidence>=t.threshold GROUP BY t.threshold ORDER BY t.threshold",params).fetchall();confidence=[]
        for th,wins,total in cr:wins,total=int(wins or 0),int(total or 0);confidence.append({"threshold":float(th),"wins":wins,"losses":total-wins,"total":total,"winrate":round(wins/total,4) if total else None,"confidence_interval_95":_wilson(wins,total)})
        rr=db.execute(BASE+"SELECT LEAST(FLOOR(confidence*10)::int,9),COUNT(*)::int,AVG(confidence)::float,AVG(won::int)::float FROM verdicts WHERE TRUE"+extra+" GROUP BY 1 ORDER BY 1",params).fetchall();reliability=[{"from":int(b)/10,"to":(int(b)+1)/10,"total":int(n),"mean_confidence":round(float(mc),4),"observed_winrate":round(float(ow),4)} for b,n,mc,ow in rr]
        dr=db.execute(BASE+"SELECT local_day,market,SUM(won::int)::int,COUNT(*)::int,AVG(confidence)::float FROM verdicts WHERE TRUE"+extra+" GROUP BY local_day,market ORDER BY local_day DESC",params).fetchall();dm={}
        for day,market,wins,total,mc in dr:dm.setdefault(str(day),[]).append((market,wins,total,mc))
        df=dict(db.execute(BASE+"SELECT local_day::text,COUNT(*)::int FROM base WHERE TRUE"+extra+" GROUP BY local_day",params).fetchall());daily=[]
        for day,rows in dm.items():a=_aggregate(rows);a["fixtures"]=int(df.get(day,0));daily.append({"date":day,**a})
        lr=db.execute(BASE+"SELECT league_name,market,SUM(won::int)::int,COUNT(*)::int,AVG(confidence)::float FROM verdicts WHERE TRUE"+extra+" GROUP BY league_name,market",params).fetchall();lm={}
        for league,market,wins,total,mc in lr:lm.setdefault(str(league),[]).append((market,wins,total,mc))
        lf=dict(db.execute(BASE+"SELECT league_name,COUNT(*)::int FROM base WHERE TRUE"+extra+" GROUP BY league_name",params).fetchall());leagues=[]
        for league,rows in lm.items():a=_aggregate(rows);a["fixtures"]=int(lf.get(league,0));leagues.append({"league":league,**a,"primary":{"wins":0,"losses":0,"total":0,"winrate":None,"confidence_interval_95":None,"markets":{}}})
        leagues.sort(key=lambda x:x["evaluated_picks"],reverse=True)
    return {"status":"READY" if fixture_n else "INSUFFICIENT","timezone":str(main.BOGOTA),"methodology":"Prequential: solo snapshots READY creados antes del inicio; cada mercado evaluable cuenta como un pick.","roi_status":"UNAVAILABLE_WITHOUT_HISTORICAL_ODDS","overall":overall,"confidence_segments":confidence,"reliability_bins":reliability,"by_league":leagues[:30],"daily":daily}
def install()->None:
    global _installed
    if _installed:return
    if main.DATABASE_URL and main.psycopg is not None:main.db_daily_performance=daily_postgres;main.log.info("PostgreSQL-side daily performance aggregation active")
    _installed=True
