from egress_daily import _aggregate
from egress_reports import _summary


def test_daily_aggregate_contract():
    result = _aggregate([("1x2", 6, 10, 0.61), ("btts", 3, 5, 0.66)])
    assert result["evaluated_picks"] == 15
    assert result["wins"] == 9
    assert result["losses"] == 6
    assert result["winrate"] == 0.6
    assert result["markets"]["1x2"]["total"] == 10


def test_scorecard_summary_contract():
    row = (100, 1.0, .62, .48, .24, .66, .25, .23, .25, .20, 3.0, .21, 2.5, .22, 4.0, 80, 70, 60)
    result = _summary(row)
    assert result["n"] == 100
    assert result["accuracy_1x2"] == .48
    assert result["advanced"]["corners"]["n"] == 80
    assert result["advanced"]["shots"]["mae"] == 4.0
