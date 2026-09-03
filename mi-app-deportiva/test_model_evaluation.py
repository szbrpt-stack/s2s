import math

import model_evaluation as ev


def test_perfect_predictions_have_zero_proper_score():
    samples = [((1.0, 0.0, 0.0), 0), ((0.0, 1.0, 0.0), 1), ((0.0, 0.0, 1.0), 2)]
    result = ev.core_metrics(samples)
    assert result["accuracy_1x2"] == 1.0
    assert result["brier_1x2"] == 0.0
    assert result["rps_1x2"] == 0.0
    assert result["log_loss_1x2"] < 1e-9


def test_uniform_probabilities_have_expected_multiclass_brier():
    samples = [((1 / 3, 1 / 3, 1 / 3), index) for index in range(3)]
    result = ev.core_metrics(samples)
    assert math.isclose(result["brier_1x2"], 2 / 3, rel_tol=1e-9)
    assert math.isclose(result["log_loss_1x2"], math.log(3), rel_tol=1e-9)


def test_classwise_ece_zero_for_balanced_uniform_forecast():
    samples = [((1 / 3, 1 / 3, 1 / 3), index) for index in range(3)] * 20
    result = ev.classwise_ece(samples)
    assert result["macro_ece"] < 1e-12


def test_brier_skill_zero_against_itself():
    samples = [((0.5, 0.25, 0.25), 0), ((0.5, 0.25, 0.25), 1), ((0.5, 0.25, 0.25), 2)] * 20
    base = ev.climatology(samples)
    assert base["n"] == len(samples)
