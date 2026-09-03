"""Rigorous probabilistic evaluation for the S2S sports forecasting model.

Pure statistical diagnostics only. Measures calibration, discrimination,
uncertainty and temporal stability without odds or wagering logic.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Iterable

import main
from training_dataset import TrainingRow, build_dataset

EPS = 1e-12
CLASSES = ("home", "draw", "away")


def clamp(value: float, low: float = EPS, high: float = 1.0 - EPS) -> float:
    return max(low, min(high, value))


def _float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def probabilities(row: TrainingRow) -> tuple[float, float, float] | None:
    candidates = (row.prediction, row.snapshot)
    for source in candidates:
        nested = source.get("probabilities") if isinstance(source, dict) else None
        if isinstance(nested, dict):
            vals = tuple(_float(nested.get(key)) for key in CLASSES)
            if all(value is not None for value in vals):
                total = sum(float(value) for value in vals)
                if total > 0:
                    return tuple(float(value) / total for value in vals)  # type: ignore[return-value]
        vals = tuple(_float(source.get(f"p_{key}")) for key in CLASSES) if isinstance(source, dict) else (None, None, None)
        if all(value is not None for value in vals):
            total = sum(float(value) for value in vals)
            if total > 0:
                return tuple(float(value) / total for value in vals)  # type: ignore[return-value]
    return None


def actual_index(row: TrainingRow) -> int:
    return 0 if row.actual_home > row.actual_away else 1 if row.actual_home == row.actual_away else 2


def usable(rows: Iterable[TrainingRow]) -> list[tuple[tuple[float, float, float], int]]:
    result = []
    for row in rows:
        probs = probabilities(row)
        if probs is not None:
            result.append((probs, actual_index(row)))
    return result


def core_metrics(samples: list[tuple[tuple[float, float, float], int]]) -> dict[str, float | int | None]:
    if not samples:
        return {"n": 0}
    n = len(samples)
    correct = 0
    brier = log_loss = rps = entropy = confidence = 0.0
    for probs, actual in samples:
        correct += int(max(range(3), key=lambda index: probs[index]) == actual)
        log_loss -= math.log(clamp(probs[actual]))
        brier += sum((probs[index] - float(index == actual)) ** 2 for index in range(3))
        # Ranked Probability Score for ordered 1-X-2 categories.
        cdf_p1 = probs[0]
        cdf_p2 = probs[0] + probs[1]
        cdf_o1 = float(actual <= 0)
        cdf_o2 = float(actual <= 1)
        rps += ((cdf_p1 - cdf_o1) ** 2 + (cdf_p2 - cdf_o2) ** 2) / 2.0
        entropy -= sum(p * math.log(clamp(p)) for p in probs)
        confidence += max(probs)
    return {
        "n": n,
        "accuracy_1x2": correct / n,
        "brier_1x2": brier / n,
        "log_loss_1x2": log_loss / n,
        "rps_1x2": rps / n,
        "mean_entropy_nats": entropy / n,
        "mean_confidence": confidence / n,
    }


def climatology(samples: list[tuple[tuple[float, float, float], int]]) -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    counts = [1.0, 1.0, 1.0]
    for _, actual in samples:
        counts[actual] += 1.0
    total = sum(counts)
    probs = tuple(value / total for value in counts)
    baseline = [(probs, actual) for _, actual in samples]
    metrics = core_metrics(baseline)
    return {"probabilities": dict(zip(CLASSES, probs)), **metrics}


def top_label_ece(samples: list[tuple[tuple[float, float, float], int]], bins: int = 10) -> dict[str, Any]:
    buckets: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    for probs, actual in samples:
        predicted = max(range(3), key=lambda index: probs[index])
        confidence = probs[predicted]
        bucket = min(int(confidence * bins), bins - 1)
        buckets[bucket].append((confidence, float(predicted == actual)))
    n = max(len(samples), 1)
    rows = []
    for bucket in range(bins):
        values = buckets.get(bucket, [])
        if not values:
            continue
        avg_p = mean(value[0] for value in values)
        avg_y = mean(value[1] for value in values)
        rows.append({"bin": bucket, "n": len(values), "mean_confidence": avg_p, "accuracy": avg_y, "gap": abs(avg_p - avg_y)})
    return {"ece": sum(row["n"] * row["gap"] for row in rows) / n, "bins": rows}


def classwise_ece(samples: list[tuple[tuple[float, float, float], int]], bins: int = 10) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for class_index, name in enumerate(CLASSES):
        buckets: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
        for probs, actual in samples:
            p = probs[class_index]
            buckets[min(int(p * bins), bins - 1)].append((p, float(actual == class_index)))
        rows = []
        for bucket in range(bins):
            values = buckets.get(bucket, [])
            if not values:
                continue
            avg_p = mean(value[0] for value in values)
            avg_y = mean(value[1] for value in values)
            rows.append({"bin": bucket, "n": len(values), "mean_probability": avg_p, "observed_frequency": avg_y, "gap": abs(avg_p - avg_y)})
        result[name] = {"ece": sum(row["n"] * row["gap"] for row in rows) / max(len(samples), 1), "bins": rows}
    result["macro_ece"] = mean(result[name]["ece"] for name in CLASSES) if samples else None
    return result


def _solve_2x2(a: float, b: float, c: float, d: float, x: float, y: float) -> tuple[float, float] | None:
    det = a * d - b * c
    if abs(det) < 1e-12:
        return None
    return ((x * d - b * y) / det, (a * y - x * c) / det)


def calibration_intercept_slope(samples: list[tuple[tuple[float, float, float], int]], class_index: int) -> dict[str, Any]:
    # Logistic recalibration: logit(P(Y=1)) = alpha + beta * logit(p_model).
    if len(samples) < 30:
        return {"intercept": None, "slope": None, "n": len(samples)}
    alpha, beta = 0.0, 1.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for probs, actual in samples:
            p = clamp(probs[class_index], 1e-6, 1 - 1e-6)
            x = math.log(p / (1 - p))
            y = float(actual == class_index)
            eta = max(-30.0, min(30.0, alpha + beta * x))
            q = 1.0 / (1.0 + math.exp(-eta))
            w = max(q * (1 - q), 1e-9)
            residual = y - q
            g0 += residual
            g1 += residual * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        step = _solve_2x2(h00, h01, h01, h11, g0, g1)
        if step is None:
            break
        da, db = step
        alpha += da
        beta += db
        if abs(da) + abs(db) < 1e-8:
            break
    return {"intercept": alpha, "slope": beta, "n": len(samples), "ideal": {"intercept": 0.0, "slope": 1.0}}


def calibration_slopes(samples: list[tuple[tuple[float, float, float], int]]) -> dict[str, Any]:
    return {name: calibration_intercept_slope(samples, index) for index, name in enumerate(CLASSES)}


def brier_decomposition(samples: list[tuple[tuple[float, float, float], int]], bins: int = 10) -> dict[str, Any]:
    # Murphy decomposition applied one-vs-rest per class.
    if not samples:
        return {"n": 0}
    output: dict[str, Any] = {}
    for class_index, name in enumerate(CLASSES):
        prevalence = mean(float(actual == class_index) for _, actual in samples)
        buckets: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
        for probs, actual in samples:
            p = probs[class_index]
            buckets[min(int(p * bins), bins - 1)].append((p, float(actual == class_index)))
        reliability = resolution = 0.0
        for values in buckets.values():
            weight = len(values) / len(samples)
            p_bar = mean(v[0] for v in values)
            y_bar = mean(v[1] for v in values)
            reliability += weight * (p_bar - y_bar) ** 2
            resolution += weight * (y_bar - prevalence) ** 2
        uncertainty = prevalence * (1 - prevalence)
        output[name] = {
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
            "reconstructed_brier": reliability - resolution + uncertainty,
        }
    output["macro"] = {
        key: mean(output[name][key] for name in CLASSES)
        for key in ("reliability", "resolution", "uncertainty", "reconstructed_brier")
    }
    return output


def sharpness(samples: list[tuple[tuple[float, float, float], int]]) -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    maxima = [max(probs) for probs, _ in samples]
    entropies = [-sum(p * math.log(clamp(p)) for p in probs) for probs, _ in samples]
    return {
        "n": len(samples),
        "max_probability_mean": mean(maxima),
        "max_probability_std": pstdev(maxima),
        "entropy_mean": mean(entropies),
        "entropy_std": pstdev(entropies),
        "fraction_confidence_gte_0_50": sum(value >= 0.50 for value in maxima) / len(maxima),
        "fraction_confidence_gte_0_60": sum(value >= 0.60 for value in maxima) / len(maxima),
        "fraction_confidence_gte_0_70": sum(value >= 0.70 for value in maxima) / len(maxima),
    }


def temporal_stability(samples: list[tuple[tuple[float, float, float], int]], blocks: int = 6) -> dict[str, Any]:
    if not samples:
        return {"blocks": []}
    width = max(1, len(samples) // blocks)
    reports = []
    for index in range(blocks):
        start = index * width
        end = len(samples) if index == blocks - 1 else min(len(samples), (index + 1) * width)
        if start >= len(samples):
            break
        report = core_metrics(samples[start:end])
        report["block"] = index + 1
        reports.append(report)
    briers = [float(row["brier_1x2"]) for row in reports if row.get("brier_1x2") is not None]
    loglosses = [float(row["log_loss_1x2"]) for row in reports if row.get("log_loss_1x2") is not None]
    return {
        "blocks": reports,
        "brier_std_across_blocks": pstdev(briers) if len(briers) > 1 else 0.0,
        "logloss_std_across_blocks": pstdev(loglosses) if len(loglosses) > 1 else 0.0,
    }


def bootstrap_ci(samples: list[tuple[tuple[float, float, float], int]], repeats: int = 300, seed: int = 8502) -> dict[str, Any]:
    if len(samples) < 30 or repeats <= 0:
        return {"repeats": 0}
    rng = random.Random(seed)
    names = ("accuracy_1x2", "brier_1x2", "log_loss_1x2", "rps_1x2")
    distributions: dict[str, list[float]] = {name: [] for name in names}
    n = len(samples)
    for _ in range(repeats):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        metrics = core_metrics(resample)
        for name in names:
            distributions[name].append(float(metrics[name]))
    def percentile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        low = int(math.floor(pos)); high = int(math.ceil(pos))
        if low == high:
            return ordered[low]
        return ordered[low] * (high - pos) + ordered[high] * (pos - low)
    return {
        "repeats": repeats,
        "seed": seed,
        name: {"low_95": percentile(values, 0.025), "high_95": percentile(values, 0.975)}
        for name, values in distributions.items()
    }


def full_evaluation(rows: list[TrainingRow], bootstrap_repeats: int = 300) -> dict[str, Any]:
    samples = usable(rows)
    core = core_metrics(samples)
    base = climatology(samples)
    model_brier = float(core.get("brier_1x2") or 0.0)
    baseline_brier = float(base.get("brier_1x2") or 0.0)
    model_logloss = float(core.get("log_loss_1x2") or 0.0)
    baseline_logloss = float(base.get("log_loss_1x2") or 0.0)
    return {
        "model_version": rows[0].model_version if rows else None,
        "n_rows": len(rows),
        "n_probability_rows": len(samples),
        "core": core,
        "climatology_baseline": base,
        "skill": {
            "brier_skill_score": 1.0 - model_brier / baseline_brier if baseline_brier else None,
            "logloss_improvement_pct": 100.0 * (baseline_logloss - model_logloss) / baseline_logloss if baseline_logloss else None,
        },
        "top_label_calibration": top_label_ece(samples),
        "classwise_calibration": classwise_ece(samples),
        "calibration_intercept_slope": calibration_slopes(samples),
        "brier_decomposition": brier_decomposition(samples),
        "sharpness": sharpness(samples),
        "temporal_stability": temporal_stability(samples),
        "bootstrap_ci95": bootstrap_ci(samples, repeats=bootstrap_repeats),
        "interpretation_contract": {
            "lower_is_better": ["brier_1x2", "log_loss_1x2", "rps_1x2", "ece", "reliability"],
            "ideal_calibration_intercept": 0.0,
            "ideal_calibration_slope": 1.0,
            "positive_brier_skill_is_better_than_climatology": True,
            "chronological_input_required": True,
        },
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate S2S probabilistic model rigorously")
    parser.add_argument("--model-version", default=main.MODEL_VERSION)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    main.init_db()
    rows, dataset_summary = build_dataset(args.model_version or None)
    result = full_evaluation(rows, bootstrap_repeats=max(0, args.bootstrap))
    result["dataset"] = dataset_summary
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()
