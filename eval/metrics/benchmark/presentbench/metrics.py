from __future__ import annotations

from typing import Any


def presentation_fundamentals_metric(verdicts: dict[str, Any], weights: dict[str, Any]) -> float:
    return _class_score_percent(verdicts["material_independent"]["1"], float(weights["material_independent"]["1"]))


def visual_design_and_layout_metric(verdicts: dict[str, Any], weights: dict[str, Any]) -> float:
    return _class_score_percent(verdicts["material_independent"]["2"], float(weights["material_independent"]["2"]))


def content_completeness_metric(verdicts: dict[str, Any], weights: dict[str, Any]) -> float:
    return _class_score_percent(verdicts["material_dependent"]["1"], float(weights["material_dependent"]["1"]))


def content_correctness_metric(verdicts: dict[str, Any], weights: dict[str, Any]) -> float:
    return _class_score_percent(verdicts["material_dependent"]["2"], float(weights["material_dependent"]["2"]))


def content_fidelity_metric(verdicts: dict[str, Any], weights: dict[str, Any]) -> float:
    return _class_score_percent(verdicts["material_dependent"]["3"], float(weights["material_dependent"]["3"]))


def presentbench_overall_metric(verdicts: dict[str, Any], weights: dict[str, Any]) -> float:
    total_weighted_points = (
        _class_points(verdicts["material_independent"]["1"], float(weights["material_independent"]["1"]))
        + _class_points(verdicts["material_independent"]["2"], float(weights["material_independent"]["2"]))
        + _class_points(verdicts["material_dependent"]["1"], float(weights["material_dependent"]["1"]))
        + _class_points(verdicts["material_dependent"]["2"], float(weights["material_dependent"]["2"]))
        + _class_points(verdicts["material_dependent"]["3"], float(weights["material_dependent"]["3"]))
    )
    total_possible = (
        float(weights["material_independent"]["1"])
        + float(weights["material_independent"]["2"])
        + float(weights["material_dependent"]["1"])
        + float(weights["material_dependent"]["2"])
        + float(weights["material_dependent"]["3"])
    )
    if total_possible <= 0:
        return 0.0
    return total_weighted_points / total_possible * 100.0


def _class_score_percent(class_verdicts: dict[str, Any], class_weight: float) -> float:
    if class_weight <= 0:
        return 0.0
    return _class_points(class_verdicts, class_weight) / class_weight * 100.0


def _class_points(class_verdicts: dict[str, Any], class_weight: float) -> float:
    yes_count = 0
    valid_count = 0
    for item in class_verdicts.values():
        answer = str((item or {}).get("answer") or "").strip().lower()
        if answer == "not applicable":
            continue
        if answer in {"yes", "no"}:
            valid_count += 1
            if answer == "yes":
                yes_count += 1
    if valid_count == 0:
        return 0.0
    return class_weight * yes_count / valid_count
