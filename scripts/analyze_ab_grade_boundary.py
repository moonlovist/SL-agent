"""Label-blind A/B feature extraction and small-sample rule audit.

The script never imports or changes LensAgent grading code.  It first extracts
measurements from saved observations without labels, then joins the directory
labels only for evaluation.  Candidate rules in ``candidate_rules`` are
deliberately short, interpretable stress tests rather than trained classifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def values(items: list[dict[str, Any]], key: str) -> list[float]:
    return [value for item in items if (value := finite(item.get(key))) is not None]


def maximum(items: list[dict[str, Any]], key: str) -> float | None:
    found = values(items, key)
    return max(found) if found else None


def minimum(items: list[dict[str, Any]], key: str) -> float | None:
    found = values(items, key)
    return min(found) if found else None


def med(items: list[dict[str, Any]], key: str) -> float | None:
    found = values(items, key)
    return median(found) if found else None


def number(item: dict[str, Any], key: str, missing: float) -> float:
    value = finite(item.get(key))
    return missing if value is None else value


def locate_observations(root: Path) -> list[Path]:
    paths = sorted(root.rglob("reasoning_observation.json"))
    if not paths:
        raise FileNotFoundError(f"No reasoning_observation.json below {root}")
    return paths


def extract_features(path: Path) -> dict[str, Any]:
    observation = json.loads(path.read_text(encoding="utf-8"))
    object_id = observation["object_id"]
    structures = list(observation.get("structures") or [])
    scene = observation.get("scene_geometry") or {}
    alternative = observation.get("alternative_indicators") or {}
    comparison = observation.get("foreground_model_comparison") or {}

    # Reproduce only the documented morphology membership definitions.  This
    # is local analysis, not a call into the grading implementation.
    disc_risk = [
        item
        for item in structures
        if number(item, "subtraction_core_overlap_fraction", 0.0) >= 0.25
        and number(item, "radial_fractional_width", 0.0) >= 0.55
        and number(item, "local_tangential_fraction", 1.0) <= 0.34
    ]
    disc_ids = {item["id"] for item in disc_risk}
    curved = [
        item
        for item in structures
        if number(item, "angular_occupied_deg", 0.0) >= 90.0
        and number(item, "radial_fractional_width", math.inf) <= 0.50
        and number(item, "local_shape_count", 0.0) >= 3
        and number(item, "local_tangential_fraction", 0.0) >= 0.67
        and finite(item.get("subtraction_core_overlap_fraction")) != 1.0
    ]
    curved_ids = {item["id"] for item in curved}
    tangential = [
        item
        for item in structures
        if item["id"] not in disc_ids
        and (
            item["id"] in curved_ids
            or (
                number(item, "tangential_misalignment_deg", math.inf) <= 35.0
                and number(item, "axis_ratio", 0.0) >= 1.5
                and number(item, "angular_extent_deg", 0.0) >= 20.0
            )
        )
    ]
    robust_tangential = [
        item
        for item in tangential
        if number(item, "integrated_snr", -math.inf) >= 4.0
        and number(item, "angular_extent_deg", 0.0) >= 30.0
        and number(item, "threshold_survival_fraction", 0.0) > 0.0
    ]
    clear_tangential = [
        item
        for item in tangential
        if number(item, "integrated_snr", -math.inf) >= 7.0
        and number(item, "angular_extent_deg", 0.0) >= 50.0
        and number(item, "axis_ratio", 0.0) >= 1.5
        and number(item, "tangential_misalignment_deg", math.inf) <= 25.0
    ]
    relevant = tangential + [item for item in curved if item["id"] not in {x["id"] for x in tangential}]
    if not relevant:
        relevant = structures
    relevant_candidate_ids = {
        candidate_id
        for item in relevant
        for candidate_id in item.get("source_candidate_ids", [])
    }

    relations = (
        observation.get("supplementary_diagnostics", {})
        .get("multiband", {})
        .get("relationship_audit", {})
    )
    colour_pairs = [
        item
        for item in relations.get("component_pairs", [])
        if {item.get("first"), item.get("second")} <= relevant_candidate_ids
    ]
    foreground_colours = [
        item
        for item in relations.get("candidate_vs_foreground", [])
        if item.get("candidate") in relevant_candidate_ids
    ]
    reliable_pairs = [
        item
        for item in colour_pairs
        if item.get("status")
        in {"consistent_with_achromatic_scaling", "inconclusive", "strongly_inconsistent"}
    ]
    consistent_pairs = [
        item for item in colour_pairs if item.get("status") == "consistent_with_achromatic_scaling"
    ]
    inconsistent_pairs = [
        item for item in colour_pairs if item.get("status") == "strongly_inconsistent"
    ]
    distinct_foreground = [
        item for item in foreground_colours if item.get("status") == "distinct_from_foreground"
    ]
    same_foreground = [
        item
        for item in foreground_colours
        if item.get("status") == "indistinguishable_from_foreground"
    ]

    positive_comparison = [
        item
        for item in relevant
        if number(item, "comparison_integrated_snr", -math.inf) >= 4.0
    ]
    non_core = [item for item in structures if not item.get("near_core")]
    candidate_systems = list(observation.get("candidate_systems") or [])
    strongest = max(
        tangential or structures,
        key=lambda item: (
            number(item, "angular_extent_deg", 0.0),
            number(item, "integrated_snr", 0.0),
        ),
        default={},
    )

    return {
        "object_id": object_id,
        "saved_grade": None,
        "structure_count": len(structures),
        "relevant_candidate_count": len(relevant_candidate_ids),
        "tangential_count": len(tangential),
        "robust_tangential_count": len(robust_tangential),
        "clear_tangential_count": len(clear_tangential),
        "foreground_disc_risk_count": len(disc_risk),
        "positive_comparison_count": len(positive_comparison),
        "non_core_fraction": len(non_core) / len(structures) if structures else 0.0,
        "common_radius_cv": finite(scene.get("common_radius_cv")),
        "total_angular_coverage_deg": finite(scene.get("total_angular_coverage_deg")),
        "mean_tangential_misalignment_deg": finite(scene.get("mean_tangential_misalignment_deg")),
        "tangential_alignment_fraction": finite(scene.get("tangential_alignment_fraction")),
        "max_structure_snr": maximum(structures, "integrated_snr"),
        "second_structure_snr": sorted(values(structures, "integrated_snr"), reverse=True)[1]
        if len(values(structures, "integrated_snr")) >= 2
        else None,
        "max_tangential_snr": maximum(tangential, "integrated_snr"),
        "max_tangential_extent_deg": maximum(tangential, "angular_extent_deg"),
        "max_tangential_axis_ratio": maximum(tangential, "axis_ratio"),
        "min_tangential_misalignment_deg": minimum(tangential, "tangential_misalignment_deg"),
        "max_tangential_local_fraction": maximum(tangential, "local_tangential_fraction"),
        "max_tangential_angular_occupied_deg": maximum(tangential, "angular_occupied_deg"),
        "median_structure_model_survival": med(structures, "foreground_comparison_survival_fraction"),
        "pooled_model_survival": finite(comparison.get("measured_structure_pixel_survival_fraction")),
        "residual_model_correlation": finite(comparison.get("residual_correlation")),
        "core_residual_fraction": finite(alternative.get("core_residual_fraction")),
        "tidal_asymmetry": finite(alternative.get("tidal_asymmetry")),
        "inner_connected_structure_count": finite(alternative.get("inner_connected_structure_count")),
        "candidate_system_count": len(candidate_systems),
        "max_candidate_system_components": max(
            (int(item.get("component_count", 0)) for item in candidate_systems), default=0
        ),
        "max_candidate_system_extent_deg": maximum(candidate_systems, "angular_extent_deg"),
        "colour_reliable_pair_count": len(reliable_pairs),
        "colour_consistent_pair_count": len(consistent_pairs),
        "colour_inconsistent_pair_count": len(inconsistent_pairs),
        "colour_usable_pair_fraction": len(reliable_pairs) / len(colour_pairs) if colour_pairs else 0.0,
        "colour_consistent_pair_fraction": len(consistent_pairs) / len(reliable_pairs)
        if reliable_pairs
        else None,
        "colour_inconsistent_pair_fraction": len(inconsistent_pairs) / len(reliable_pairs)
        if reliable_pairs
        else None,
        "colour_max_departure_sigma": maximum(reliable_pairs, "maximum_departure_sigma"),
        "colour_min_departure_sigma": minimum(reliable_pairs, "maximum_departure_sigma"),
        "foreground_distinct_count": len(distinct_foreground),
        "foreground_same_count": len(same_foreground),
        "foreground_distinct_fraction": len(distinct_foreground)
        / (len(distinct_foreground) + len(same_foreground))
        if distinct_foreground or same_foreground
        else None,
        "strongest_structure_id": strongest.get("id"),
    }


def add_saved_grade(row: dict[str, Any], observation_path: Path) -> None:
    summary_path = observation_path.with_name("summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row["saved_grade"] = summary.get("published_grade") or (
            summary.get("expert_assessment") or {}
        ).get("vote_category")


def has(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def candidate_rules() -> dict[str, Callable[[dict[str, Any]], bool]]:
    """Return fixed, scientifically interpretable A gates.

    ``True`` means A; ``False`` means B.  These are stress tests only.  The
    directory labels are not referenced by any rule.
    """

    def colour_ok(row: dict[str, Any]) -> bool:
        return row["colour_inconsistent_pair_count"] == 0

    def reliable_comparison(row: dict[str, Any]) -> bool:
        return row["positive_comparison_count"] >= 1

    def clear_single_arc(row: dict[str, Any], extent: float) -> bool:
        return bool(
            row["clear_tangential_count"] >= 1
            and (row["max_tangential_extent_deg"] or 0.0) >= extent
            and reliable_comparison(row)
        )

    def coherent_multiple(row: dict[str, Any], coverage: float) -> bool:
        return bool(
            row["tangential_count"] >= 2
            and has(row["common_radius_cv"])
            and row["common_radius_cv"] <= 0.30
            and (row["total_angular_coverage_deg"] or 0.0) >= coverage
            and row["non_core_fraction"] >= 0.50
            and reliable_comparison(row)
        )

    return {
        "R0_saved_grade": lambda row: row["saved_grade"] == "A",
        "R1_colour_cap_only": lambda row: colour_ok(row),
        "R2_clear_arc_90_or_coherent_multiple_90": lambda row: colour_ok(row)
        and (clear_single_arc(row, 90.0) or coherent_multiple(row, 90.0)),
        "R3_clear_arc_120_or_coherent_multiple_120": lambda row: colour_ok(row)
        and (clear_single_arc(row, 120.0) or coherent_multiple(row, 120.0)),
        "R4_two_tangential_common_radius": lambda row: colour_ok(row)
        and coherent_multiple(row, 90.0),
        "R5_arc90_plus_colour_support_or_multiple": lambda row: colour_ok(row)
        and (
            coherent_multiple(row, 90.0)
            or (
                clear_single_arc(row, 90.0)
                and (
                    row["colour_consistent_pair_count"] >= 1
                    or row["foreground_distinct_count"] >= 1
                )
            )
        ),
        "R6_arc90_or_multiple_colour_not_required": lambda row: (
            clear_single_arc(row, 90.0) or coherent_multiple(row, 90.0)
        ),
    }


def confusion(rows: list[dict[str, Any]], predict_a: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    errors: list[str] = []
    predictions: dict[str, str] = {}
    for row in rows:
        actual_a = row["known_label"] == "A"
        predicted_a = bool(predict_a(row))
        predictions[row["object_id"]] = "A" if predicted_a else "B"
        if actual_a and predicted_a:
            tp += 1
        elif not actual_a and not predicted_a:
            tn += 1
        elif not actual_a and predicted_a:
            fp += 1
            errors.append(row["object_id"])
        else:
            fn += 1
            errors.append(row["object_id"])
    sensitivity_a = tp / (tp + fn) if tp + fn else 0.0
    specificity_b = tn / (tn + fp) if tn + fp else 0.0
    return {
        "A_as_A": tp,
        "A_as_B": fn,
        "B_as_B": tn,
        "B_as_A": fp,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": (sensitivity_a + specificity_b) / 2.0,
        "errors": errors,
        "predictions": predictions,
    }


def balanced_accuracy(actual: list[bool], predicted: list[bool]) -> float:
    a_indices = [index for index, value in enumerate(actual) if value]
    b_indices = [index for index, value in enumerate(actual) if not value]
    a_recall = sum(predicted[index] for index in a_indices) / len(a_indices) if a_indices else 0.0
    b_recall = sum(not predicted[index] for index in b_indices) / len(b_indices) if b_indices else 0.0
    return (a_recall + b_recall) / 2.0


def stump_candidates(rows: list[dict[str, Any]], feature: str):
    observed = sorted({float(row[feature]) for row in rows if has(row.get(feature))})
    if not observed:
        return []
    thresholds = [observed[0] - 1e-9]
    thresholds += [(left + right) / 2.0 for left, right in zip(observed, observed[1:])]
    thresholds.append(observed[-1] + 1e-9)
    return [
        (threshold, direction, missing_a)
        for threshold in thresholds
        for direction in ("ge", "le")
        for missing_a in (False, True)
    ]


def stump_predict(row: dict[str, Any], feature: str, stump: tuple[float, str, bool]) -> bool:
    threshold, direction, missing_a = stump
    value = row.get(feature)
    if not has(value):
        return missing_a
    return float(value) >= threshold if direction == "ge" else float(value) <= threshold


def fit_stump(rows: list[dict[str, Any]], feature: str) -> tuple[float, str, bool]:
    actual = [row["known_label"] == "A" for row in rows]
    candidates = stump_candidates(rows, feature)
    scored = []
    for stump in candidates:
        predicted = [stump_predict(row, feature, stump) for row in rows]
        score = balanced_accuracy(actual, predicted)
        # Prefer a non-A missing branch and a central threshold on ties.
        scored.append((score, not stump[2], -abs(stump[0] - median([float(r[feature]) for r in rows if has(r.get(feature))])), stump))
    return max(scored, key=lambda item: item[:3])[3]


def leave_one_out_stump(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    actual = [row["known_label"] == "A" for row in rows]
    predicted: list[bool] = []
    for index, row in enumerate(rows):
        training = rows[:index] + rows[index + 1 :]
        predicted.append(stump_predict(row, feature, fit_stump(training, feature)))
    full_stump = fit_stump(rows, feature)
    threshold, direction, missing_a = full_stump
    return {
        "feature": feature,
        "loo_balanced_accuracy": balanced_accuracy(actual, predicted),
        "loo_accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(rows),
        "full_sample_direction": direction,
        "full_sample_threshold": threshold,
        "missing_predicts_A": missing_a,
        "loo_A_as_A": sum(a and p for a, p in zip(actual, predicted)),
        "loo_A_as_B": sum(a and not p for a, p in zip(actual, predicted)),
        "loo_B_as_B": sum(not a and not p for a, p in zip(actual, predicted)),
        "loo_B_as_A": sum(not a and p for a, p in zip(actual, predicted)),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-root", type=Path, required=True)
    parser.add_argument("--b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blind_rows: list[dict[str, Any]] = []
    labelled_rows: list[dict[str, Any]] = []
    for label, root in (("A", args.a_root), ("B", args.b_root)):
        for observation_path in locate_observations(root):
            row = extract_features(observation_path)
            blind_rows.append({key: value for key, value in row.items() if key != "saved_grade"})
            add_saved_grade(row, observation_path)
            labelled_rows.append({**row, "known_label": label})

    object_ids = [row["object_id"] for row in labelled_rows]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Duplicate object IDs found")
    if len(labelled_rows) != 31:
        raise ValueError(f"Expected 31 objects, found {len(labelled_rows)}")

    args.output.mkdir(parents=True, exist_ok=True)
    feature_names = list(blind_rows[0])
    write_csv(args.output / "blind_features.csv", blind_rows, feature_names)

    rules = candidate_rules()
    evaluations = {name: confusion(labelled_rows, rule) for name, rule in rules.items()}
    comparison_fields = [
        "object_id",
        "known_label",
        "saved_grade",
        "tangential_count",
        "clear_tangential_count",
        "common_radius_cv",
        "total_angular_coverage_deg",
        "max_tangential_extent_deg",
        "max_tangential_snr",
        "positive_comparison_count",
        "non_core_fraction",
        "foreground_disc_risk_count",
        "colour_reliable_pair_count",
        "colour_consistent_pair_count",
        "colour_inconsistent_pair_count",
        "foreground_distinct_count",
        "foreground_same_count",
    ] + [f"prediction_{name}" for name in rules]
    comparison_rows = []
    for row in labelled_rows:
        output_row = dict(row)
        for name, evaluation in evaluations.items():
            output_row[f"prediction_{name}"] = evaluation["predictions"][row["object_id"]]
        comparison_rows.append(output_row)
    write_csv(args.output / "labelled_rule_comparison.csv", comparison_rows, comparison_fields)

    matrix_rows = []
    for name, evaluation in evaluations.items():
        matrix_rows.append(
            {
                "rule": name,
                "A_as_A": evaluation["A_as_A"],
                "A_as_B": evaluation["A_as_B"],
                "B_as_B": evaluation["B_as_B"],
                "B_as_A": evaluation["B_as_A"],
                "accuracy": evaluation["accuracy"],
                "balanced_accuracy": evaluation["balanced_accuracy"],
                "error_ids": ";".join(evaluation["errors"]),
            }
        )
    write_csv(
        args.output / "confusion_matrices.csv",
        matrix_rows,
        list(matrix_rows[0]),
    )

    excluded_from_stumps = {"object_id", "saved_grade", "strongest_structure_id"}
    numeric_features = [
        name
        for name in feature_names
        if name not in excluded_from_stumps
        and any(has(row.get(name)) for row in labelled_rows)
    ]
    stump_results = sorted(
        (leave_one_out_stump(labelled_rows, name) for name in numeric_features),
        key=lambda item: (item["loo_balanced_accuracy"], item["loo_accuracy"]),
        reverse=True,
    )
    write_csv(args.output / "single_feature_loo_audit.csv", stump_results, list(stump_results[0]))

    lines = [
        "# A/B 等级边界盲审（31 个样本）",
        "",
        "## 方法边界",
        "",
        "- 第一步仅从保存的 `reasoning_observation.json` 提取测量量，不读取 A/B 目录标签。",
        "- 第二步才把 20 个已知 A 和 11 个已知 B 标签接回，用于混淆矩阵。",
        "- 所有候选规则都是可解释的压力测试，不是已校准分类器，也不改 LensAgent。",
        "- 这批样本既用于提出问题又用于比较规则，结果只能用于选下一步，不能作为泛化性能。",
        "",
        "## 候选规则混淆矩阵",
        "",
        "| 规则 | A→A | A→B | B→B | B→A | 准确率 | 平衡准确率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, evaluation in evaluations.items():
        lines.append(
            f"| {name} | {evaluation['A_as_A']} | {evaluation['A_as_B']} | "
            f"{evaluation['B_as_B']} | {evaluation['B_as_A']} | "
            f"{evaluation['accuracy']:.3f} | {evaluation['balanced_accuracy']:.3f} |"
        )

    r5_predictions = evaluations["R5_arc90_plus_colour_support_or_multiple"]["predictions"]
    r5_recovered_b = [
        row["object_id"]
        for row in labelled_rows
        if row["known_label"] == "B" and r5_predictions[row["object_id"]] == "B"
    ]
    r5_remaining_b = [
        row["object_id"]
        for row in labelled_rows
        if row["known_label"] == "B" and r5_predictions[row["object_id"]] == "A"
    ]
    r5_lost_a = [
        row["object_id"]
        for row in labelled_rows
        if row["known_label"] == "A" and r5_predictions[row["object_id"]] == "B"
    ]

    lines += [
        "",
        "## 主要结果",
        "",
        "- R0 与 R1 完全相同：本批保存等级实际上等价于‘只有强颜色矛盾封顶 B’；形态门槛几乎没有继续区分 A/B。",
        f"- R5 是预先写下的候选规则中平衡准确率最高者（{evaluations['R5_arc90_plus_colour_support_or_multiple']['balanced_accuracy']:.3f}）："
        f"把 {len(r5_recovered_b)}/11 个已知 B 拉回 B，但同时把 {len(r5_lost_a)}/20 个已知 A 压到 B。",
        "- 因而 R5 仍不能实施：它只是把错误从‘B 大量升 A’换成‘A 大量降 B’。",
        "- 单特征留一法中最强的是前景模型间的结构存活率，而不是颜色、弧长、切向角或共同半径；它适合触发人工/模型复核，不适合作为天体物理否决。",
        "- `candidate_system_count` 较多、`tangential_alignment_fraction` 较高反而更偏向本批 B，说明当前分割会把盘、残光或一条弧拆成多个看似相干的结构；继续收紧这些旧标量无法解决根因。",
        "",
        "R5 拉回的 B：",
        "",
        "- " + "\n- ".join(r5_recovered_b),
        "",
        "R5 仍误升为 A 的 B：",
        "",
        "- " + "\n- ".join(r5_remaining_b),
        "",
        "R5 新误降为 B 的 A：",
        "",
        "- " + "\n- ".join(r5_lost_a),
        "",
        "规则含义：",
        "",
        "- R0：保存结果，作为现状基线。",
        "- R1：只有候选像间强颜色矛盾才阻止 A；这是当前颜色上限的单独效果。",
        "- R2：无强颜色矛盾，且存在模型间仍为正信号的 ≥90° 清晰切向弧，或满足共同半径、覆盖角和非核心比例的多像系统。",
        "- R3：R2 的更严格 120° 版本。",
        "- R4：只允许几何相干多像进入 A，不允许单弧独立进入 A。",
        "- R5：多像可直接进入；单弧除 R2 条件外还要求可靠颜色支持。",
        "- R6：与 R2 相同，但移除颜色矛盾上限，用来隔离颜色规则的净影响。",
        "",
        "## 逐对象错误清单",
        "",
    ]
    for name, evaluation in evaluations.items():
        errors = evaluation["errors"]
        lines.append(f"### {name}")
        lines.append("")
        lines.append("- 无。" if not errors else "- " + "\n- ".join(errors))
        lines.append("")

    lines += [
        "## 单一测量量的留一法压力测试",
        "",
        "每次留出一个对象，只用其余 30 个对象选择该测量量的最佳单阈值及方向，再预测留出的对象。"
        "这比直接在 31 个对象上选阈值更严格，但特征列表本身仍由本批数据提出，不能当作独立验证。",
        "",
        "| 测量量 | 留一平衡准确率 | 留一准确率 | A→A | A→B | B→B | B→A | 全样本探索阈值 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in stump_results[:12]:
        direction = "≥" if item["full_sample_direction"] == "ge" else "≤"
        missing = "A" if item["missing_predicts_A"] else "B"
        lines.append(
            f"| {item['feature']} | {item['loo_balanced_accuracy']:.3f} | "
            f"{item['loo_accuracy']:.3f} | {item['loo_A_as_A']} | {item['loo_A_as_B']} | "
            f"{item['loo_B_as_B']} | {item['loo_B_as_A']} | A if {direction} "
            f"{item['full_sample_threshold']:.4g}; 缺失→{missing} |"
        )

    lines += [
        "## 审计解释注意事项",
        "",
        "- `candidate_systems` 是搜索路线关联，不是已验证多像系统，因此没有把其数量直接当作 A 证据。",
        "- 候选与前景颜色相同没有作为负证据；前景残光会把源颜色向前景颜色拉近。",
        "- 两种前景模型来自同一曝光，模型间存活仅表示减光敏感性，不是独立天体物理证据。",
        "- R2–R6 使用的 90°/120°、S/N 和半径离散阈值均是诊断性候选值，不是普适物理常数。",
        "",
        "## 建议的下一项新测量（尚未实现）",
        "",
        "现有标量不足以安全重现 A/B。下一项应增加‘弧脊线一致性’而不是继续调整颜色阈值：沿候选源的脊线逐段测量"
        "曲率中心散布、切向方向连续性、宽度/PSF 比以及不同前景模型下脊线位置漂移。真正的弧允许亮度不均匀，"
        "但几何脊线应围绕同一质量中心连续；前景盘、旋臂和减光残差通常更容易在这些联合量上暴露。"
        "颜色仅在质量门通过后作为支持或矛盾证据，缺失/受污染颜色不投票。",
        "",
        "## 科学依据",
        "",
        "- Euclid ELSE 的 A/B 是人工置信等级：A 为无需额外信息的明确透镜，B 为仍需额外信息的可能透镜；并非一条颜色阈值定义的物理类别。",
        "- 强透镜检索通常联合曲率、相对前景中心的切向性、反像/多像关系、颜色以及盘/旋臂等替代解释。",
        "- 参考：https://www.aanda.org/articles/aa/pdf/forth/aa51868-24.pdf",
        "- 参考：https://academic.oup.com/pasj/article/70/SP1/S29/4102352",
    ]
    (args.output / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "object_count": len(labelled_rows),
        "known_A": sum(row["known_label"] == "A" for row in labelled_rows),
        "known_B": sum(row["known_label"] == "B" for row in labelled_rows),
        "source_roots": {"A": str(args.a_root), "B": str(args.b_root)},
        "rules": {
            name: {key: value for key, value in evaluation.items() if key != "predictions"}
            for name, evaluation in evaluations.items()
        },
    }
    (args.output / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
