"""Post-inference catalogue comparison; reference labels never enter reasoning."""
from __future__ import annotations
from collections import Counter
import csv
import json
from pathlib import Path

CATEGORIES = ("A", "B", "C", "X", "insufficient_data", "unavailable")


def grade_metrics(pairs: list[tuple[str, str]]) -> dict:
    matrix = {actual: {prediction: 0 for prediction in CATEGORIES}
              for actual in ("A", "B", "C", "X")}
    for actual, prediction in pairs:
        matrix[actual][prediction] += 1
    definite = [(actual, prediction) for actual, prediction in pairs if prediction in {"A", "B", "C", "X"}]
    correct = sum(actual == prediction for actual, prediction in pairs)
    support = Counter(actual for actual, _ in pairs)
    return {"n": len(pairs), "definite_n": len(definite), "correct_n": correct,
            "coverage": len(definite) / len(pairs) if pairs else None,
            "accuracy_all_targets": correct / len(pairs) if pairs else None,
            "accuracy_definite_only": correct / len(definite) if definite else None,
            "recall_by_grade": {grade: matrix[grade][grade] / support[grade]
                                if support[grade] else None for grade in matrix},
            "confusion_matrix": matrix}


def evaluate_catalogue(catalogue_path: Path, results_root: Path) -> dict:
    with catalogue_path.open(encoding="utf-8", newline="") as stream:
        catalogue = list(csv.DictReader(stream))
    selected = [row for row in catalogue if row.get("grade") in {"A", "B", "C", "X"}]
    ids = [row['id_str'] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalogue contains duplicate target IDs")
    results = {}
    for path in sorted(results_root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        object_id = summary.get("object_id")
        if object_id in results:
            raise ValueError("Multiple runs for a target: select a single result directory")
        results[object_id] = (summary, path)
    targets = []
    for row in selected:
        object_id = row["id_str"]
        entry = results.get(object_id)
        rule = llm = "unavailable"
        error = "missing_result"
        path = None
        if entry:
            summary, path = entry
            audit = summary.get("decision_audit") or {}
            if audit.get("audit_passed"):
                category = (summary.get("expert_assessment") or {}).get("vote_category")
                rule = category if category in CATEGORIES else "insufficient_data"
            review = summary.get("llm_assessment") or {}
            if review.get("status") == "completed" and (review.get("verification") or {}).get("passed"):
                category = (review.get("assessment") or {}).get("vote_category")
                llm = category if category in CATEGORIES else "unavailable"
            error = review.get("error") if review else "llm_not_requested"
        targets.append({"id_str": object_id, "reference_grade": row["grade"],
                        "rules_grade": rule, "llm_grade": llm,
                        "llm_error": error, "summary_path": str(path) if path else None})
    return {"evaluation_version": 1, "reference_catalogue": str(catalogue_path.resolve()),
            "results_root": str(results_root.resolve()),
            "rules": grade_metrics([(t["reference_grade"], t["rules_grade"]) for t in targets]),
            "deepseek": grade_metrics([(t["reference_grade"], t["llm_grade"]) for t in targets]),
            "targets": targets,
            "limitations": ["Pilot comparison, not independent validation or calibration.",
                            "Samples drawn from lens.zip are selected for available modelling data.",
                            "A/B-only samples cannot estimate contamination or non-lens rejection.",
                            "Abstentions and missing results remain in the all-target denominator."]}
