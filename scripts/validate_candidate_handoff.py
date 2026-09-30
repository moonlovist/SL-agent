"""Capture a reproducible pre-edit baseline; replay the opt-in handoff trial."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def assess(document):
    from expert_agent.core.state import ExpertState
    from expert_agent.core.workflow import ExpertWorkflow
    with redirect_stdout(io.StringIO()):
        return ExpertWorkflow().run(ExpertState(
            image_path=document.input_path, object_id=document.object_id,
            observation_document=document,
        ))


def capture(output):
    from expert_agent.schemas.observation import ObservationDocument
    if output.exists():
        raise FileExistsError(f"Baseline must be new: {output}")
    records = []
    for folder in sorted(ROOT.glob("outputs*")):
        if folder.name.startswith("outputs_candidate_handoff_trial_"):
            continue  # Validation replays are not new sample grading runs.
        for path in sorted(folder.rglob("summary.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("object_id"):
                records.append(dict(
                    object_id=value["object_id"], summary=str(path),
                    vote=(value.get("expert_assessment") or {}).get("vote_category"),
                ))
    votes = [r for r in records if r["vote"] in {"A", "B", "C", "X", "insufficient_for_grading"}]
    selected = {}
    for row in records:
        path = Path(row["summary"]).with_name("observation.json")
        if path.exists():
            oid = row["object_id"]
            if oid not in selected or path.stat().st_mtime > selected[oid].stat().st_mtime:
                selected[oid] = path
    output.mkdir(parents=True)
    hashes = {}
    for path in sorted((ROOT / "src" / "expert_agent").rglob("*.py")):
        relative = path.relative_to(ROOT / "src")
        target = output / "baseline_src" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshots = []
    for oid, path in sorted(selected.items()):
        raw = json.loads(path.read_text(encoding="utf-8"))
        document = ObservationDocument.model_validate(raw)
        result = assess(document)
        snapshots.append(dict(
            object_id=oid, observation_path=str(path), observation=raw,
            observation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            assessment=result.expert_assessment.model_dump(mode="json"),
            audit=result.decision_audit.model_dump(mode="json"),
        ))
    inventory = dict(
        categorical_run_count=len(votes),
        categorical_unique_count=len({r["object_id"] for r in votes}),
        all_summary_unique_count=len({r["object_id"] for r in records}),
        paired_observation_count=len(snapshots),
        runs=records, code_sha256=hashes,
    )
    write_json(output / "inventory.json", inventory)
    write_json(output / "baseline.json", snapshots)
    print(json.dumps({k: v for k, v in inventory.items() if k.endswith("count")}, ensure_ascii=False))


def compare(output):
    from expert_agent.schemas.observation import ObservationDocument
    from expert_agent.vision.candidate_handoff import attach_candidate_review
    from expert_agent.vision.summary_cn import write_observation_summary_cn

    cache = output / ".matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))

    baseline = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    historical = []
    for path in sorted((ROOT / "outputs_v0180_A_samples_21_30_full").glob("*/observation.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = assess(ObservationDocument.model_validate(raw))
        published = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        historical.append(dict(
            object_id=raw["object_id"], observation_path=str(path), observation=raw,
            observation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            assessment=state.expert_assessment.model_dump(mode="json"),
            audit=state.decision_audit.model_dump(mode="json"),
            historical_vote=published["expert_assessment"]["vote_category"],
        ))
    rows = []
    for cohort, snapshots in (("latest_unique", baseline), ("historical_21_30", historical)):
        for snapshot in snapshots:
            doc = ObservationDocument.model_validate(snapshot["observation"])
            original = doc.model_dump(mode="json")
            default = assess(doc)
            folder = output / cohort / doc.object_id
            folder.mkdir(parents=True, exist_ok=True)
            trial_doc = attach_candidate_review(
                doc, output_dir=folder if doc.object_id.endswith(("3316", "7877", "1682", "6283", "0963")) else None,
            )
            # Reparse JSON to test the persisted observation-only reason path.
            trial_doc = ObservationDocument.model_validate_json(trial_doc.model_dump_json())
            trial = assess(trial_doc)
            entries = trial_doc.candidate_handoff_review.entries
            preserved = all(getattr(trial_doc, field) == getattr(doc, field) for field in (
                "structures", "measurements", "scene_geometry", "alternative_indicators",
                "model_candidates", "candidate_structure_audits", "foreground_model_comparison",
            ))
            row = dict(
                cohort=cohort, object_id=doc.object_id,
                historical_vote=snapshot.get("historical_vote"),
                baseline_vote=snapshot["assessment"]["vote_category"],
                default_vote=default.expert_assessment.vote_category,
                trial_vote=trial.expert_assessment.vote_category,
                default_matches_frozen_assessment=default.expert_assessment.model_dump(mode="json") == snapshot["assessment"],
                default_matches_frozen_audit=default.decision_audit.model_dump(mode="json") == snapshot["audit"],
                trial_assessment_unchanged=trial.expert_assessment == default.expert_assessment,
                trial_audit_unchanged=trial.decision_audit == default.decision_audit,
                audit_passed=trial.decision_audit.audit_passed,
                formal_measurements_unchanged=preserved,
                input_document_unchanged=doc.model_dump(mode="json") == original,
                source_file_unchanged=hashlib.sha256(Path(snapshot["observation_path"]).read_bytes()).hexdigest() == snapshot["observation_sha256"],
                model_candidate_count=len(doc.model_candidates),
                measured_count=sum(e.measurement is not None for e in entries),
                newly_measured_count=sum(e.status == "measured_for_review" for e in entries),
                unmeasurable_count=sum(e.status == "unmeasurable" for e in entries),
                review_group_count=len(trial.evidence_dossier.candidate_configuration_review),
                candidates=[dict(candidate_id=e.candidate_id, legacy_verdict=e.legacy_verdict,
                                 legacy_handoff_class=e.legacy_handoff_class, status=e.status,
                                 comparison_detection=e.comparison_detection,
                                 comparison_flux=e.comparison_aperture_flux,
                                 notes=e.uncertainty_notes) for e in entries],
                report=str(folder / "report.md"),
            )
            write_json(folder / "observation.json", trial_doc.model_dump(mode="json"))
            write_json(folder / "reasoning.json", trial.to_dict())
            (folder / "report.md").write_text(trial.final_report_markdown or "", encoding="utf-8")
            write_observation_summary_cn(trial_doc, folder)
            rows.append(row)
    checks = ["default_matches_frozen_assessment", "default_matches_frozen_audit", "trial_assessment_unchanged",
              "trial_audit_unchanged", "formal_measurements_unchanged", "input_document_unchanged", "source_file_unchanged"]
    failures = [dict(object_id=r["object_id"], cohort=r["cohort"], failed_checks=[k for k in checks if not r[k]])
                for r in rows if not all(r[k] for k in checks)]
    totals = {}
    for cohort in ("latest_unique", "historical_21_30"):
        subset = [r for r in rows if r["cohort"] == cohort]
        totals[cohort] = dict(
            count=len(subset), vote_changed=sum(r["trial_vote"] != r["baseline_vote"] for r in subset),
            fully_measured_objects=sum(r["model_candidate_count"] > 0 and r["unmeasurable_count"] == 0 for r in subset),
            objects_without_candidate_catalog=sum(r["model_candidate_count"] == 0 for r in subset),
            candidate_count=sum(r["model_candidate_count"] for r in subset),
            measured_count=sum(r["measured_count"] for r in subset),
            newly_measured_count=sum(r["newly_measured_count"] for r in subset),
            unmeasurable_count=sum(r["unmeasurable_count"] for r in subset),
        )
    result = dict(totals=totals, failures=failures, rows=rows)
    write_json(output / "comparison.json", result)
    lines = ["# 完整候选交接试验：回归对照", "",
             "冻结对象：修改前当前工作区代码（v0.19.0 未提交状态），不是把旧 v0.18 历史等级当作当前基线。",
             "40 个独立目标各取最新观测，另回放 21–30 的 10 份历史观测；两组重叠，不能相加为 50 个目标。",
             "保留/测量层试验：评分规则未改。等级不变是兼容性检查，不是证明未来放开评分不会误升，也不是科学准确率验证。", ""]
    for cohort, total in totals.items():
        lines.extend([f"## {cohort}", "", f"- 观测 {total['count']} 份；等级变化 {total['vote_changed']} 份。",
                      f"- model-chain 候选 {total['candidate_count']} 个，成功完整测量 {total['measured_count']} 个，新增待复核测量 {total['newly_measured_count']} 个，像素不可用 {total['unmeasurable_count']} 个。",
                      f"- {total['objects_without_candidate_catalog']} 份旧观测没有候选目录，不能据此验证检出完整性。", "",
                      "| 目标 | 历史v0.18等级 | 修改前当前基线 | 试验等级 | 候选/完整测量/新增待复核 | 对照检查 |", "|---|---|---|---|---|---|"])
        for row in [r for r in rows if r["cohort"] == cohort]:
            ok = all(row[k] for k in checks)
            link = Path(row["report"]).as_posix()
            lines.append(f"| [{row['object_id']}](<{link}>) | {row['historical_vote'] or '—'} | {row['baseline_vote']} | {row['trial_vote']} | {row['model_candidate_count']}/{row['measured_count']}/{row['newly_measured_count']} | {'通过' if ok else '异常'} |")
        lines.append("")
    lines.extend(["## 验收边界", "",
                  "- 本次不修改半径离散阈值、0.75 比较阈值、A/B/C/X 映射或检测器。",
                  "- 新试验把低于比较像素阈值与物理污染分开；旧隔离标签只留作对照。",
                  "- 待复核候选有像素形态、孔径信号和模型敏感性测量，也有搜索路线分组，但分组不是已证实的透镜构型。",
                  "- 后续改变评级需有非透镜负例、低SNR/弧注入回收、固定独立测试集和等级迁移检查；当前 A 样本不能衡量误报率。",
                  "- 原结果文件不覆盖；新增结果仅用于用户确认。", "", f"对照失败：{failures or '无'}"])
    (output / "comparison_cn.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(totals=totals, failures=failures), ensure_ascii=False, indent=2))
    return 1 if failures else 0


def check_e2e(output):
    """Compare fresh FITS runs from frozen pre-edit code and opt-in trial."""
    import numpy as np

    oid = "102020061_NEG607087127495633316"
    baseline_dir, trial_dir = (output / name / oid for name in ("e2e_baseline", "e2e_trial"))
    observations = [json.loads((folder / "observation.json").read_text(encoding="utf-8")) for folder in (baseline_dir, trial_dir)]
    summaries = [json.loads((folder / "summary.json").read_text(encoding="utf-8")) for folder in (baseline_dir, trial_dir)]
    left, right = observations
    fields = ("nucleus_xy", "data_quality", "foreground_galaxy", "structures", "measurements", "scene_geometry",
              "alternative_indicators", "model_candidates", "candidate_structure_audits", "geometric_paths", "candidate_systems", "foreground_model_comparison")
    checks = {f"unchanged_{field}": left[field] == right[field] for field in fields}
    checks["assessment_unchanged"] = summaries[0]["expert_assessment"] == summaries[1]["expert_assessment"]
    checks["audit_unchanged"] = summaries[0]["decision_audit"] == summaries[1]["decision_audit"]
    with np.load(left["artifacts"]["vision_arrays"], allow_pickle=False) as arrays_left, np.load(right["artifacts"]["vision_arrays"], allow_pickle=False) as arrays_right:
        checks["pixel_array_keys_unchanged"] = set(arrays_left.files) == set(arrays_right.files)
        checks["all_pixel_arrays_unchanged"] = all(
            key in arrays_right and np.array_equal(arrays_left[key], arrays_right[key], equal_nan=True)
            for key in arrays_left.files
        )
    from expert_agent.schemas.observation import ObservationDocument
    from expert_agent.vision.candidate_handoff import attach_candidate_review
    current = ObservationDocument.model_validate(right)
    replay = attach_candidate_review(current)
    checks["saved_mask_replay_matches_live_measurements"] = (
        replay.candidate_handoff_review.entries == current.candidate_handoff_review.entries
    )
    checks["live_candidates_all_measured"] = all(e.measurement is not None for e in current.candidate_handoff_review.entries)
    checks["current_reason_matches_live_assessment"] = assess(replay).expert_assessment.model_dump(mode="json") == summaries[1]["expert_assessment"]
    result = dict(object_id=oid, baseline_vote=summaries[0]["expert_assessment"]["vote_category"],
                  trial_vote=summaries[1]["expert_assessment"]["vote_category"], checks=checks,
                  passed=all(checks.values()))
    write_json(output / "e2e_comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "compare", "e2e-check"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.output.resolve())
    elif args.mode == "compare":
        raise SystemExit(compare(args.output.resolve()))
    else:
        raise SystemExit(check_e2e(args.output.resolve()))
