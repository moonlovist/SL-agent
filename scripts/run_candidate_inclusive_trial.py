"""Replay saved pixel artifacts through the opt-in candidate grading policy."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expert_agent.core.state import ExpertState
from expert_agent.core.workflow import ExpertWorkflow
from expert_agent.schemas.observation import ObservationDocument
from expert_agent.vision.candidate_handoff import attach_candidate_review


def assess(document):
    with redirect_stdout(io.StringIO()):
        return ExpertWorkflow().run(ExpertState(
            image_path=document.input_path, object_id=document.object_id,
            observation_document=document,
        ))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(source.glob("*/observation.json")):
        document = ObservationDocument.model_validate_json(path.read_text(encoding="utf-8"))
        # A source root may also contain named control runs. Keep exactly the
        # canonical per-object directory and never count one target twice.
        if path.parent.name != document.object_id:
            continue
        baseline = assess(document)
        trial_document = attach_candidate_review(
            document, grading_enabled=True, output_dir=output / document.object_id,
        )
        trial = assess(trial_document)
        revised = trial.candidate_inclusive_assessment
        if revised is None:
            raise RuntimeError(f"candidate grading missing for {document.object_id}")
        folder = output / document.object_id
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "observation.json", trial_document.model_dump(mode="json"))
        write_json(folder / "summary.json", trial.final_decision)
        (folder / "report.md").write_text(trial.final_report_markdown or "", encoding="utf-8")
        roles = Counter(item.role for item in revised.grading_items)
        rows.append(dict(
            object_id=document.object_id,
            baseline=baseline.expert_assessment.vote_category,
            revised=revised.revised_vote_category,
            changed=baseline.expert_assessment.vote_category != revised.revised_vote_category,
            candidates=len(revised.grading_items),
            clear_arcs=roles["clear_arc"], companions=roles["companion"],
            excluded=roles["excluded"], configurations=len(revised.configurations),
            competitive_non_lens_explanation=revised.competitive_non_lens_explanation,
            audit_passed=trial.decision_audit.audit_passed,
            rationale=revised.rationale,
            report=str((folder / "report.md").resolve()),
        ))
    transitions = Counter(f"{row['baseline']}→{row['revised']}" for row in rows)
    result = dict(
        source=str(source), object_count=len(rows), changed_count=sum(row["changed"] for row in rows),
        transitions=dict(sorted(transitions.items())),
        all_audits_passed=all(row["audit_passed"] for row in rows), rows=rows,
    )
    write_json(output / "candidate_inclusive_comparison.json", result)
    lines = ["# 候选包容式修订评级试验", "",
             f"- 对象：{len(rows)} 个；评级变化：{result['changed_count']} 个；审计全部通过：{result['all_audits_passed']}。",
             f"- 迁移：{', '.join(f'{key} {value}' for key, value in transitions.items())}。", "",
             "基线评级与修订评级同时保留。比较模型阈值占比不作为删除条件；路线不作为独立票。该策略尚未用独立负例集校准，不是默认评级。", "",
             "| 目标 | 基线 | 修订 | clear arc | companion | 排除 | 构型 | 竞争替代 | 报告 |",
             "|---|---|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        link = Path(row["report"]).as_posix()
        lines.append(
            f"| {row['object_id']} | {row['baseline']} | **{row['revised']}** | "
            f"{row['clear_arcs']} | {row['companions']} | {row['excluded']} | "
            f"{row['configurations']} | {'有' if row['competitive_non_lens_explanation'] else '无'} | [查看](<{link}>) |"
        )
    lines.extend(["", "## 逐对象理由", ""])
    for row in rows:
        lines.append(f"- `{row['object_id']}`：{row['baseline']}→{row['revised']}。{row['rationale']}")
    (output / "CANDIDATE_INCLUSIVE_SUMMARY_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("object_count", "changed_count", "transitions", "all_audits_passed")}, ensure_ascii=False, indent=2))
    return 0 if result["all_audits_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
