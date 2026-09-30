"""Local, labelled regression diagnostics; labels are never passed to grading."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from expert_agent.core.state import ExpertState
from expert_agent.core.workflow import ExpertWorkflow
from expert_agent.schemas.observation import ObservationDocument


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["before", "after", "final"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    phase_dir = root / args.phase
    phase_dir.mkdir(parents=True, exist_ok=False)
    source_dir = phase_dir / "source_snapshot"
    shutil.copytree(ROOT / "src" / "expert_agent", source_dir / "expert_agent", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "pyproject.toml", source_dir / "pyproject.toml")
    rows = []
    sources = [("A", ROOT / "outputs_A_51_70_new_20260906"),
               ("B", ROOT / "outputs_candidate_inclusive_B_first5_20260905")]
    for expected, folder in sources:
        for path in sorted(folder.glob("*/observation.json")):
            doc = ObservationDocument.model_validate_json(path.read_text(encoding="utf-8"))
            if args.phase != "before":
                from expert_agent.vision.candidate_handoff import attach_candidate_review
                doc = attach_candidate_review(doc)
            with contextlib.redirect_stdout(io.StringIO()):
                states = [ExpertWorkflow().run(ExpertState(observation_document=doc)) for _ in range(2)]
            result = states[0].to_dict()
            same = result == states[1].to_dict()
            target = phase_dir / doc.object_id
            target.mkdir()
            (target / "observation.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            (target / "summary.json").write_text(json.dumps(states[0].final_decision, ensure_ascii=False, indent=2), encoding="utf-8")
            (target / "report.md").write_text(states[0].final_report_markdown, encoding="utf-8")
            effective = states[0].require_observation()
            (target / "reasoning_observation.json").write_text(effective.model_dump_json(indent=2), encoding="utf-8")
            if args.phase != "before":
                from expert_agent.core.reproducibility import write_manifest
                write_manifest(doc, target, mode="saved_pixel_remeasurement_and_reasoning", result=states[0])
            row = dict(object_id=doc.object_id, expected=expected,
                       vote=states[0].expert_assessment.vote_category,
                       published=states[0].final_decision.get("published_grade"),
                       audit=states[0].decision_audit.audit_passed, repeat_equal=same,
                       source_observation=str(path), input_path=doc.input_path,
                       source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (phase_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
