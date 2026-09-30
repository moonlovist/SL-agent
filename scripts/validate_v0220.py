"""Isolated developer regression, never consumes folder labels in inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from expert_agent.cli import _reason
from expert_agent.vision import VisionPipeline
from expert_agent.schemas.observation import ObservationDocument

PROBLEMS = ["102020061_NEG607087127495633316", "102020532_NEG589257492489565961",
            "102020062_NEG617330086494647877", "102020538_NEG637045736489835935",
            "102020528_NEG557594873487530963"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["problems", "B_first5", "replay25", "repeat3316"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()/args.group
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite regression output: {root}")
    root.mkdir(parents=True)
    data = Path("G:/Astronomy/New data")
    if args.group == "B_first5":
        paths = sorted(p for p in (data/"B").iterdir() if p.is_dir())[:5]
    elif args.group in {"problems", "repeat3316"}:
        paths = [data/"A"/s for s in (PROBLEMS if args.group == "problems" else PROBLEMS[:1])]
    else:
        paths = sorted(Path("outputs_deterministic_review_20260907/after").glob("*/observation.json"))
    results = []
    for path in paths:
        start = time.monotonic()
        identifier = path.parent.name if args.group == "replay25" else path.name
        output = root/identifier
        output.mkdir()
        print(f"START {identifier}", flush=True)
        try:
            if args.group == "replay25":
                observation = ObservationDocument.model_validate_json(path.read_text(encoding="utf-8"))
                mode = "historical_observation_grading_replay_no_new_pixel_measurements"
            else:
                observation = VisionPipeline().run(path, output)
                mode = "full_image_pipeline"
                (output/"observation.json").write_text(observation.model_dump_json(indent=2), encoding="utf-8")
            result = _reason(observation, output, mode=mode)
            record = {"object_id": identifier, "grade": result.final_decision["published_grade"],
                "vote": result.expert_assessment.vote_category, "audit": result.decision_audit.audit_passed,
                "mode": mode, "seconds": round(time.monotonic()-start, 2),
                "chains": len(observation.arc_chains),
                "multiband": observation.supplementary_diagnostics.get("multiband", {}).get("status", "not_remeasured"),
                "center": observation.nucleus_xy, "quality_flags": observation.data_quality.flags,
                "rationale": result.expert_assessment.assessment_rationale}
            if args.group == "replay25":
                summary = json.loads((path.parent/"summary.json").read_text(encoding="utf-8"))
                record["historical_grade"] = summary.get("published_grade")
            results.append(record)
            print(json.dumps(record, ensure_ascii=True), flush=True)
        except Exception as exc:
            results.append({"object_id": identifier, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(results[-1], ensure_ascii=True), flush=True)
        (root/"results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any("error" in r or not r.get("audit", False) for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
