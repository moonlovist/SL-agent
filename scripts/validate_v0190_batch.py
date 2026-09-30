"""Reproducible full-pipeline regression against frozen v0.18 output files."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ["outputs_v0180_handoff_configuration_validation", "outputs_v0180_A_samples_10_20_full", "outputs_v0180_A_samples_21_30_full"]
TARGETS = {"102019594_NEG620978815501656388", "102019595_NEG630246532498667881", "102020056_NEG569381219495998178"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--others-only", action="store_true")
    parser.add_argument("--output", default="outputs_v0190_center_structure_regression")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    baselines = {}
    for directory in BASELINES:
        for path in (ROOT / directory).glob("*/observation.json"):
            if not (path.parent / "summary.json").exists():
                continue
            observation = read(path)
            if args.targets_only and observation["object_id"] not in TARGETS:
                continue
            if args.others_only and observation["object_id"] in TARGETS:
                continue
            baselines[observation["object_id"]] = {"observation": observation, "summary": read(path.parent / "summary.json"), "baseline_dir": str(path.parent)}
    source_digest = hashlib.sha256()
    for path in sorted((ROOT / "src/expert_agent").rglob("*.py")):
        source_digest.update(str(path.relative_to(ROOT)).encode())
        source_digest.update(path.read_bytes())
    suffix = "targets" if args.targets_only else "others" if args.others_only else "all"
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "source_sha256": source_digest.hexdigest(), "baselines": baselines}
    (out / f"baseline_{suffix}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def run(item):
        oid, baseline = item
        old = baseline["observation"]
        print(f"START {oid}", flush=True)
        with (out / f"{oid}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, "-m", "expert_agent.cli", "analyze", old["input_path"], "--output", str(out)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {"object_id": oid, "baseline_dir": baseline["baseline_dir"], "old_grade": baseline["summary"]["expert_assessment"]["vote_category"], "old_center": old["nucleus_xy"], "exit_code": process.returncode}
        dest = out / oid
        required = ["observation.json", "summary.json", "reasoning.json", "report.md", "vision_signal_audit.png", "vision_field_context.png", "vision_foreground_model_comparison.png", "vision_segmentation.fits"]
        row["missing_artifacts"] = [f for f in required if not (dest / f).is_file()]
        if process.returncode == 0 and not row["missing_artifacts"]:
            obs, summary = read(dest / "observation.json"), read(dest / "summary.json")
            assessment = summary["expert_assessment"]
            row.update(new_grade=assessment["vote_category"], new_center=obs["nucleus_xy"], center_selection=obs["foreground_galaxy"]["center_selection"], audit_passed=summary["decision_audit"]["audit_passed"], schema=obs["schema_version"], structures=len(obs["structures"]), rationale=assessment["assessment_rationale"], score_is_null=assessment["provisional_expert_score"] is None)
        print(f"DONE {oid}: {row['old_grade']} -> {row.get('new_grade', 'ERROR')} center {row.get('new_center')}", flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, item) for item in sorted(baselines.items())]
        for future in as_completed(futures):
            rows.append(future.result())
            (out / f"comparison_{suffix}.json").write_text(json.dumps(sorted(rows, key=lambda r: r["object_id"]), indent=2, ensure_ascii=False), encoding="utf-8")
    return int(any(r["exit_code"] or r["missing_artifacts"] or not r.get("audit_passed", False) for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
