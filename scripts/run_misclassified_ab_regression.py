"""Replay 20 historically misgraded Q1 A/B samples with the current agent."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs_regression_misclassified_AB_current_20260929"
ITEMS = [
    ("A", "outputs_A_51_70_new_20260906", "102021018_NEG640120110483854165"),
    ("A", "outputs_A_51_70_new_20260906", "102021493_NEG586398445479765591"),
    ("A", "outputs_A_51_70_new_20260906", "102021493_NEG587807121477852004"),
    ("A", "outputs_A_51_70_new_20260906", "102021497_NEG615658160478755691"),
    ("A", "outputs_A_51_70_new_20260906", "102021503_NEG655073934479995542"),
    ("A", "outputs_A_51_70_new_20260906", "102021986_NEG619535738476339304"),
    ("A", "outputs_A_51_70_new_20260906", "102021987_NEG632575884473969454"),
    ("A", "outputs_A_71_90_full_20260909", "102021988_NEG640289253474412348"),
    ("A", "outputs_A_71_90_full_20260909", "102021990_NEG651246796473086761"),
    ("A", "outputs_A_71_90_full_20260909", "102022474_NEG591186165469192016"),
    ("B", "outputs_B_102019126_all_band_colors_v028_20260910", "102019126_NEG589666030505609975"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020063_NEG626156743494681651"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020065_NEG639706305497277844"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020528_NEG557780503491374019"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020529_NEG564427129491602014"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020530_NEG572303674490459976"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020530_NEG572378924491097026"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020532_NEG592320863488461402"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020534_NEG604929086489066527"),
    ("B", "outputs_B_16_25_latest_v028_20260910", "102020539_NEG640534948489252606"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    OUTPUT.mkdir(exist_ok=True)
    manifest = []
    for expected, baseline_root, object_id in ITEMS:
        baseline = ROOT / baseline_root / object_id
        manifest.append({
            "object_id": object_id,
            "expected_q1_grade": expected,
            "baseline_root": baseline_root,
            "baseline_dir": str(baseline),
            "old_grade": load(baseline / "summary.json")["expert_assessment"]["vote_category"],
            "input_path": load(baseline / "observation.json")["input_path"],
        })
    (OUTPUT / "baseline_manifest.json").write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "items": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(item: dict) -> dict:
        object_id = item["object_id"]
        print(f"START {object_id}", flush=True)
        with (OUTPUT / f"{object_id}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, "-m", "expert_agent.cli", "analyze", item["input_path"], "--output", str(OUTPUT)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {**item, "exit_code": process.returncode}
        summary_path = OUTPUT / object_id / "summary.json"
        if process.returncode == 0 and summary_path.is_file():
            summary, observation = load(summary_path), load(OUTPUT / object_id / "observation.json")
            assessment = summary["expert_assessment"]
            row.update(new_grade=assessment["vote_category"], audit_passed=summary["decision_audit"]["audit_passed"], structures=len(observation.get("structures", [])), rationale=assessment["assessment_rationale"])
        else:
            row.update(new_grade="ERROR", audit_passed=False)
        print(f"DONE {object_id}: {item['old_grade']} -> {row['new_grade']}", flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, item) for item in manifest]
        for future in as_completed(futures):
            rows.append(future.result())
            (OUTPUT / "comparison_partial.json").write_text(json.dumps(sorted(rows, key=lambda row: row["object_id"]), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "comparison.json").write_text(json.dumps(sorted(rows, key=lambda row: row["object_id"]), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
