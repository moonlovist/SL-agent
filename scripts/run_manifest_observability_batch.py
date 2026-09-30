"""Run a labelled manifest through the full image pipeline for held-out audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--replay-root", default=None,
        help="Reuse source_observation.json and rerun reasoning only",
    )
    args = parser.parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", payload)
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def run(item: dict) -> dict:
        object_id = item["object_id"]
        input_path = Path(item["input_path"]).expanduser().resolve()
        print(f"START {object_id} expected={item.get('expected_q1_grade')}", flush=True)
        replay_observation = (
            Path(args.replay_root).expanduser().resolve()
            / object_id / "source_observation.json"
            if args.replay_root else None
        )
        if replay_observation is not None and not replay_observation.is_file():
            raise FileNotFoundError(replay_observation)
        command = (
            [
                sys.executable, "-m", "expert_agent.cli", "reason",
                str(replay_observation), "--output", str(output / object_id),
            ]
            if replay_observation is not None
            else [
                sys.executable, "-m", "expert_agent.cli", "analyze",
                str(input_path), "--output", str(output),
            ]
        )
        with (output / f"{object_id}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
        row = {
            "object_id": object_id,
            "expected_q1_grade": item.get("expected_q1_grade"),
            "input_path": str(input_path),
            "exit_code": process.returncode,
        }
        summary_path = output / object_id / "summary.json"
        if process.returncode == 0 and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            decision = next(
                (
                    entry.get("content", {})
                    for entry in summary.get("reasoning_trace", [])
                    if entry.get("action") == "decision_conditions"
                ),
                {},
            )
            shadow = decision.get("geometry_boundary_shadow", {})
            row.update(
                grade=summary["expert_assessment"]["vote_category"],
                shadow_grade=shadow.get("shadow_grade"),
                shadow_a_eligible=shadow.get("a_eligible"),
                shadow_anchor_types=shadow.get("anchor_types", []),
                shadow_blocking_reasons=shadow.get("blocking_reasons", []),
                audit_passed=summary["decision_audit"]["audit_passed"],
            )
        else:
            row.update(
                grade="ERROR", shadow_grade=None, shadow_a_eligible=None,
                shadow_anchor_types=[], shadow_blocking_reasons=[],
                audit_passed=False,
            )
        print(
            f"DONE {object_id}: official={row['grade']} shadow={row['shadow_grade']}",
            flush=True,
        )
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run, item) for item in items]
        for future in as_completed(futures):
            rows.append(future.result())
            (output / "batch_results_partial.json").write_text(
                json.dumps(
                    sorted(rows, key=lambda row: row["object_id"]),
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
    rows.sort(key=lambda row: row["object_id"])
    (output / "batch_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return int(any(row["exit_code"] or not row["audit_passed"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
