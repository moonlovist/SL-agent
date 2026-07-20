#!/usr/bin/env python3
"""Batch runner: run AnalysisAuditor Stage B N times on QSO_116_narrow.

Counterfactual: QSO broad lines narrowed to ~550-700 km/s.
The auditor should catch that the key QSO lines are too narrow.

Usage::

    cd /home/wbc/code3/llm-spectro-agent/data/fake/fake_data/QSO_116_narrow
    python run_result_auditor_batch.py
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
LLM_MODEL = "deepseek-v4-pro"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # set in .env
LLM_BASE_URL = "https://api.deepseek.com"

NUM_RUNS = 100

# ---------------------------------------------------------------------------
# Make project src importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(_PROJECT_ROOT))

from AstroAgent.agents.multi_agents.HypothesisAnalyst import HypothesisAnalyst  # noqa: F401
from AstroAgent.agents.multi_agents.AnalysisAuditor import AnalysisAuditor
from AstroAgent.core.runtime.runtime_container import RuntimeContainer


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_max_tokens() -> int | None:
    env_val = os.environ.get("LLM_MAX_TOKENS", "").strip()
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass
    if "deepseek" in os.environ.get("LLM_BASE_URL", "").lower():
        return 65536
    return None


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

def load_spectrum_npz(npz_path: str) -> dict:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Spectrum NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    required = {"wavelength", "flux", "snr"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"NPZ missing required keys: {missing}  (got {list(data.keys())})")
    return {
        "wavelength": data["wavelength"].tolist(),
        "flux": data["flux"].tolist(),
        "snr": data["snr"].tolist(),
    }


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_synthesis_verdict(stream_md_path: str) -> dict:
    if not os.path.exists(stream_md_path):
        raise FileNotFoundError(f"Synthesis stream not found: {stream_md_path}")
    with open(stream_md_path, encoding="utf-8") as f:
        text = f.read()
    blocks = re.findall(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    for block in reversed(blocks):
        try:
            d = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "redshift" in d and "best_hypothesis_idx" in d:
            return d
    raise ValueError("Could not find synthesis verdict JSON in stream.md")


def reconstruct_hypothesis_results(harness_dir: str, synthesis_verdict: dict) -> list[dict]:
    rz_map: dict[int, float] = {}
    best_idx = synthesis_verdict.get("best_hypothesis_idx")
    if best_idx is not None:
        rz_map[best_idx] = synthesis_verdict.get("redshift", 0.0)
    for exc in synthesis_verdict.get("excluded_hypotheses", []):
        if isinstance(exc, dict):
            rz_map[exc.get("idx")] = exc.get("z", 0.0)
    results = []
    single_dir = os.path.join(harness_dir, "single_hypothesis")
    for idx in range(1, 20):
        csv_path = os.path.join(single_dir, f"{idx}_lines.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(single_dir, f"{idx}_lines_cleaned.csv")
            if not os.path.exists(csv_path):
                continue
        results.append({"hypothesis_idx": idx, "redshift": rz_map.get(idx, 0.0)})
    if not results:
        raise FileNotFoundError(f"No hypothesis line CSVs found under {single_dir}/")
    return results


# ---------------------------------------------------------------------------
# runtime construction
# ---------------------------------------------------------------------------

def _resolve(key: str, default: str = "") -> str:
    val = {k: v for k, v in globals().items() if k == key}.get(key, "")
    return val or os.environ.get(key, default)


class _ModelCfg:
    def __init__(self) -> None:
        max_tokens = _resolve_max_tokens()
        self.llm = {
            "model": _resolve("LLM_MODEL", "deepseek-v4-pro"),
            "api_key": _resolve("LLM_API_KEY"),
            "base_url": _resolve("LLM_BASE_URL", "https://api.deepseek.com"),
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "thinking": "disabled",
            "stream": False,
        }
        self.vlm = {
            "model": "placeholder",
            "api_key": "placeholder",
            "base_url": "https://placeholder.local",
            "temperature": 0.1,
            "max_tokens": None,
            "thinking": "none",
            "stream": False,
        }


class _Configs:
    def __init__(self) -> None:
        self.model = _ModelCfg()
        self.max_tries = int(os.environ.get("MAX_TRIES", "3"))
        self.retry_delay = int(os.environ.get("RETRY_DELAY", "180"))


def create_runtime() -> RuntimeContainer:
    return RuntimeContainer(_Configs())


# ---------------------------------------------------------------------------
# analysis helpers — narrow-line detection
# ---------------------------------------------------------------------------

# Line names to track for narrowness
_QSO_BROAD_LINE_NAMES = {
    "lyα":    {"lyα", "lya", "ly-alpha", "ly_alpha", "lyman", "lyman-α", "lyman_alpha"},
    "c_iv":   {"civ", "c_iv", "c iv", "carbon", "1549"},
    "c_iii":  {"ciii", "c_iii", "c iii]", "c_iii]", "1909"},
    "mg_ii":  {"mgii", "mg_ii", "mg ii", "magnesium", "2800"},
}

# Keywords suggesting narrow-line concern
_NARROW_KW = {"narrow", "narrower", "too narrow", "fwhm", "width",
              "not broad", "no broad", "below", "too small", "insufficient",
              "not wide enough", "only", "km/s"}

_QSO_LINE_KW = {"lyα", "lya", "lyman", "c iv", "civ", "c iii", "ciii", "c_iii",
                "mg ii", "mgii", "1549", "1909", "2800"}


def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _match_qso_line(name: str) -> str | None:
    """Return the canonical line key if name matches a QSO broad line, else None."""
    n = _normalise(name)
    for key, variants in _QSO_BROAD_LINE_NAMES.items():
        if any(v in n for v in variants):
            return key
    return None


def _is_narrow_concern(text: str) -> bool:
    """Check if text expresses concern about line width being too narrow."""
    t = text.lower()
    return any(kw in t for kw in _NARROW_KW)


def _count_line_revisions_narrow(verdict_json: dict) -> dict:
    """Count line_revisions that flag QSO broad lines as too narrow."""
    counts = {k: 0 for k in _QSO_BROAD_LINE_NAMES}
    for rev in verdict_json.get("line_revisions", []):
        line_name = rev.get("line", "") or rev.get("name", "")
        matched = _match_qso_line(line_name)
        if matched and _is_narrow_concern(str(rev)):
            counts[matched] += 1
    return counts


def _count_spectrum_issues_narrow(verdict_json: dict) -> int:
    """Count spectrum_issues that mention QSO lines + narrow/width concerns."""
    count = 0
    for issue in verdict_json.get("spectrum_issues", []):
        il = issue.lower()
        has_qso_line = any(kw in il for kw in _QSO_LINE_KW)
        has_narrow = _is_narrow_concern(il)
        if has_qso_line and has_narrow:
            count += 1
    return count


def _summarise_run(verdict_json: dict) -> dict:
    """Extract structured counts from a single run's verdict JSON."""
    narrow_lr = _count_line_revisions_narrow(verdict_json)
    narrow_si = _count_spectrum_issues_narrow(verdict_json)
    total_lr_narrow = sum(narrow_lr.values())
    return {
        "verdict": verdict_json.get("verdict", "?"),
        "confidence": verdict_json.get("calibrated_confidence", "?"),
        "n_line_revisions_narrow": narrow_lr,
        "n_line_revisions_narrow_total": total_lr_narrow,
        "n_spectrum_issues_narrow": narrow_si,
        "n_line_revisions_total": len(verdict_json.get("line_revisions", [])),
        "n_spectrum_issues_total": len(verdict_json.get("spectrum_issues", [])),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-run AnalysisAuditor Stage B on QSO_116_narrow."
    )
    p.add_argument(
        "--data-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing QSO_116_narrow upstream outputs (default: script dir).",
    )
    p.add_argument(
        "-n", "--num-runs", type=int, default=None,
        help=f"Number of runs (default: {NUM_RUNS} from script variable).",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    num_runs = args.num_runs or NUM_RUNS

    # ── 0. validate config ────────────────────────────────────────────
    missing = []
    for v in ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        if not _resolve(v):
            missing.append(v)
    if missing:
        print(f"ERROR: Missing config: {', '.join(missing)}")
        sys.exit(1)

    for key in ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        val = _resolve(key)
        if val:
            os.environ[key] = val

    # ── 1. load upstream data (once) ──────────────────────────────────
    print(f"Data directory : {data_dir}")
    print(f"Runs           : {num_runs}")
    print("Loading upstream data ...")

    spectrum = load_spectrum_npz(
        os.path.join(data_dir, "visual_interpreter", "116_spectrum.npz")
    )
    print(f"  Spectrum: {len(spectrum['wavelength'])} px")

    fa_verdict = load_json(os.path.join(data_dir, "feature_auditor", "verdict.json"))
    print(f"  FA verdicts: {len(fa_verdict.get('feature_verdicts', []))} features")

    synthesis_verdict = extract_synthesis_verdict(
        os.path.join(data_dir, "hypothesis_synthesis", "stream.md")
    )
    print(f"  Synthesis: z={synthesis_verdict.get('redshift')}, "
          f"best=H{synthesis_verdict.get('best_hypothesis_idx')}, "
          f"confidence={synthesis_verdict.get('confidence')}")

    hypothesis_results = reconstruct_hypothesis_results(data_dir, synthesis_verdict)
    print(f"  Hypotheses: {len(hypothesis_results)}")

    # ── 2. build base state ───────────────────────────────────────────
    base_state = {
        "spectrum": spectrum,
        "hypothesis_analysis": synthesis_verdict,
        "feature_audit_verdict": fa_verdict,
        "harness_dir": data_dir,
        "hypothesis_results": hypothesis_results,
        "continuum": {"description": "Counterfactual: QSO lines narrowed to 550-700 km/s (QSO_116_narrow)."},
    }

    # ── 3. setup output dirs ──────────────────────────────────────────
    batch_dir = os.path.join(data_dir, "result_auditor_batch")
    os.makedirs(batch_dir, exist_ok=True)

    default_stream = os.path.join(data_dir, "result_auditor", "stream.md")

    # ── 4. init auditor once ──────────────────────────────────────────
    print("\nInitializing AnalysisAuditor ...")
    runtime = create_runtime()
    auditor = AnalysisAuditor(runtime)

    # ── 5. batch loop ─────────────────────────────────────────────────
    results: list[dict] = []
    t_start = time.monotonic()

    for n in range(1, num_runs + 1):
        print(f"\n{'─' * 50}")
        print(f"Run {n}/{num_runs} ...")

        state = dict(base_state)

        t0 = time.monotonic()
        try:
            result_state = await auditor.run(state)
            verdict_json = result_state.get("auditor_verdict_json", {})
            elapsed = time.monotonic() - t0
        except Exception as exc:
            print(f"  ❌ LLM call failed: {exc}")
            verdict_json = {"verdict": "ERROR", "error": str(exc)}
            elapsed = time.monotonic() - t0

        # ── save artefacts ──
        stream_dst = os.path.join(batch_dir, f"stream_{n}.md")
        if os.path.exists(default_stream):
            shutil.move(default_stream, stream_dst)
        else:
            with open(stream_dst, "w") as f:
                f.write(f"(no stream produced for run {n})\n")

        verdict_dst = os.path.join(batch_dir, f"auditor_verdict_{n}.json")
        with open(verdict_dst, "w", encoding="utf-8") as f:
            json.dump(verdict_json, f, indent=2, ensure_ascii=False)

        # ── summarise ──
        s = _summarise_run(verdict_json)
        s["run"] = n
        s["elapsed_s"] = round(elapsed)
        results.append(s)

        lr_n = s["n_line_revisions_narrow_total"]
        si_n = s["n_spectrum_issues_narrow"]
        flag = f"narrow_LR={lr_n} narrow_SI={si_n}" if (lr_n or si_n) else "narrowness not flagged"
        print(f"  Verdict: {s['verdict']} | Confidence: {s['confidence']} | "
              f"LR: {s['n_line_revisions_total']} | SI: {s['n_spectrum_issues_total']} | "
              f"{flag} | {elapsed:.0f}s")

    # ── 6. final summary ──────────────────────────────────────────────
    t_total = time.monotonic() - t_start

    verdict_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    total_lr_narrow: dict[str, int] = {k: 0 for k in _QSO_BROAD_LINE_NAMES}
    total_si_narrow = 0
    n_narrow_flagged = 0

    for r in results:
        v = r["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        c = r["confidence"]
        confidence_counts[c] = confidence_counts.get(c, 0) + 1
        for line_key, count in r["n_line_revisions_narrow"].items():
            total_lr_narrow[line_key] += count
        total_si_narrow += r["n_spectrum_issues_narrow"]
        if r["n_line_revisions_narrow_total"] > 0 or r["n_spectrum_issues_narrow"] > 0:
            n_narrow_flagged += 1

    print(f"\n{'=' * 60}")
    print(f"BATCH SUMMARY — {num_runs} runs in {t_total:.0f}s ({t_total/num_runs:.0f}s/run)")
    print(f"{'=' * 60}")
    print(f"  1. Verdict:")
    for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"       {v}: {c}")
    print(f"  2. Calibrated confidence:")
    for v, c in sorted(confidence_counts.items(), key=lambda x: -x[1]):
        print(f"       {v}: {c}")
    print(f"  3. Line revisions targeting narrow QSO lines:")
    for line_key in _QSO_BROAD_LINE_NAMES:
        print(f"       {line_key}: {total_lr_narrow[line_key]}")
    print(f"       → total: {sum(total_lr_narrow.values())}")
    print(f"  4. Spectrum issues mentioning narrow QSO lines:")
    print(f"       {total_si_narrow} total across {num_runs} runs")
    print(f"  → Narrowness flagged (LR or SI): {n_narrow_flagged}/{num_runs} ({n_narrow_flagged/num_runs*100:.0f}%)")
    print(f"{'=' * 60}")

    # Save summary JSON
    summary_path = os.path.join(batch_dir, "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_runs": num_runs,
            "total_time_s": round(t_total),
            "verdict_counts": verdict_counts,
            "confidence_counts": confidence_counts,
            "total_line_revisions_narrow": total_lr_narrow,
            "total_spectrum_issues_narrow": total_si_narrow,
            "n_narrow_flagged": n_narrow_flagged,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
