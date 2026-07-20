#!/usr/bin/env python3
"""Standalone script: run AnalysisAuditor Stage B (Result Auditor) on pre-computed data.

Counterfactual: QSO broad lines narrowed to ~550-700 km/s FWHM.
The auditor should flag that the key QSO broad lines (Lyα, C IV, Mg II) are
too narrow for a Type 1 QSO classification.

Usage::

    cd /home/wbc/code3/llm-spectro-agent/data/fake/fake_data/QSO_116_narrow
    python run_result_auditor.py
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# LLM connection
# ---------------------------------------------------------------------------
LLM_MODEL = "deepseek-v4-pro"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # set in .env
LLM_BASE_URL = "https://api.deepseek.com"


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
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AnalysisAuditor Stage B (Result Auditor) on QSO_116_narrow."
    )
    p.add_argument(
        "--data-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing the QSO_116_narrow upstream outputs (default: script dir).",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    # ── 0. validate config ────────────────────────────────────────────
    missing = []
    for v in ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        if not _resolve(v):
            missing.append(v)
    if missing:
        print(
            "ERROR: Missing config. Set these in the script (top of file) or as env vars:\n"
            f"  {', '.join(missing)}"
        )
        sys.exit(1)

    print(f"Data directory : {data_dir}")

    # ── 1. load upstream data ──────────────────────────────────────────
    print("Loading upstream data ...")

    spectrum = load_spectrum_npz(
        os.path.join(data_dir, "visual_interpreter", "116_spectrum.npz")
    )
    print(f"  Spectrum: {len(spectrum['wavelength'])} px")

    fa_verdict = load_json(
        os.path.join(data_dir, "feature_auditor", "verdict.json")
    )
    print(f"  FA verdicts: {len(fa_verdict.get('feature_verdicts', []))} features")

    synthesis_verdict = extract_synthesis_verdict(
        os.path.join(data_dir, "hypothesis_synthesis", "stream.md")
    )
    print(
        f"  Synthesis: z={synthesis_verdict.get('redshift')}, "
        f"best=H{synthesis_verdict.get('best_hypothesis_idx')}, "
        f"confidence={synthesis_verdict.get('confidence')}"
    )

    hypothesis_results = reconstruct_hypothesis_results(data_dir, synthesis_verdict)
    print(f"  Hypotheses: {len(hypothesis_results)} ({[h['hypothesis_idx'] for h in hypothesis_results]})")

    # ── 2. build state ─────────────────────────────────────────────────
    state = {
        "spectrum": spectrum,
        "hypothesis_analysis": synthesis_verdict,
        "feature_audit_verdict": fa_verdict,
        "harness_dir": data_dir,
        "hypothesis_results": hypothesis_results,
        "continuum": {"description": "Counterfactual: QSO broad lines narrowed to 550-700 km/s (QSO_116_narrow)."},
    }

    # ── 3. inject config into os.environ ────────────────────────────────
    for key in ("LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        val = _resolve(key)
        if val:
            os.environ[key] = val

    # ── 4. run auditor ─────────────────────────────────────────────────
    print("\nInitializing AnalysisAuditor ...")
    runtime = create_runtime()
    auditor = AnalysisAuditor(runtime)

    print("Running audit (this will take a while) ...\n")
    result_state = await auditor.run(state)

    # ── 5. output ──────────────────────────────────────────────────────
    verdict_json = result_state.get("auditor_verdict_json", {})
    verdict = result_state.get("auditor_verdict", "?")

    print(f"\n{'=' * 60}")
    print(f"Auditor verdict: {verdict}")
    print(f"Confidence     : {verdict_json.get('calibrated_confidence', '?')}")
    print(f"Quality        : {verdict_json.get('spectrum_quality', '?')}")
    print(f"Confirmed lines: {verdict_json.get('confirmed_lines', [])}")
    print(f"Line revisions : {len(verdict_json.get('line_revisions', []))}")
    print(f"Spectrum issues: {len(verdict_json.get('spectrum_issues', []))}")
    print(f"Reobserve      : {verdict_json.get('reobserve', False)}")
    print(f"{'=' * 60}")

    # Save verdict JSON
    verdict_path = os.path.join(data_dir, "result_auditor", "auditor_verdict.json")
    os.makedirs(os.path.dirname(verdict_path), exist_ok=True)
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict_json, f, indent=2, ensure_ascii=False)
    print(f"\nVerdict saved to: {verdict_path}")


if __name__ == "__main__":
    asyncio.run(main())
