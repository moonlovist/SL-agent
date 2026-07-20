#!/usr/bin/env python3
"""Standalone script: run AnalysisAuditor Stage B (Result Auditor) on pre-computed data.

Reads all upstream outputs from the QSO_116 directory and invokes the
AnalysisAuditor in isolation.  Useful for counterfactual checks — you can
tweak the upstream data (CSVs, verdict JSONs, synthesis verdict) and re-run
just the auditor without re-running the full pipeline.

Usage::

    cd /home/wbc/code3/llm-spectro-agent/data/fake/fake_data/QSO_116
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
# LLM connection — fill in your values here (override env vars if set)
# ---------------------------------------------------------------------------
# LLM_MODEL = ""          # e.g. "deepseek-v4-pro"
# LLM_API_KEY = ""        # your API key
# LLM_BASE_URL = ""       # e.g. "https://api.deepseek.com"

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")  # set in .env
LLM_BASE_URL="https://api.deepseek.com"
LLM_MODEL="deepseek-v4-pro"


# ---------------------------------------------------------------------------
# Make project src importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4] / "src"
sys.path.insert(0, str(_PROJECT_ROOT))

# Pre-import modules in the same order as WorkflowOrchestrator to break a
# circular import chain in harness/__init__.py → hypothesis_synthesis →
# AnalysisAuditor.  Importing HypothesisAnalyst first forces AnalysisAuditor
# to be fully initialised before any later direct import touches it.
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
    """Load the cleaned spectrum NPZ, return {wavelength, flux, snr} as lists."""
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
    """Load a JSON file, with clear error on failure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_synthesis_verdict(stream_md_path: str) -> dict:
    """Extract the hypothesis synthesis verdict JSON from stream.md.

    The stream contains many fenced JSON blocks.  We pick the **last** block
    that contains both ``"redshift"`` and ``"best_hypothesis_idx"`` keys.
    """
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

    raise ValueError(
        "Could not find synthesis verdict JSON (with 'redshift' + "
        "'best_hypothesis_idx') in stream.md"
    )


def reconstruct_hypothesis_results(harness_dir: str, synthesis_verdict: dict) -> list[dict]:
    """Build a minimal hypothesis_results list from on-disk CSV files.

    Each item needs at least ``hypothesis_idx`` and ``redshift`` for the
    per-hypothesis line tables in the auditor prompt.
    """
    # Build redshift map from synthesis verdict
    rz_map: dict[int, float] = {}
    best_idx = synthesis_verdict.get("best_hypothesis_idx")
    if best_idx is not None:
        rz_map[best_idx] = synthesis_verdict.get("redshift", 0.0)
    for exc in synthesis_verdict.get("excluded_hypotheses", []):
        if isinstance(exc, dict):
            rz_map[exc.get("idx")] = exc.get("z", 0.0)

    results = []
    single_dir = os.path.join(harness_dir, "single_hypothesis")
    for idx in range(1, 20):  # generous upper bound
        csv_path = os.path.join(single_dir, f"{idx}_lines.csv")
        if not os.path.exists(csv_path):
            # Also check for cleaned variant
            csv_path = os.path.join(single_dir, f"{idx}_lines_cleaned.csv")
            if not os.path.exists(csv_path):
                continue
        results.append({
            "hypothesis_idx": idx,
            "redshift": rz_map.get(idx, 0.0),
        })

    if not results:
        raise FileNotFoundError(f"No hypothesis line CSVs found under {single_dir}/")

    return results


# ---------------------------------------------------------------------------
# runtime construction
# ---------------------------------------------------------------------------

def _resolve(key: str, default: str = "") -> str:
    """Use script-level variable if set, otherwise fall back to env var."""
    val = {k: v for k, v in globals().items() if k == key}.get(key, "")
    return val or os.environ.get(key, default)


class _ModelCfg:
    """Minimal config object whose __dict__ satisfies RuntimeContainer.get_model()."""

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
        # VLM is never called by AnalysisAuditor — placeholders are fine
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
    """Minimal top-level config object for RuntimeContainer."""

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
        description="Run AnalysisAuditor Stage B (Result Auditor) on pre-computed data."
    )
    p.add_argument(
        "--data-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing the QSO_116 upstream outputs (default: script dir).",
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
        "continuum": {"description": "Loaded from pre-computed test data (QSO_116)."},
    }

    # ── 3. inject config into os.environ ────────────────────────────────
    # _run_llm_agent() reads LLM_MODEL/LLM_API_KEY/LLM_BASE_URL directly
    # from os.environ, bypassing RuntimeContainer.  Push our values in.
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

    # Save verdict JSON alongside stream.md
    verdict_path = os.path.join(data_dir, "result_auditor", "auditor_verdict.json")
    os.makedirs(os.path.dirname(verdict_path), exist_ok=True)
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict_json, f, indent=2, ensure_ascii=False)
    print(f"\nVerdict saved to: {verdict_path}")


if __name__ == "__main__":
    asyncio.run(main())
