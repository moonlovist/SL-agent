"""
SelfEvolve — ground-truth comparison, failure analysis, and self-evolution.

Runs after HypothesisAnalyst (only when ``self_evolve`` is enabled).  Compares the
synthesis verdict against ground truth, runs two-round LLM root-cause analysis
on mismatches, and writes structured failure records.
"""

import os
import logging

import numpy as np

from FORMA.agents.common.state import SpectroState
from FORMA.agents.common.base_agent import BaseAgent
from FORMA.core.runtime.runtime_container import RuntimeContainer

from FORMA.agents.multi_agents.utils.HA import (
    extract_verdict_summary,
)
from FORMA.agents.multi_agents.utils.SE import (
    analyze_failure,
    _record_failure,
)


def _normalise_spectype(raw: str) -> str:
    """Map short FITS-metadata spectral types to the long forms used
    in synthesis classification."""
    mapping = {
        "QSO": "QSO",
        "ELG": "Galaxy",
        "LRG": "Galaxy",
        "BGS": "Galaxy",
        "STAR": "Unknown",
        "GALAXY": "Galaxy",
    }
    return mapping.get(raw.upper().strip(), raw)


class SelfEvolve(BaseAgent):
    """Ground-truth comparison and failure-driven self-evolution."""

    agent_name = "SelfEvolve"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    # =====================================================================
    # Main entry point
    # =====================================================================

    async def run(self, state: SpectroState) -> SpectroState:
        harness_results = state.get("hypothesis_results", [])
        if not harness_results:
            logging.info("[SelfEvolve] No harness results — skipping.")
            return state

        spec = state.get("spectrum", {})

        # ── Resolve ground truth ──────────────────────────────────
        # Priority 1: FITS METADATA
        vi_z = spec.get("VI_Z")
        vi_type = spec.get("VI_SPECTYPE")

        # Priority 2: env vars (fallback)
        expected_z_str = os.environ.get("EXPECTED_Z")
        expected_type = os.environ.get("EXPECTED_TYPE")

        if vi_z is not None:
            expected_z_str = str(vi_z)
        if vi_type is not None:
            expected_type = _normalise_spectype(vi_type)

        if expected_z_str is None or expected_type is None:
            if expected_z_str is None and expected_type is None:
                logging.info(
                    "[SelfEvolve] No ground truth available (VI_Z/VI_SPECTYPE "
                    "not in FITS METADATA and EXPECTED_Z/EXPECTED_TYPE not set) — skipping."
                )
                return state
            logging.warning(
                "[SelfEvolve] Both z and type needed for ground-truth check. "
                "Got: z=%s, type=%s",
                expected_z_str, expected_type,
            )
            return state

        try:
            expected_z = float(expected_z_str)
        except ValueError:
            logging.warning("[SelfEvolve] Invalid z value: %s", expected_z_str)
            return state

        expected_type = str(expected_type)
        tolerance = self.runtime.configs.params.z_tolerance

        # ── Compare synthesis vs ground truth ─────────────────────
        verdict = state.get("hypothesis_analysis", {})
        summary = extract_verdict_summary(verdict)

        scoring_zs = [r.get("redshift", 0) for r in harness_results
                      if not r.get("error")]
        min_dz = min((abs(z - expected_z) for z in scoring_zs), default=999)
        in_scoring = min_dz <= tolerance

        result_z = summary["redshift"]
        z_mismatch = (
            True if result_z is None
            else abs(result_z - expected_z) > tolerance
        )
        type_mismatch = summary["classification"] != expected_type

        if not z_mismatch and not type_mismatch:
            logging.info(
                "[SelfEvolve] PASSED: z=%.4f (expected %.4f, in_scoring=%s), type=%s",
                result_z if result_z else -1, expected_z, in_scoring,
                summary["classification"],
            )
            return state

        if summary["confidence"] == "LOW":
            logging.info(
                "[SelfEvolve] Mismatch but confidence=LOW — skipping "
                "(honest abstention).  Got: z=%s type=%s; Expected: z=%.4f type=%s, in_scoring=%s",
                result_z, summary["classification"], expected_z, expected_type, in_scoring,
            )
            return state

        if not in_scoring:
            logging.info(
                "[SelfEvolve] True z=%.4f NOT in scoring (min_dz=%.4f > tol=%.4f) — "
                "skipping (upstream CWT/scoring issue, not synthesis).",
                expected_z, min_dz, tolerance,
            )
            return state

        logging.warning(
            "[SelfEvolve] MISMATCH (confidence=%s, in_scoring=%s): "
            "z=%s type=%s vs expected z=%.4f type=%s",
            summary["confidence"], in_scoring, result_z, summary["classification"],
            expected_z, expected_type,
        )

        # ── Run two-round failure analysis ────────────────────────
        model_cfg = self.runtime.configs.model.llm
        harness_dir = state["output_dir"]

        ground_truth = {"z": expected_z, "type": expected_type}
        mismatch_info = {
            "z_mismatch": z_mismatch, "type_mismatch": type_mismatch,
            "in_scoring": in_scoring, "min_dz": min_dz,
        }

        spec = state.get("spectrum", {})
        spec_wl = np.asarray(spec["wavelength"])
        spec_fl = np.asarray(spec["flux"])

        analysis = await analyze_failure(
            synthesis_result=verdict,
            harness_results=harness_results,
            harness_dir=harness_dir,
            ground_truth=ground_truth,
            mismatch_info=mismatch_info,
            wl=spec_wl,
            fl=spec_fl,
            model=model_cfg["model"],
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            stream_md_path=os.path.join(harness_dir, "self_evolve/failure_analysis_stream.md"),
        )

        _record_failure(
            synthesis_result=verdict,
            harness_dir=harness_dir,
            ground_truth=ground_truth,
            mismatch_info=mismatch_info,
            analysis=analysis,
            # 根级 output_dir，用于共享的 batch failure 队列
            output_dir=os.path.dirname(state["output_dir"]),
        )
        state["_failure_recorded"] = True

        return state
