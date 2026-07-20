"""
HypothesisAnalyst — harness-driven parallel hypothesis verification.

Replaces the old QSO/ELG/LRG step-F pipeline with concurrent LLM agent runs
(one per redshift hypothesis), followed by a final LLM synthesis step.
"""

import os
import asyncio
import logging

import numpy as np

from FORMA.agents.common.state import SpectroState
from FORMA.agents.common.base_agent import BaseAgent
from FORMA.agents.common.result_writer import ResultWriter
from FORMA.core.runtime.runtime_container import RuntimeContainer

from FORMA.agents.multi_agents.harness import (
    arun as single_hypothesis_arun,
    hypothesis_synthesis_arun,
)
from FORMA.agents.multi_agents.utils.HA import (
    collect_redshift_hypotheses,
    build_dn4000_lookup,
    a_extract_harness_summaries,
)
from FORMA.agents.multi_agents.utils.plot import plot_harness_candidate
from FORMA.agents.multi_agents.AnalysisAuditor import FeatureAuditorFailed


class HypothesisAnalyst(BaseAgent):
    """Harness-based parallel redshift hypothesis verification and ranking."""

    agent_name = "HypothesisAnalyst"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)
        self._writer = ResultWriter()

    # =====================================================================
    # Phase 1: single-hypothesis search (runs BEFORE FeatureAuditor)
    # =====================================================================

    async def run(self, state: SpectroState) -> SpectroState:
        """Run the per-hypothesis single-hypothesis search batch.

        This is Phase 1 only.  The workflow will then invoke FeatureAuditor
        to verify the line catalogs, followed by ``run_hypothesis_synthesis()``
        to cross-compare the (possibly cleaned) results.
        """
        params = self.runtime.configs.params
        mode = "redrock" if params.redrock else "nomad"
        state['_hypothesis_mode'] = mode  # stash for synthesize later

        hypotheses = collect_redshift_hypotheses(
            state.get('redshift_hypotheses', {})
        )

        if not hypotheses:
            state['hypothesis_analysis'] = {
                'verdict': 'UNKNOWN',
                'reason': 'No redshift hypotheses to test.',
                'redshift': None,
                'classification': None,
                'confidence': None,
            }
            return state

        # ── Phase 1: concurrent hypothesis runs ─────────────
        hypothesis_results = await self._run_single_hypothesis_batch(
            state, hypotheses, mode=mode,
        )
        state['hypothesis_results'] = hypothesis_results

        # ── Set hypothesis_dir in state for downstream stages ─
        hypothesis_dir = state['output_dir']
        state['harness_dir'] = hypothesis_dir

        # ── Phase 1b: plot adopted features for each candidate ──
        self._plot_harness_candidates(state, hypothesis_results)

        return state

    # =====================================================================
    # Phase 2: LLM synthesis (runs AFTER FeatureAuditor)
    # =====================================================================

    async def run_hypothesis_synthesis(self, state: SpectroState) -> SpectroState:
        """Run the LLM synthesis step over (possibly cleaned) hypothesis results.

        Expects ``state['hypothesis_results']`` and ``state['_hypothesis_mode']``
        to be set by ``run()``.  If FeatureAuditor ran successfully, the
        harness directory will contain ``{idx}_lines_cleaned.csv`` files
        which the synthesize step prefers over the original catalogs.
        """
        hypothesis_results = state.get('hypothesis_results') or []
        mode = state.get('_hypothesis_mode', 'nomad')

        if not hypothesis_results:
            state['hypothesis_analysis'] = {
                'verdict': 'UNKNOWN',
                'reason': 'No harness results to synthesize.',
                'redshift': None,
                'classification': None,
                'confidence': None,
            }
            return state

        # ── FeatureAuditor failed or skipped → abort ──────────
        feature_audit_verdict = state.get("feature_audit_verdict")
        if feature_audit_verdict and feature_audit_verdict.get("skipped"):
            reason = feature_audit_verdict.get("reason", "No features to verify.")
            logging.error(
                f"[HypothesisAnalyst] FeatureAuditor failed/skipped: {reason}. "
                f"Terminating this spectrum's analysis."
            )
            # Write partial outputs before aborting
            state['hypothesis_analysis'] = {
                'verdict': 'UNKNOWN',
                'reason': f'FeatureAuditor failed: {reason}',
                'redshift': None,
                'classification': 'Unknown',
                'confidence': 'LOW',
                'best_hypothesis_idx': None,
                'primary_evidence': (
                    'FeatureAuditor could not verify spectral features. '
                    'The LLM failed to produce a valid verdict after all retries.'
                ),
                'excluded_hypotheses': [],
                'caveats': reason,
            }
            self._writer.write_hypothesis_analysis(state)
            raise FeatureAuditorFailed(
                f"FeatureAuditor failed after all retries: {reason}"
            )

        # ── LLM synthesis ───────────────────────────────────────
        await self._run_hypothesis_synthesis(state, hypothesis_results, mode=mode)

        # ── Write outputs ────────────────────────────────────────
        self._writer.write_hypothesis_analysis(state)

        return state

    # =====================================================================
    # Concurrent harness execution
    # =====================================================================

    async def _run_single_hypothesis_batch(
        self, state: SpectroState, hypotheses: list, *, mode: str = "nomad",
    ) -> list:
        """Run the first hypothesis serially to warm the KV cache (skill prompt +
        tool definitions), then fan out the rest with bounded concurrency."""
        model_cfg = self.runtime.configs.model.llm
        concurrency = self.runtime.configs.params.harness_concurrency
        sem = asyncio.Semaphore(max(1, concurrency))

        harness_dir = state['output_dir']
        os.makedirs(harness_dir, exist_ok=True)

        overlap = state['spectrum'].get('overlap_regions')

        snr = state['spectrum'].get('snr')
        snr_median = float(np.median(snr)) if snr is not None else None

        async def _run_one(idx: int, hyp: dict) -> dict:
            try:
                spec = state['spectrum']
                if mode == "redrock":
                    z_half = 0.1
                else:
                    z_half = 0.005
                result = await single_hypothesis_arun(
                    fits_path=state['file_path'],
                    redshift=hyp['z'],
                    npz_path=state['spectrum_npz_path'],
                    mode=mode,
                    hypothesis_idx=idx + 1,
                    wavelength_min=float(spec['wavelength'][0]),
                    wavelength_max=float(spec['wavelength'][-1]),
                    snr_median=snr_median,
                    peaks=state.get('peaks', []),
                    troughs=state.get('troughs', []),
                    z_min=round(hyp['z'] - z_half, 4),
                    z_max=round(hyp['z'] + z_half, 4),
                    masked_regions=overlap,
                    report_path=os.path.join(harness_dir, 'single_hypothesis', f'{idx + 1}_report.md'),
                    csv_path=os.path.join(harness_dir, 'single_hypothesis', f'{idx + 1}_lines.csv'),
                    stream_md_path=os.path.join(harness_dir, 'single_hypothesis', f'{idx + 1}_stream.md'),
                    model=model_cfg['model'],
                    api_key=model_cfg['api_key'],
                    base_url=model_cfg['base_url'],
                )
                result['hypothesis_meta'] = hyp
                return result

            except Exception as exc:
                logging.warning(
                    f"Harness hypothesis {idx} (z={hyp['z']:.4f}) failed: {exc}"
                )
                return {
                    'hypothesis_idx': idx + 1,
                    'redshift': hyp['z'],
                    'report': f'Harness execution failed: {exc}',
                    'structured_output': None,
                    'feature_catalog': [],
                    'hypothesis_meta': hyp,
                    'error': str(exc),
                }

        results = [None] * len(hypotheses)

        if len(hypotheses) >= 1:
            # ── First hypothesis: serial, warms the KV cache ──────
            results[0] = await _run_one(0, hypotheses[0])

        if len(hypotheses) >= 2:
            # ── Remaining hypotheses: concurrent (cache hits) ─────
            async def _run_with_sem(idx: int, hyp: dict) -> dict:
                async with sem:
                    return await _run_one(idx, hyp)

            tail_tasks = [
                _run_with_sem(i, h) for i, h in enumerate(hypotheses[1:], start=1)
            ]
            tail_results = await asyncio.gather(*tail_tasks)
            for i, r in enumerate(tail_results, start=1):
                results[i] = r

        return results

    # =====================================================================
    # Plot harness candidate features
    # =====================================================================

    def _plot_harness_candidates(
        self, state: SpectroState, harness_results: list,
    ) -> None:
        """为每个 harness candidate 绘制采纳的 nearby features 图像。"""
        spec = state['spectrum']
        wavelength = spec['wavelength']
        flux = spec['flux']
        continuum = state.get('continuum', {})
        continuum_flux = continuum.get('flux', spec['flux'])

        harness_dir = state['output_dir']

        for result in harness_results:
            if result is None:
                continue
            idx = result.get('hypothesis_idx', 0)
            z = result.get('redshift', 0)
            csv_path = os.path.join(harness_dir, "single_hypothesis", f"{idx}_lines.csv")
            output_path = os.path.join(harness_dir, "single_hypothesis", f"{idx}_features.png")

            try:
                plot_harness_candidate(
                    wavelength=wavelength,
                    flux=flux,
                    continuum_flux=continuum_flux,
                    lines_csv_path=csv_path,
                    output_path=output_path,
                    redshift=z,
                    title=f"Candidate {idx}",
                )
            except Exception as exc:
                logging.warning(
                    f"Failed to plot harness candidate {idx} (z={z:.4f}): {exc}"
                )

    # =====================================================================
    # LLM synthesis
    # =====================================================================

    async def _run_hypothesis_synthesis(
        self, state: SpectroState,
        hypothesis_results: list,
        *,
        mode: str = "nomad",
    ) -> None:
        """LLM reviews all harness reports and delivers a final combined
        verdict.  Delegates to harness.hypothesis_synthesis.arun."""
        model_cfg = self.runtime.configs.model.llm
        spec = state['spectrum']
        wl = np.asarray(spec['wavelength'])
        fl = np.asarray(spec['flux'])
        snr = np.asarray(spec.get('snr', []))

        harness_dir = state['output_dir']

        # ── LLM-driven structured extraction (middleware) ──
        dn4000_lookup = build_dn4000_lookup(wl, fl, hypothesis_results)
        summaries = await a_extract_harness_summaries(
            hypothesis_results,
            dn4000_lookup,
            harness_dir=harness_dir,
            model=model_cfg['model'],
            api_key=model_cfg['api_key'],
            base_url=model_cfg['base_url'],
        )

        # ── FeatureAuditor verdict (guaranteed valid — run_hypothesis_synthesis()
        #    raises FeatureAuditorFailed before reaching here if skipped) ──
        feature_audit_verdict = state.get("feature_audit_verdict")

        state['hypothesis_analysis'] = await hypothesis_synthesis_arun(
            harness_results=hypothesis_results,
            wl=wl,
            fl=fl,
            harness_dir=harness_dir,
            snr=snr,
            mode=mode,
            model=model_cfg['model'],
            api_key=model_cfg['api_key'],
            base_url=model_cfg['base_url'],
            stream_md_path=os.path.join(harness_dir, 'hypothesis_synthesis', 'stream.md'),
            report_path=os.path.join(harness_dir, 'hypothesis_synthesis', 'report.md'),
            csv_path=os.path.join(harness_dir, 'hypothesis_synthesis', 'catalog.csv'),
            summaries=summaries,
            feature_audit_verdict=feature_audit_verdict,
        )


