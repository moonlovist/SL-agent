"""
single_hypothesis.py — LLM-driven targeted line search harness.

Equips an LLM with 5 spectrum-analysis tools and a methodology skill prompt.
The LLM autonomously confirms or refutes predicted spectral lines at a given
redshift hypothesis through targeted Gaussian fitting.

Usage as module:
    from harness import run
    result = run(fits_path="/data/spectrum.fits", redshift=2.3,
                 npz_path="/data/spectrum.npz")
    print(result["report"])
    print(result["feature_catalog"])

Usage as CLI:
    python single_hypothesis.py /data/spectrum.fits 2.3 --npz /data/spectrum.npz
"""

import json
import os
import re
import time
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from langchain.agents import create_agent
from langchain_core.tools import tool

from FORMA.core.llm import _detect_vendor, _build_thinking_extra_body, _create_chat_openai
from .tools import (
    write_report, write_lines_csv, compute_redshift,
    _do_fit_peak, _do_fit_doublet,
    EMISSION_LINES, ABSORPTION_LINES, EMISSION_LINE_WIDTHS,
)

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_HTTP_RETRIES = int(os.environ.get("MAX_TRIES", 3))
_HTTP_DELAYS = [2, 5, 10, 15, 20][:_HTTP_RETRIES]

_RETRYABLE_KW = (
    "429", "rate limit", "too many requests",
    "503", "502", "500", "server error", "service unavailable",
    "timeout", "connection", "reset", "refused", "eof", "broken pipe",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _RETRYABLE_KW)


def _resolve_max_tokens() -> int | None:
    """Resolve max_tokens from env ``LLM_MAX_TOKENS``.

    When the env var is empty / unset and the provider is DeepSeek, we default
    to the maximum supported value (65 536) to minimise output truncation
    that causes "insufficient tool messages following tool_calls" errors.
    """
    env_val = os.environ.get("LLM_MAX_TOKENS", "").strip()
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass
    base_url = os.environ.get("LLM_BASE_URL", "")
    if "deepseek" in base_url.lower():
        return 65536
    return None


from .continuation import (
    _format_tool_call, _format_tool_result,
    run_continuation_streaming, run_continuation_ainvoke,
    run_continuation_invoke,
)


# ---------------------------------------------------------------------------
# Skill prompt
# ---------------------------------------------------------------------------

SKILL_DIR = Path(__file__).resolve().parent / "skills" / "HypothesisAnalyst"
_SKILL_FILES = {
    "redrock": "single_hypothesis_redrock_skill.md",
    "nomad": "single_hypothesis_skill.md",
}


def _load_skill(mode: str = "nomad") -> str:
    filename = _SKILL_FILES.get(mode, _SKILL_FILES["nomad"])
    skill_path = SKILL_DIR / filename
    with open(skill_path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Tool list
# ---------------------------------------------------------------------------

def _build_tools(mode: str = "nomad", npz_path: str = None) -> list:
    """Return tool list appropriate for the verification mode.

    nomad (classic BFM): write_report, write_lines_csv only (CWT evaluation).
    redrock: full tool belt including fitting tools.  The LLM calls fit_peak /
    fit_doublet with only physical parameters — the spectrum arrays are captured
    via closure (no npz_path needed).
    """
    if mode == "redrock":
        data = np.load(npz_path)
        _wl = data["wavelength"]
        _fl = data["flux"]

        @tool
        def _fit_peak(
            center_guess: float,
            width_3sigma: float,
            line_type: str = "emission",
            window_half: float = 200.0,
        ) -> dict:
            """Fit a single Gaussian + linear baseline around a predicted line.

            Parameters
            ----------
            center_guess : float
                λ_obs from the Predicted Lines table (Å).
            width_3sigma : float
                Broad=90, narrow=25, both=50, absorption=20.
            line_type : "emission" or "absorption".
            window_half : float
                Half-width of fitting window (default 200 Å).
            """
            return _do_fit_peak(_wl, _fl, center_guess, width_3sigma, line_type, window_half)

        @tool
        def _fit_doublet(
            center_guess_1: float,
            center_guess_2: float,
            line_type: str = "emission",
            width_3sigma: float = 25.0,
            window_half: float = 300.0,
            separation_rest: float = None,
            separation_tolerance: float = 5.0,
            amp_ratio_expected: float = None,
        ) -> dict:
            """Fit two Gaussians + linear baseline for a close line pair.

            See the user message for doublet guidelines (Ca H/K, [O III]a/b, etc.).
            """
            return _do_fit_doublet(_wl, _fl, center_guess_1, center_guess_2,
                                   line_type, width_3sigma, window_half,
                                   separation_rest, separation_tolerance, amp_ratio_expected)

        return [
            write_report, write_lines_csv,
            _fit_peak, _fit_doublet,
            compute_redshift,
        ]
    return [write_report, write_lines_csv]


# ---------------------------------------------------------------------------
# JSON extraction from LLM response
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract the final JSON block from the LLM report."""
    # Try ```json ... ``` fence
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare { ... } block at end
    m = re.search(r"\{[^{}]*\"consensus_redshift\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Feature catalog
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mask_overlap_label(
    lo: float, hi: float, obs_wl: float, masked_regions: list | None,
) -> str:
    """Return a label describing how the z-window [lo, hi] overlaps with masked regions.

    Returns empty string if no overlap. Otherwise a concise label:
      - ``[fully masked]`` — entire z-window falls in masked region(s)
      - ``[λ_pred masked]`` — nominal λ_pred is masked but part of window is clean
      - ``[window partially masked]`` — part of window overlaps mask, λ_pred is clean
    """
    if not masked_regions:
        return ""
    lo_masked = any(m_lo <= lo <= m_hi for m_lo, m_hi in masked_regions)
    hi_masked = any(m_lo <= hi <= m_hi for m_lo, m_hi in masked_regions)
    obs_masked = any(m_lo <= obs_wl <= m_hi for m_lo, m_hi in masked_regions)
    any_overlap = any(
        lo <= m_hi and hi >= m_lo for m_lo, m_hi in masked_regions
    )
    if not any_overlap:
        return ""
    if lo_masked and hi_masked:
        return " [fully masked]"
    if obs_masked:
        return " [λ_pred masked]"
    return " [window partially masked]"


def _find_nearby_features(
    rest_wl: float, obs_wl: float, z_min: float, z_max: float,
    peaks: list, troughs: list,
) -> str:
    """Find peaks/troughs whose wavelength falls within the z-verification window
    for a given rest-frame line:  [λ_rest×(1+z_min), λ_rest×(1+z_max)].

    Results are sorted by distance to ``obs_wl`` (closest first).

    Returns brief refs like ``peak@5001.2±0.3(z=0.0085)`` — full feature details
    are in the CWT Features table above the predictions section.
    """
    lo = rest_wl * (1.0 + z_min)
    hi = rest_wl * (1.0 + z_max)
    nearby = []
    for p in (peaks or []):
        pw = p.get('wavelength', 0)
        if lo <= pw <= hi:
            pw_err = p.get('wavelength_err')
            z_implied = pw / rest_wl - 1.0
            err_str = f"±{pw_err:.1f}" if isinstance(pw_err, (int, float)) else ""
            nearby.append((abs(pw - obs_wl), f"peak@{pw:.1f}{err_str}(z={z_implied:.4f})"))
    for t in (troughs or []):
        tw = t.get('wavelength', 0)
        if lo <= tw <= hi:
            tw_err = t.get('wavelength_err')
            z_implied = tw / rest_wl - 1.0
            err_str = f"±{tw_err:.1f}" if isinstance(tw_err, (int, float)) else ""
            nearby.append((abs(tw - obs_wl), f"trough@{tw:.1f}{err_str}(z={z_implied:.4f})"))
    nearby.sort(key=lambda x: x[0])
    return ", ".join(n[1] for n in nearby) if nearby else "—"


def _build_cwt_features_table(peaks: list, troughs: list) -> str:
    """Build a compact table of all CWT-detected features — printed once, above
    the predicted-lines table.  The per-line ``Nearby Features`` column only
    references wavelengths so the LLM can cross-reference this table for
    FWHM / ridge / SNR details."""
    lines = ["## CWT Detected Features\n"]
    col = "| λ (Å) | λ_err (Å) | Amp | FWHM (Å) | FWHM (km/s) | Ridge | SNR |"
    sep = "|-------|-----------|-----|----------|-------------|-------|-----|"

    def _row(f: dict) -> str:
        wl = f.get('wavelength', 0)
        wl_err = f.get('wavelength_err', None)
        amp = f.get('amplitude', 0)
        fwhm_a = f.get('FWHM_A', None)
        fwhm_k = f.get('FWHM_km_s', None)
        ridge = f.get('ridge_length', None)
        snr = f.get('snr', None)
        return (
            f"| {wl:.1f} | "
            f"{f'{wl_err:.1f}' if isinstance(wl_err, (int, float)) else '—'} | "
            f"{amp:.1f} | "
            f"{f'{fwhm_a:.1f}' if isinstance(fwhm_a, (int, float)) else '—'} | "
            f"{f'{fwhm_k:.0f}' if isinstance(fwhm_k, (int, float)) else '—'} | "
            f"{f'{ridge}' if isinstance(ridge, (int, float)) else '—'} | "
            f"{f'{snr:.1f}' if isinstance(snr, (int, float)) else '—'} |"
        )

    if peaks:
        lines.append(f"### Emission (peaks) — {len(peaks)} features\n")
        lines.append(col)
        lines.append(sep)
        for p in peaks:
            lines.append(_row(p))
        lines.append("")

    if troughs:
        lines.append(f"### Absorption (troughs) — {len(troughs)} features\n")
        lines.append(col)
        lines.append(sep)
        for t in troughs:
            lines.append(_row(t))
        lines.append("")

    return "\n".join(lines)


def _build_predictions_section(
    redshift: float,
    z_min: float,
    z_max: float,
    wl_min: float,
    wl_max: float,
    peaks: list,
    troughs: list,
    masked_regions: list | None = None,
) -> str:
    """Pre-compute predicted lines and cross-reference with detected features
    using the z-verification window for each line.

    Always shows nearby CWT features. Mask overlap annotations are appended to
    the features column when the z-window wavelength range overlaps a masked
    region: ``[fully masked]``, ``[λ_pred masked]``, or ``[window partially masked]``.
    """
    rows = []

    for name, rest_wl in EMISSION_LINES.items():
        obs_wl = rest_wl * (1.0 + redshift)
        if wl_min is not None and obs_wl < wl_min:
            continue
        if wl_max is not None and obs_wl > wl_max:
            continue
        lo = rest_wl * (1.0 + z_min)
        hi = rest_wl * (1.0 + z_max)
        features = _find_nearby_features(rest_wl, obs_wl, z_min, z_max, peaks, [])
        mask_note = _mask_overlap_label(lo, hi, obs_wl, masked_regions)
        rows.append((
            obs_wl,
            f"| {name} | {rest_wl:.1f} | {obs_wl:.1f} | em | "
            f"{EMISSION_LINE_WIDTHS.get(name, 'narrow')} | {features}{mask_note} |",
        ))

    for name, rest_wl in ABSORPTION_LINES.items():
        obs_wl = rest_wl * (1.0 + redshift)
        if wl_min is not None and obs_wl < wl_min:
            continue
        if wl_max is not None and obs_wl > wl_max:
            continue
        lo = rest_wl * (1.0 + z_min)
        hi = rest_wl * (1.0 + z_max)
        features = _find_nearby_features(rest_wl, obs_wl, z_min, z_max, [], troughs)
        mask_note = _mask_overlap_label(lo, hi, obs_wl, masked_regions)
        rows.append((
            obs_wl,
            f"| {name} | {rest_wl:.1f} | {obs_wl:.1f} | abs | "
            f"absorption | {features}{mask_note} |",
        ))

    if not rows:
        return "\n## Predicted Lines\n\n(No predicted lines fall within the observed wavelength range.)\n"

    rows.sort(key=lambda r: r[0])

    header = (
        "| Name | λ_rest (Å) | λ_obs (Å) | Type | Width | Nearby Features "
        "(ref@λ(z); cross-ref CWT table above for details) |\n"
        "|------|-----------|----------|------|-------|----------------------------------------------------------|"
    )
    return "\n## Predicted Lines at z = {:.4f}\n\n{}\n{}\n".format(
        redshift, header, "\n".join(r[1] for r in rows)
    )


def _build_user_message(
    redshift: float,
    fits_path: str,
    npz_path: str,
    *,
    mode: str = "nomad",
    wavelength_min: float = None,
    wavelength_max: float = None,
    snr_median: float = None,
    peaks: list = None,
    troughs: list = None,
    z_min: float = None,
    z_max: float = None,
    masked_regions: list = None,
    report_path: str = None,
    csv_path: str = None,
) -> str:
    _z_min = z_min if z_min is not None else round(redshift - 0.1, 4)
    _z_max = z_max if z_max is not None else round(redshift + 0.1, 4)

    _peaks = list(peaks or [])
    _troughs = list(troughs or [])

    # ── Spectrum summary (replaces load_spectrum tool call) ──
    spec_lines = [
        f"| Wavelength range | {wavelength_min:.1f} – {wavelength_max:.1f} Å |"
        if wavelength_min is not None and wavelength_max is not None
        else "| Wavelength range | (not provided) |",
    ]
    if snr_median is not None:
        spec_lines.append(f"| Median SNR | {snr_median:.1f} |")

    _masked_msg = ""
    if masked_regions:
        regions_desc = ", ".join(f"[{s:.1f}, {e:.1f}]" for s, e in masked_regions)
        spec_lines.append(f"| Masked regions | {regions_desc} |")
        _masked_msg = (
            f"\nThese regions were removed due to arm overlap or quality cuts. "
            f"The spectrum has NO data in these ranges. "
            f"In the Predicted Lines table, mask overlap is noted in [brackets]:\n"
            f"  - ``[fully masked]`` — entire z-window has no data → assign MASKED\n"
            f"  - ``[λ_pred masked]`` — nominal position masked, but CWT features may\n"
            f"    still exist in the clean part of the z-window → evaluate as usual\n"
            f"  - ``[window partially masked]`` — part of z-window overlapped → flag\n"
            f"    as caution, but evaluate available features normally\n"
        )

    spectrum_summary = (
        "## Spectrum Summary\n\n"
        + "\n".join(spec_lines)
        + "\n"
    )

    predictions_section = _build_predictions_section(
        redshift, _z_min, _z_max, wavelength_min, wavelength_max, _peaks, _troughs,
        masked_regions=masked_regions,
    )

    cwt_table = _build_cwt_features_table(_peaks, _troughs)

    # ── Mode-specific instructions ──────────────────────────────
    _mode_instructions = ""
    if mode == "redrock":
        _mode_instructions = (
            "\n## Fitting Tools Available\n\n"
            "You have additional tools beyond CWT feature evaluation:\n"
            "- **`fit_peak`**: Fit a single Gaussian + linear baseline around a predicted position. "
            "Use the λ_obs value from the Predicted Lines table as center_guess. "
            "Only call this when the Features column is ``—`` (no CWT detection) or when CWT features are "
            "all rejected. **CWT features are higher-trust than your own fitting.** Only call fit_peak "
            "as a LAST RESORT.\n"
            "- **`fit_doublet`**: Fit two Gaussians + linear baseline for close line pairs "
            "(Ca H/K, [O III]a/b, [S II]a/b, [N II]a/b, Na D). Use when both components fall "
            "in the observed range and no CWT features exist for either. The separation check "
            "provides strong independent confirmation of both redshift AND line identification.\n"
            "- **`compute_redshift`**: Compute z from a fitted center and rest wavelength.\n\n"
            "**Doublet guidelines**:\n"
            "- Ca H/K (3934.8/3969.6 Å): broad absorption, width_3sigma=90, separation_rest=34.8\n"
            "- [O III]a/b (4960.3/5008.2 Å): narrow emission, width_3sigma=25, separation_rest=47.9\n"
            "- [S II]a/b (6718.3/6732.7 Å): narrow emission, width_3sigma=25, separation_rest=14.4\n"
            "- [N II]a/b (6549.8/6585.3 Å): narrow emission, width_3sigma=25, separation_rest=35.5\n\n"
            "**Workflow**: Evaluate CWT features first → for lines with NO CWT coverage, "
            "use λ_obs from the table above as center_guess → call `fit_peak` (single) "
            "or `fit_doublet` (pairs) → write CSV → write report → JSON block.\n"
        )

    return (
        spectrum_summary
        + f"\nVerify the redshift hypothesis z ≈ {redshift}.\n\n"
        f"Verification window: z ∈ [{_z_min}, {_z_max}]\n"
        f"A CWT feature supports the hypothesis ONLY if its pre-computed implied_z falls within this window. "
        f"Each feature in the table already has its z pre-computed — no need to recalculate.\n"
        f"Lines marked [fully masked] have NO data in the entire z-window — assign status MASKED.\n"
        f"Lines with [λ_pred masked] or [window partially masked] can still be evaluated.\n"
        + _masked_msg
        + _mode_instructions
        + f"\nFITS file: {fits_path}\n"
        f"Cleaned spectrum: {npz_path}\n"
        + (f"Output Report file: {report_path}\n" if report_path else "")
        + (f"Output CSV file: {csv_path}\n" if csv_path else "")
        + "\n" + cwt_table + "\n"
        + predictions_section
        + "\nThe CWT table lists all pre-detected features with their quality metrics. "
        "The Predicted Lines table references nearby features by wavelength + implied_z — "
        "cross-reference with the CWT table for FWHM, ridge, and SNR details. "
        "Evaluate and adopt features per the skill prompt criteria. "
        "Knowledge base (lines, ionization rules, classification diagnostics) is embedded "
        "in the system prompt — no need to search external files.\n"
        + "\nBatch ALL line decisions in a single parallel turn.\n"
        + "\nPhases: evaluate all lines → write CSV → write report → JSON block."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    fits_path: str,
    redshift: float,
    npz_path: str,
    *,
    mode: str = "nomad",
    hypothesis_idx: int = 0,
    wavelength_min: float = None,
    wavelength_max: float = None,
    snr_median: float = None,
    peaks: list = None,
    troughs: list = None,
    z_min: float = None,
    z_max: float = None,
    masked_regions: list = None,
    report_path: str = None,
    csv_path: str = None,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    temperature: float = 0.1,
    max_turns: int = 500,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the LLM harness for a single redshift hypothesis.

    Parameters
    ----------
    fits_path : str
        Path to the original FITS file (for context — actual data comes from npz_path).
    redshift : float
        The central redshift hypothesis to test.
    npz_path : str
        Path to the cleaned spectrum .npz file (wavelength, flux, snr arrays).
    mode : str
        "nomad" (classic BFM) or "redrock". Controls skill, tools, and behaviour.
    z_min, z_max : float, optional
        Verification window. A fitted line is considered to support the
        hypothesis only if its implied redshift falls within [z_min, z_max].
        If not provided, defaults to redshift ± 0.005 (nomad) or ± 0.1 (redrock).
    model : str, optional
        Model name. Defaults to LLM_MODEL env var or "deepseek-v4-pro".
    api_key : str, optional
        API key. Defaults to LLM_API_KEY env var.
    base_url : str, optional
        API base URL. Defaults to LLM_BASE_URL env var.
    temperature : float
        LLM temperature (default 0.1).
    max_turns : int
        Maximum agent turns (recursion_limit). Default 30.
    verbose : bool
        If True, log intermediate messages.

    Returns
    -------
    dict with keys:
        report : str
            Full LLM text response.
        structured_output : dict or None
            Parsed JSON block from the LLM report (redshift, classification, lines, etc.).
        feature_catalog : list[dict]
            Collected tool results.
        messages : list
            Full message history (for debugging).
    """
    model = model or os.environ.get("LLM_MODEL", "deepseek-v4-pro")
    api_key = api_key or os.environ.get("LLM_API_KEY")
    base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Always disable thinking — multi-turn tool calling can't pass back reasoning_content
    _vendor = _detect_vendor(base_url)
    _extra_body = _build_thinking_extra_body("disabled", _vendor) if _vendor != "unknown" else None
    llm = _create_chat_openai(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=_resolve_max_tokens(),
        extra_body=_extra_body,
    )

    system_prompt = _load_skill(mode)

    agent = create_agent(
        model=llm,
        tools=_build_tools(mode, npz_path),
        system_prompt=system_prompt,
    )

    user_message = _build_user_message(
        redshift, fits_path, npz_path,
        mode=mode,
        wavelength_min=wavelength_min, wavelength_max=wavelength_max,
        snr_median=snr_median,
        peaks=peaks, troughs=troughs,
        z_min=z_min, z_max=z_max, masked_regions=masked_regions,
        report_path=report_path, csv_path=csv_path,
    )
    config = {"recursion_limit": max_turns}

    # ── HTTP-level retry ──────────────────────────────────────
    last_exc = None
    for attempt in range(_HTTP_RETRIES + 1):
        try:
            result = agent.invoke({"messages": [("user", user_message)]}, config=config)
            break
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt >= _HTTP_RETRIES:
                raise
            delay = _HTTP_DELAYS[min(attempt, len(_HTTP_DELAYS) - 1)]
            logging.warning(f"HTTP retry {attempt + 1}/{_HTTP_RETRIES} in {delay}s: {e}")
            time.sleep(delay)
    else:
        raise last_exc

    # ── Truncation detection (shared continuation loop) ─
    messages = result.get("messages", [])
    run_continuation_invoke(
        agent,
        messages,
        config=config,
        continuation_prompt=(
            "Your previous response was truncated by the token limit. "
            "Continue EXACTLY from where you stopped. Do NOT repeat "
            "anything already written."
        ),
        log_prefix="TargetedSearch",
    )

    feature_catalog = []

    # Concatenate ALL AI messages (original + continuation if truncated)
    report = "\n".join(
        msg.content for msg in messages
        if getattr(msg, "type", None) == "ai" and hasattr(msg, "content")
    )
    structured_output = _extract_json_block(report)

    return {
        "hypothesis_idx": hypothesis_idx,
        "redshift": redshift,
        "report": report,
        "structured_output": structured_output,
        "feature_catalog": feature_catalog,
        "messages": messages,
    }


async def arun(
    fits_path: str,
    redshift: float,
    npz_path: str,
    *,
    mode: str = "nomad",
    hypothesis_idx: int = 0,
    wavelength_min: float = None,
    wavelength_max: float = None,
    snr_median: float = None,
    peaks: list = None,
    troughs: list = None,
    z_min: float = None,
    z_max: float = None,
    masked_regions: list = None,
    report_path: str = None,
    csv_path: str = None,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    temperature: float = 0.1,
    max_turns: int = 100,
    verbose: bool = False,
    stream_md_path: str = None,
) -> Dict[str, Any]:
    """Run LLM harness for a single redshift hypothesis (async, optional streaming).

    Parameters
    ----------
    stream_md_path : str, optional
        If set, stream the full conversation (system prompt + every LLM turn +
        tool calls + tool results) to this .md file in real time. Parent
        directories are created if needed.
    """
    model = model or os.environ.get("LLM_MODEL", "deepseek-v4-pro")
    api_key = api_key or os.environ.get("LLM_API_KEY")
    base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Always disable thinking — multi-turn tool calling can't pass back reasoning_content
    _vendor = _detect_vendor(base_url)
    _extra_body = _build_thinking_extra_body("disabled", _vendor) if _vendor != "unknown" else None
    llm = _create_chat_openai(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=_resolve_max_tokens(),
        extra_body=_extra_body,
    )

    system_prompt = _load_skill(mode)
    agent = create_agent(model=llm, tools=_build_tools(mode, npz_path), system_prompt=system_prompt)

    user_message = _build_user_message(
        redshift, fits_path, npz_path,
        mode=mode,
        wavelength_min=wavelength_min, wavelength_max=wavelength_max,
        snr_median=snr_median,
        peaks=peaks, troughs=troughs,
        z_min=z_min, z_max=z_max, masked_regions=masked_regions,
        report_path=report_path, csv_path=csv_path,
    )

    # ── Streaming path ──────────────────────────────────────────
    if stream_md_path:
        os.makedirs(os.path.dirname(stream_md_path) or ".", exist_ok=True)

        config = {"recursion_limit": max_turns}

        # HTTP-level retry for the streaming call
        last_exc = None
        for attempt in range(_HTTP_RETRIES + 1):
            try:
                md = open(stream_md_path, "w", encoding="utf-8")
                md.write(f"# Hypothesis — z={redshift}\n\n")
                md.write(f"**FITS**: `{fits_path}`\n\n")
                md.write(f"**NPZ**: `{npz_path}`\n\n")
                md.write("---\n\n")
                md.write("<details>\n<summary>System Prompt</summary>\n\n")
                md.write(system_prompt)
                md.write("\n</details>\n\n---\n\n")
                md.write(f"### User\n\n{user_message}\n\n")
                if attempt > 0:
                    md.write(f"\n> ⚠ Retry attempt {attempt} after error: {last_exc}\n\n")
                md.flush()

                accumulated_messages = []
                turn = 0

                async for event in agent.astream(
                    {"messages": [("user", user_message)]},
                    config=config,
                    stream_mode="updates",
                ):
                    for _node_name, update in event.items():
                        msgs = update.get("messages", [])
                        for msg in msgs:
                            accumulated_messages.append(msg)
                            msg_type = getattr(msg, "type", None)

                            if msg_type == "ai":
                                turn += 1
                                content = msg.content if hasattr(msg, "content") else ""
                                tool_calls = getattr(msg, "tool_calls", None)
                                lines = []
                                if content:
                                    lines.append(content.strip())
                                if tool_calls:
                                    for tc in tool_calls:
                                        lines.append(_format_tool_call(tc))
                                if lines:
                                    md.write(f"### Assistant (turn {turn})\n\n")
                                    md.write("\n\n".join(lines))
                                    md.write("\n\n")
                                    md.flush()

                            elif msg_type == "tool":
                                md.write(f"### Tool Result\n\n")
                                md.write(_format_tool_result(msg))
                                md.write("\n\n")
                                md.flush()

                md.close()
                break  # success

            except Exception as e:
                last_exc = e
                try:
                    md.close()
                except Exception:
                    pass
                if not _is_retryable(e) or attempt >= _HTTP_RETRIES:
                    raise
                delay = _HTTP_DELAYS[min(attempt, len(_HTTP_DELAYS) - 1)]
                logging.warning(f"HTTP retry {attempt + 1}/{_HTTP_RETRIES} in {delay}s: {e}")
                await asyncio.sleep(delay)
        else:
            raise last_exc

        # ── Truncation detection (shared continuation loop) ─
        await run_continuation_streaming(
            agent,
            accumulated_messages,
            config=config,
            stream_md_path=stream_md_path,
            continuation_prompt=(
                "Your previous response was truncated by the token limit. "
                "Continue EXACTLY from where you stopped. Do NOT repeat "
                "anything already written. Be concise — focus on completing "
                "the remaining line evaluations and the Phase 3 report "
                "(sections 13a–13g)."
            ),
            log_prefix="TargetedSearch",
        )

        # Concatenate ALL AI messages (original + continuation if truncated)
        report = "\n".join(
            msg.content for msg in accumulated_messages
            if getattr(msg, "type", None) == "ai" and hasattr(msg, "content")
        )
        feature_catalog = []

        return {
            "hypothesis_idx": hypothesis_idx,
            "redshift": redshift,
            "report": report,
            "structured_output": _extract_json_block(report),
            "feature_catalog": feature_catalog,
            "messages": accumulated_messages,
        }

    # ── Non-streaming path ──────────────────────────────────────
    config = {"recursion_limit": max_turns}

    last_exc = None
    for attempt in range(_HTTP_RETRIES + 1):
        try:
            result = await agent.ainvoke({"messages": [("user", user_message)]}, config=config)
            break
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt >= _HTTP_RETRIES:
                raise
            delay = _HTTP_DELAYS[min(attempt, len(_HTTP_DELAYS) - 1)]
            logging.warning(f"HTTP retry {attempt + 1}/{_HTTP_RETRIES} in {delay}s: {e}")
            await asyncio.sleep(delay)
    else:
        raise last_exc

    # ── Truncation detection (shared continuation loop) ─
    messages = result.get("messages", [])
    await run_continuation_ainvoke(
        agent,
        messages,
        config=config,
        continuation_prompt=(
            "Your previous response was truncated by the token limit. "
            "Continue EXACTLY from where you stopped. Do NOT repeat "
            "anything already written."
        ),
        log_prefix="TargetedSearch",
    )

    feature_catalog = []

    # Concatenate ALL AI messages (original + continuation if truncated)
    report = "\n".join(
        msg.content for msg in messages
        if getattr(msg, "type", None) == "ai" and hasattr(msg, "content")
    )

    return {
        "hypothesis_idx": hypothesis_idx,
        "redshift": redshift,
        "report": report,
        "structured_output": _extract_json_block(report),
        "feature_catalog": feature_catalog,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="LLM-driven targeted line search harness")
    parser.add_argument("fits_path", help="Path to the original FITS file")
    parser.add_argument("redshift", type=float, help="Redshift hypothesis z")
    parser.add_argument("--npz", required=True, help="Path to cleaned spectrum .npz")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--json", action="store_true", help="Output structured JSON only")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    result = run(
        fits_path=args.fits_path,
        redshift=args.redshift,
        npz_path=args.npz,
        model=args.model,
        verbose=args.verbose,
    )

    if args.json:
        output = {
            "report": result["report"],
            "structured_output": result["structured_output"],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(result["report"])


if __name__ == "__main__":
    main()
