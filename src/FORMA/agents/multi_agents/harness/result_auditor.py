"""
Report Writer — Final Report Writing Harness.

Runs AFTER the Result Auditor.  Takes all upstream outputs (synthesis
verdict, RA verdict, FA structured verdicts, continuum description, cleaned
line tables) and produces a structured 6-section final report + a JSON
comprehensive assessment block.
"""

import json
import os
import re
import csv as _csv
from typing import Optional, Tuple

import numpy as np
from langchain.agents import create_agent
from langchain_core.tools import tool

from FORMA.agents.common.state import SpectroState
from FORMA.agents.multi_agents.harness.tools import (
    compute_redshift_error, write_report,
)
from FORMA.core.llm import (
    _detect_vendor, _build_thinking_extra_body, _create_chat_openai,
)
from FORMA.agents.multi_agents.harness.hypothesis_synthesis import _resolve_max_tokens


# ---------------------------------------------------------------------------
# Skill loader
# ---------------------------------------------------------------------------

def _load_skill() -> str:
    skill_path = os.path.join(
        os.path.dirname(__file__), "skills", "ReportWriter", "report_writer_skill.md"
    )
    if os.path.exists(skill_path):
        return open(skill_path, encoding="utf-8").read()
    return ""


# ---------------------------------------------------------------------------
# Wavelength error lookup
# ---------------------------------------------------------------------------

def _lookup_wavelength_errors(harness_dir: str, output_dir: str) -> dict:
    """Look up wavelength_error for each unique wavelength in all line CSVs.

    Reads ``{idx}_lines.csv`` and emission/absorption CSVs, builds a mapping
    from integer-rounded wavelength to wavelength_err.  Also reads
    ``{idx}_lines_cleaned.csv`` for any additional entries.

    Returns a dict: {int(wavelength): wavelength_err}
    """
    errors: dict[int, float] = {}
    spectrum_id = os.path.basename(os.path.normpath(harness_dir)).split("_")[0]

    csv_paths = []
    import glob as _glob
    csv_paths.extend(sorted(_glob.glob(os.path.join(harness_dir, "single_hypothesis", "*_lines.csv"))))
    csv_paths.extend(sorted(_glob.glob(os.path.join(harness_dir, "single_hypothesis", "*_lines_cleaned.csv"))))
    for csv_name in [f"visual_interpreter/{spectrum_id}_emission.csv", f"visual_interpreter/{spectrum_id}_absorption.csv"]:
        p = os.path.join(output_dir, csv_name)
        if os.path.exists(p):
            csv_paths.append(p)

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                wl_str = (row.get("wavelength") or row.get("fitted_center") or "").strip()
                err_str = (row.get("wavelength_err") or row.get("wavelength_error") or row.get("fitted_center_err") or "").strip()
                if not wl_str or not err_str:
                    continue
                try:
                    wl_int = int(float(wl_str))
                    err = float(err_str)
                except (ValueError, TypeError):
                    continue
                if wl_int not in errors:
                    errors[wl_int] = err

    return errors


def _format_confirmed_lines_with_errors(
    confirmed_lines: list, wavelength_errors: dict
) -> str:
    """Format confirmed lines with their wavelength errors.

    confirmed_lines: list of [line_name, wavelength] from RA.
    wavelength_errors: {int(wavelength): error} from _lookup_wavelength_errors.
    """
    if not confirmed_lines:
        return "*(no confirmed lines)*\n"

    lines = []
    for entry in confirmed_lines:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        name, wl = entry[0], entry[1]
        wl_int = int(float(wl))
        err = wavelength_errors.get(wl_int)
        err_str = f" ± {err:.4f}" if err is not None else " (error unknown)"
        rest_wl = "?"  # LLM looks up rest wavelength from the line tables in the prompt
        lines.append(f"- {name}: λ_obs = {wl}{err_str}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_message(state: SpectroState, harness_dir: str) -> str:
    """Build the user prompt for the Report Writer report-writing LLM."""
    rule_analysis = state.get("hypothesis_analysis") or {}
    auditor_json = state.get("auditor_verdict_json") or {}
    continuum = state.get("continuum") or {}
    harness_results = state.get("hypothesis_results") or []

    # ── Spectrum metadata ──
    wl = state["spectrum"]["wavelength"]
    wl_left = float(np.min(wl))
    wl_right = float(np.max(wl))
    snr_val = state["spectrum"].get("snr")
    snr_median = float(np.median(snr_val)) if snr_val is not None else None

    output_dir = harness_dir

    parts = []

    parts.append("## Spectrum")
    parts.append("")
    parts.append(f"- Wavelength range: {wl_left:.0f} – {wl_right:.0f} Å")
    if snr_median is not None:
        parts.append(f"- Median SNR: {snr_median:.1f}")
    parts.append(f"- Blue edge: {wl_left:.0f} – 4000 Å (throughput drop)")
    parts.append(f"- Red edge (OH zone): 7800 – {wl_right:.0f} Å")
    parts.append("")

    # ── Continuum description ──
    continuum_desc = continuum.get("description", "")
    if continuum_desc:
        parts.append("## Continuum Description (VisualInterpreter)")
        parts.append("")
        parts.append(continuum_desc)
        parts.append("")

    # ── Synthesis summary ──
    parts.append("## Hypothesis Synthesis Verdict")
    parts.append("")
    best_idx = rule_analysis.get("best_hypothesis_idx")
    parts.append(f"- Best hypothesis: H{best_idx}, z = {rule_analysis.get('redshift')}")
    parts.append(f"- Classification: {rule_analysis.get('classification', '?')}")
    parts.append(f"- Confidence: {rule_analysis.get('confidence', '?')}")
    parts.append(f"- Anchor line: {rule_analysis.get('anchor_line', '?')} at {rule_analysis.get('anchor_wavelength', '?')} Å")
    parts.append(f"- Primary evidence: {rule_analysis.get('primary_evidence', '?')}")
    parts.append("")

    excluded = rule_analysis.get("excluded_hypotheses") or []
    if excluded:
        parts.append("Excluded hypotheses:")
        for exc in excluded:
            if isinstance(exc, dict):
                parts.append(f"  - H{exc.get('idx')} (z={exc.get('z')}): {exc.get('reason', '')}")
        parts.append("")

    # ── RA verdict ──
    parts.append("## Result Auditor Verdict")
    parts.append("")
    parts.append(f"- Verdict: {auditor_json.get('verdict', '?')}")
    parts.append(f"- Calibrated confidence: {auditor_json.get('calibrated_confidence', '?')}")
    parts.append(f"- Spectrum quality: {auditor_json.get('spectrum_quality', '?')}")
    parts.append(f"- has_real_peak: {auditor_json.get('has_real_peak', '?')}")
    parts.append("")

    key_issues = auditor_json.get("key_issues") or []
    if key_issues:
        parts.append("Key issues:")
        for ki in key_issues:
            parts.append(f"  - {ki}")
        parts.append("")

    # ── Confirmed lines with wavelength errors ──
    confirmed_lines = auditor_json.get("confirmed_lines") or []
    wavelength_errors = _lookup_wavelength_errors(harness_dir, output_dir)
    parts.append("## Confirmed Lines (with wavelength errors)")
    parts.append("")
    parts.append(_format_confirmed_lines_with_errors(confirmed_lines, wavelength_errors))

    # ── Per-hypothesis line tables ──
    from FORMA.agents.multi_agents.harness.hypothesis_synthesis import _build_line_tables
    audit_indices = []
    if best_idx is not None:
        audit_indices.append(best_idx)
    for exc in excluded[:2]:
        if isinstance(exc, dict) and exc.get("idx") is not None:
            if exc["idx"] not in audit_indices:
                audit_indices.append(exc["idx"])
    if audit_indices:
        audit_results = [r for r in harness_results if r.get("hypothesis_idx") in audit_indices]
        if audit_results:
            parts.append("## Per-Hypothesis Line Tables (post-FeatureAuditor)")
            parts.append("")
            parts.append(_build_line_tables(audit_results, harness_dir))

    # ── FA structured verdicts ──
    fa = state.get("feature_audit_verdict") or {}
    composite_v = [v for v in fa.get("composite_profile_verdicts", []) if v.get("hypothesis_idx") in audit_indices]
    doublet_v = [v for v in fa.get("doublet_verdicts", []) if v.get("hypothesis_idx") in audit_indices]
    oii_v = [v for v in fa.get("oii_morphology_verdicts", []) if v.get("hypothesis_idx") in audit_indices]

    if composite_v or doublet_v or oii_v:
        parts.append("## FA Structured Verdicts")
        parts.append("")
        if composite_v:
            parts.append("### Composite Profile Verdicts")
            for cv in composite_v:
                parts.append(f"- H{cv['hypothesis_idx']} {cv.get('species','?')}: composite={cv.get('is_composite')}, {cv.get('notes','')}")
            parts.append("")
        if doublet_v:
            parts.append("### Doublet Verdicts")
            for dv in doublet_v:
                parts.append(f"- H{dv['hypothesis_idx']} {dv.get('name_a','?')}+{dv.get('name_b','?')}: ratio_ok={dv.get('ratio_ok')}, {dv.get('notes','')}")
            parts.append("")
        if oii_v:
            parts.append("### [O II] Morphology Verdicts")
            for ov in oii_v:
                parts.append(f"- H{ov['hypothesis_idx']} {ov.get('wl_obs','?')}: detected={ov.get('detected')}, {ov.get('notes','')}")
            parts.append("")

    # ── Task ──
    report_path = os.path.join(harness_dir, "final_report.md")
    parts.append("## Task")
    parts.append("")
    parts.append(
        f"Write the final report following the 6-section structure in your system prompt. "
        f"Use `compute_redshift_error(rest_wavelength, wavelength_error)` "
        f"to compute σ_z for each confirmed line that has a wavelength error. "
        f"Call `write_report(file_path=\"{report_path}\", content=<full markdown>)` to save the report. "
        f"All judgments have been made upstream — your job is to **summarise and present**, not to re-analyse."
    )
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract a JSON object from LLM output text."""
    if not text:
        return None
    # Try fenced block first
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
    # Try to find a bare JSON object (DOTALL for multi-line)
    m = re.search(r'\{[^{}]*"type"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Try the whole text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    return None


async def arun(
    state: SpectroState,
    harness_dir: str,
    *,
    model: str = "deepseek-v4-pro",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_turns: int = 30,
    stream_md_path: Optional[str] = None,
) -> Tuple[str, Optional[dict]]:
    """Run the Report Writer report-writing agent.

    Returns (final_report_markdown, comprehensive_assessment_json).

    If ``stream_md_path`` is set, writes the full conversation (system prompt,
    tool calls, tool results, and final output) to that file.
    """
    system_prompt = _load_skill()
    user_prompt = _build_user_message(state, harness_dir)

    # ── Streaming setup ──
    md = None
    if stream_md_path:
        os.makedirs(os.path.dirname(stream_md_path) or ".", exist_ok=True)
        md = open(stream_md_path, "w", encoding="utf-8")
        md.write("# Report Writer — Final Report Writing\n\n")
        md.write("## System Prompt\n\n```\n")
        md.write(system_prompt)
        md.write("\n```\n\n## User Prompt\n\n```\n")
        md.write(user_prompt)
        md.write("\n```\n\n## Conversation\n\n")

    # ── Build LLM ──
    vendor = _detect_vendor(base_url)
    extra_body = (
        _build_thinking_extra_body("disabled", vendor)
        if vendor != "unknown"
        else None
    )

    llm = _create_chat_openai(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=_resolve_max_tokens(),
        extra_body=extra_body,
    )

    agent = create_agent(
        model=llm,
        tools=[compute_redshift_error, write_report],
        system_prompt=system_prompt,
    )

    config = {"recursion_limit": max_turns}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config=config,
    )

    # ── Write stream ──
    messages = result.get("messages", [])
    if md:
        _write_stream(md, messages)
        md.close()

    # ── Extract final text and JSON ──
    final_text = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str) and content.strip():
            final_text = content
            break

    parsed_json = _extract_json_from_text(final_text)
    return final_text, parsed_json


def _write_stream(md, messages: list) -> None:
    """Write conversation messages to a stream markdown file."""
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "") or ""
        tool_calls = getattr(msg, "tool_calls", None) or []

        if role == "human":
            continue  # skip the initial user message (already written)

        if role == "ai":
            if content and isinstance(content, str) and content.strip():
                md.write(f"### LLM\n\n{content}\n\n")
            for tc in tool_calls:
                name = tc.get("name", "?")
                args = tc.get("args", {})
                md.write(f"### Tool Call: {name}\n\n```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```\n\n")

        elif role == "tool":
            name = getattr(msg, "name", "?")
            if isinstance(content, str) and content.strip():
                md.write(f"### Tool Result: {name}\n\n```\n{content[:2000]}\n```\n\n")
            else:
                md.write(f"### Tool Result: {name}\n\n*(empty)*\n\n")
