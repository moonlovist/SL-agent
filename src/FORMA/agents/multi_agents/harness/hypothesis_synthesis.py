"""
hypothesis_synthesis.py — LLM-driven redshift hypothesis synthesis.

Equips an LLM with a spectrum-reading tool and a synthesis methodology skill
prompt.  The LLM cross-compares multiple harness reports (one per redshift
hypothesis), optionally reads raw spectrum regions, and delivers a final
combined verdict.

This is Phase 2 of the HypothesisAnalyst pipeline (Phase 1 = per-hypothesis
runs via single_hypothesis.py).
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from langchain.agents import create_agent
from langchain_core.tools import tool
from FORMA.core.llm import _detect_vendor, _build_thinking_extra_body, _create_chat_openai
from FORMA.agents.multi_agents.utils.HA import (
    prepare_diagnostic_slices,
    build_dn4000_lookup,
    extract_harness_summary,
)


def _resolve_max_tokens() -> int | None:
    """Resolve max_tokens from env ``LLM_MAX_TOKENS``.

    When the env var is empty / unset and the provider is DeepSeek, we default
    to the maximum supported value (65 536) to minimise output truncation.
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


from FORMA.agents.multi_agents.harness.tools import grep_kb, write_report, write_synthesis_csv, _detect_oii_slope_change_core, _resolve_csv_path, _build_line_tables
from FORMA.agents.multi_agents.AnalysisAuditor import build_contradiction_matrix
from FORMA.agents.multi_agents.harness.continuation import (
    _format_tool_call, _format_tool_result,
    run_continuation_streaming, run_continuation_ainvoke,
)


# ---------------------------------------------------------------------------
# Skill prompt
# ---------------------------------------------------------------------------

SKILL_PATH = Path(__file__).resolve().parent / "skills" / "HypothesisAnalyst" / "hypothesis_synthesis_skill.md"


def _load_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON extraction from LLM response
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract the final JSON verdict block from the LLM response."""
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare { ... } at end
    m = re.search(r"\{[^{}]*\"redshift\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Feature Audit Results section builder
# ---------------------------------------------------------------------------

def _build_feature_audit_summary(feature_audit_verdict: dict) -> str:
    """Build a 'Feature Audit Results' section from the FeatureAuditor verdict.

    Shows spectrum quality, KEEP/REMOVE/FLAG counts, REMOVED list with
    reasons, FLAGGED list with concerns, and global issues.
    """
    if not feature_audit_verdict or not feature_audit_verdict.get("feature_verdicts"):
        return ""

    quality = feature_audit_verdict.get("spectrum_quality", "unknown")
    justification = feature_audit_verdict.get("spectrum_quality_justification", "")
    global_issues = feature_audit_verdict.get("global_issues", [])
    verdicts = feature_audit_verdict["feature_verdicts"]

    keep = []
    removed = []
    flagged = []
    for v in verdicts:
        rec = _clean_rec(v.get("recommendation"))
        if rec.startswith("REMOVE"):
            removed.append(v)
        elif rec.startswith("FLAG"):
            flagged.append(v)
        else:
            keep.append(v)

    lines = ["## Feature Audit Results", ""]
    lines.append(f"**Spectrum quality: {quality}** — {justification}")
    lines.append("")
    lines.append("| | Count |")
    lines.append("|---|-------|")
    lines.append(f"| KEEP | {len(keep)} |")
    lines.append(f"| FLAG | {len(flagged)} |")
    lines.append(f"| REMOVE | {len(removed)} |")
    lines.append("")

    if flagged:
        lines.append("### FLAGGED Features (real but caveated)")
        lines.append("")
        lines.append("| λ_obs (Å) | Confidence | Issues |")
        lines.append("|-----------|------------|--------|")
        for v in flagged:
            conf = v.get("confidence", "—")
            issues = "; ".join(v.get("issues", [])) or "—"
            wl = v.get("wl_obs", "?")
            wl_str = f"{float(wl):.1f}" if isinstance(wl, (int, float)) else str(wl)
            lines.append(f"| {wl_str} | {conf} | {issues} |")
        lines.append("")

    if global_issues:
        lines.append("### Global Issues")
        lines.append("")
        for issue in global_issues:
            lines.append(f"- {issue}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Contradiction matrix filter (post-FeatureAuditor)
# ---------------------------------------------------------------------------

def _filter_contradiction_matrix(
    matrix_rows: list,
    doublet_annotations: list,
    stats: dict,
    feature_audit_verdict: dict,
) -> tuple:
    """Filter the contradiction matrix and doublet annotations.

    Keeps only KEEP + FLAG features.  Augments surviving rows with
    ``confidence``, ``issues``, and ``is_flagged`` from the verdict.
    Recomputes statistics for the filtered set.
    """
    verdicts = feature_audit_verdict.get("feature_verdicts", []) if feature_audit_verdict else []
    if not verdicts:
        return matrix_rows, doublet_annotations, stats

    # Build verdict lookup keyed by (int_wl, amp_sign) from verdict's feature_type
    verdict_lookup = {}
    for v in verdicts:
        wl = v.get("wl_obs")
        if wl is None:
            continue
        ft = (v.get("feature_type") or "").lower()
        amp_sign = -1 if ft.startswith("abs") else 1
        verdict_lookup[(int(float(wl)), amp_sign)] = v

    def _is_survivor_by_key(key: tuple) -> bool:
        v = verdict_lookup.get(key)
        if v is None:
            return True  # not in verdict — keep
        rec = _clean_rec(v.get("recommendation"))
        return rec.startswith("KEEP") or rec.startswith("FLAG")

    def _get_verdict(key: tuple) -> dict:
        return verdict_lookup.get(key) or {}

    # Filter matrix rows
    # Rebuild group_key from row_type (matching FA's feature_type-based
    # convention) rather than relying on pre-computed amplitude-sign keys
    # which misclassify zero-amplitude emission features (amp=0 → >0 is False).
    filtered_rows = []
    n_removed = 0
    for row in matrix_rows:
        rt = row.get("row_type", "")
        if rt == "emission":
            amp_sign = 1
        elif rt == "absorption":
            amp_sign = -1
        else:
            amp_sign = 1 if row.get("row_amp", 0) >= 0 else -1
        gk = (int(row["wl_obs"]), amp_sign)
        if _is_survivor_by_key(gk):
            verdict = _get_verdict(gk)
            row["confidence"] = verdict.get("confidence", "—")
            row["issues"] = verdict.get("issues", [])
            row["is_flagged"] = (
                _clean_rec(verdict.get("recommendation")).startswith("FLAG")
            )
            filtered_rows.append(row)
        else:
            n_removed += 1

    # Annotate doublet annotations with per-component FA status.
    # Always keep every doublet — even if a component was REMOVED, the pair
    # carries context that the Synthesis agent needs (e.g. an orphan whose
    # claimed feature is noise tells the agent the doublet is spurious).
    def _comp_status(key: tuple) -> str:
        v = verdict_lookup.get(key)
        if v is None:
            return "not_reviewed"
        rec = _clean_rec(v.get("recommendation"))
        if rec.startswith("REMOVE"):
            return "removed"
        if rec.startswith("KEEP") or rec.startswith("FLAG"):
            return "surviving"
        return "not_reviewed"

    filtered_doublets = []
    for da in doublet_annotations:
        if da.get("complete"):
            key_a = (int(da["wl_a"]), 1 if da.get("amp_a", 0) >= 0 else -1)
            key_b = (int(da["wl_b"]), 1 if da.get("amp_b", 0) >= 0 else -1)
            da["comp_a_status"] = _comp_status(key_a)
            da["comp_b_status"] = _comp_status(key_b)
            filtered_doublets.append(da)
        else:
            key = (int(da["wl_claimed"]), 1 if da.get("amp_claimed", 0) >= 0 else -1)
            da["comp_claimed_status"] = _comp_status(key)
            filtered_doublets.append(da)

    # Recompute stats
    n_total = sum(row["n_hypotheses"] for row in filtered_rows)
    amplitudes = [
        abs(row["row_amp"]) for row in filtered_rows
        if abs(row.get("row_amp", 0)) > 0
    ]
    median_amp = (
        round(float(np.median(amplitudes)), 4) if amplitudes else 0.0
    )
    top_q = (
        round(float(sorted(amplitudes, reverse=True)[max(0, len(amplitudes) // 4)]), 4)
        if amplitudes else 0.0
    )

    filtered_stats = {
        **stats,
        "n_rows": len(filtered_rows),
        "n_total_features": n_total,
        "n_edge_blue": sum(1 for r in filtered_rows if r["is_edge_blue"]),
        "n_edge_red": sum(1 for r in filtered_rows if r["is_edge_red"]),
        "median_amplitude": median_amp,
        "top_quartile_amplitude": top_q,
        "n_removed": n_removed,
    }

    return filtered_rows, filtered_doublets, filtered_stats


# ---------------------------------------------------------------------------
# Verified matrix section builder
# ---------------------------------------------------------------------------

def _build_verified_matrix_section(
    matrix_rows: list,
    stats: dict,
    harness_results: list,
) -> str:
    """Build a 'Verified Feature Contradiction Matrix' section.

    Uses the same ``|λ_obs|Type|Amp|Width|H1|H2|...|`` format as the
    FeatureAuditor matrix, with an added Confidence column.
    """
    if not matrix_rows:
        return (
            "\n## Verified Feature Contradiction Matrix\n\n"
            "*All features were removed by FeatureAuditor. "
            "No surviving features for cross-comparison.*\n"
        )

    hyp_indices = stats.get("hypothesis_indices", [])
    hyp_info = {}
    for r in harness_results:
        idx = r.get("hypothesis_idx")
        if idx is not None:
            hyp_info[idx] = {"redshift": r.get("redshift", 0)}

    lines = ["## Verified Feature Contradiction Matrix", ""]
    lines.append(
        "Each row is a FeatureAuditor-verified feature (KEEP or FLAG). "
        "REMOVED features (noise/artifacts) have been excluded. "
        "Confidence is from FeatureAuditor's visual spectrum verification. "
        "⚠ = FLAGGED (real but caveated — treat as weakened evidence)."
    )
    lines.append("")

    n_removed = stats.get("n_removed", 0)
    lines.append(
        f"**{stats['n_rows']} surviving features** across "
        f"{len(hyp_indices)} hypotheses"
        + (f" ({n_removed} removed as noise/artifact)." if n_removed else ".")
    )
    lines.append("")

    # Header
    col_header = "| λ_obs (Å) | Type | Amp | Width | Conf |"
    col_sep = "|-----------|------|------|--------|------|"
    for hi in hyp_indices:
        z = hyp_info.get(hi, {}).get("redshift", "?")
        col_header += f" H{hi} (z={z}) |"
        col_sep += "-------------|"
    lines.append(col_header)
    lines.append(col_sep)

    for row in matrix_rows:
        wl_obs = row["wl_obs"]
        edge_prefix = ""
        if row["is_edge_blue"]:
            edge_prefix = "🔵 "
        elif row["is_edge_red"]:
            edge_prefix = "🔴 "

        conf = row.get("confidence", "—")
        if row.get("is_flagged"):
            conf += " ⚠"

        vals = [
            f"{edge_prefix}{wl_obs:.1f}",
            row["row_type"],
            f"{row['row_amp']:+.3f}" if row.get("row_amp") is not None else "—",
            row.get("row_width", "—"),
            conf,
        ]
        for hi in hyp_indices:
            cell = row["cells"].get(hi)
            if cell is None:
                vals.append("—")
            else:
                status_mark = " (MARG)" if cell["status"] == "MARGINAL" else ""
                vals.append(f"{cell['name']}{status_mark}")
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Surviving doublets section builder
# ---------------------------------------------------------------------------

def _build_surviving_doublets_section(
    doublet_annotations: list,
    feature_audit_verdict: dict | None = None,
) -> str:
    """Build a 'Surviving Doublets' section.

    Only shows doublets where all relevant components survived FeatureAuditor
    verification.  Merges pre-computed separation/ratio data with
    FeatureAuditor's visual verification from ``doublet_verdicts``.
    """
    if not doublet_annotations:
        return ""

    complete = [da for da in doublet_annotations if da.get("complete")]
    orphans = [da for da in doublet_annotations if not da.get("complete")]

    if not complete and not orphans:
        return ""

    # ── Build lookup from FeatureAuditor doublet verdicts ──
    dv_lookup = {}  # key: (hypothesis_idx, name_a, name_b) or (hypothesis_idx, claimed)
    if feature_audit_verdict:
        for dv in feature_audit_verdict.get("doublet_verdicts", []) or []:
            hi = dv.get("hypothesis_idx")
            na = (dv.get("name_a") or "").strip()
            nb = (dv.get("name_b") or "").strip()
            if nb:
                dv_lookup[(hi, na, nb)] = dv
            else:
                dv_lookup[(hi, na)] = dv  # orphan keyed by claimed name

    lines = ["## Surviving Doublets", ""]
    lines.append(
        "Doublet pairs and orphans from the FeatureAuditor contradiction matrix. "
        "Components removed by FA (noise/artifact) are marked with ~~strikethrough~~. "
        "FeatureAuditor's visual verification of doublet ratio is included where available."
    )
    lines.append("")

    if complete:
        lines.append("### Complete Pairs")
        lines.append("")
        for da in complete:
            hi = da["hypothesis_idx"]
            na = da["name_a"]
            nb = da["name_b"]
            dv = dv_lookup.get((hi, na, nb)) or {}

            ratio_ok = dv.get("ratio_ok")
            notes = dv.get("notes", "")

            ratio_status = ""
            if ratio_ok is True:
                ratio_status = " ✓ ratio"
            elif ratio_ok is False:
                ratio_status = " ✗ ratio anomaly"

            fa_note = f" — {notes}" if notes else ""

            # Format components: strikethrough if removed by FA
            a_status = da.get("comp_a_status", "not_reviewed")
            b_status = da.get("comp_b_status", "not_reviewed")
            a_disp = f"~~{na}@{da['wl_a']:.1f}~~" if a_status == "removed" else f"{na}@{da['wl_a']:.1f}"
            b_disp = f"~~{nb}@{da['wl_b']:.1f}~~" if b_status == "removed" else f"{nb}@{da['wl_b']:.1f}"

            # Build removal note
            removal_notes = []
            if a_status == "removed":
                removal_notes.append(f"{na} removed by FA (noise)")
            if b_status == "removed":
                removal_notes.append(f"{nb} removed by FA (noise)")
            removal_str = f" — {', '.join(removal_notes)}" if removal_notes else ""

            lines.append(
                f"- **H{hi}**: {a_disp} + {b_disp} → "
                f"ratio {da['note']}"
                f"{ratio_status}{removal_str}{fa_note}"
            )
        lines.append("")

    if orphans:
        lines.append("### Orphans (only one component claimed)")
        lines.append("")
        for da in orphans:
            hi = da["hypothesis_idx"]
            claimed = da["claimed"]
            dv = dv_lookup.get((hi, claimed)) or {}

            fa_note = dv.get("notes", "")
            note_str = f" — {fa_note}" if fa_note else ""

            # Format claimed component: strikethrough if removed by FA
            c_status = da.get("comp_claimed_status", "not_reviewed")
            if c_status == "removed":
                claimed_disp = f"~~{claimed}@"
                removal_note = " — claimed component removed by FA (noise)"
            else:
                claimed_disp = f"{claimed}@"
                removal_note = ""

            lines.append(
                f"- **H{hi}**: {claimed_disp}"
                f"{da['wl_claimed']:.1f} (amp={da['amp_claimed']:+.3f}) → "
                f"**missing {da['missing']}** at λ ≈ {da['wl_missing']:.1f} Å"
                f"{removal_note}{note_str}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FA morphology result forwarders (from FeatureAuditor to Synthesis)
# ---------------------------------------------------------------------------

def _build_oii_morphology_section(feature_audit_verdict: dict) -> str:
    """Build an 'O II Morphology Results (from FeatureAuditor)' section.

    Forwards FA's slope-change test results so Synthesis can reference them
    instead of re-calling ``detect_oii_slope_change``.
    """
    verdicts = (feature_audit_verdict or {}).get("oii_morphology_verdicts") or []
    if not verdicts:
        return ""

    lines = ["## O II Morphology Results (from FeatureAuditor)", ""]
    lines.append(
        "FeatureAuditor has already performed the `detect_oii_slope_change` test "
        "and assessed peak prominence for every [O II] claim. Use these pre-computed "
        "verdicts directly — do NOT re-call the tool."
    )
    lines.append("")
    lines.append("| H | λ_obs (Å) | Detected | Signature | dip_ratio | FWHM (km/s) | Prominence | Notes |")
    lines.append("|---|-----------|----------|-----------|-----------|-------------|------------|-------|")
    for v in verdicts:
        hi = v.get("hypothesis_idx", "?")
        wl = v.get("wl_obs", "?")
        wl_str = f"{float(wl):.1f}" if isinstance(wl, (int, float)) else str(wl)
        detected = "✓" if v.get("detected") else "✗"
        sig = v.get("signature_type") or "—"
        dip = f"{v['dip_ratio']:.3f}" if v.get("dip_ratio") is not None else "—"
        fwhm = f"{v['fwhm_km_s']:.0f}" if v.get("fwhm_km_s") is not None else "—"
        prom = v.get("peak_prominence", "—")
        notes = v.get("notes", "—")
        lines.append(f"| H{hi} | {wl_str} | {detected} | {sig} | {dip} | {fwhm} | {prom} | {notes} |")
    lines.append("")
    return "\n".join(lines)


def _build_lyalpha_forest_section(feature_audit_verdict: dict) -> str:
    """Build a 'Lyα Forest Results (from FeatureAuditor)' section.

    Forwards FA's Lyα forest assessment so Synthesis does not need to re-read
    ±300 Å around every Lyα claim.
    """
    verdicts = (feature_audit_verdict or {}).get("lyalpha_forest_verdicts") or []
    if not verdicts:
        return ""

    lines = ["## Lyα Forest Results (from FeatureAuditor)", ""]
    lines.append(
        "FeatureAuditor has already read the spectrum ±300 Å around every Lyα "
        "claim and assessed forest visibility. Use these pre-computed verdicts."
    )
    lines.append("")
    lines.append("| H | λ_obs (Å) | Forest Visible | Observable | DLA | Notes |")
    lines.append("|---|-----------|----------------|------------|-----|-------|")
    for v in verdicts:
        hi = v.get("hypothesis_idx", "?")
        wl = v.get("wl_obs", "?")
        wl_str = f"{float(wl):.1f}" if isinstance(wl, (int, float)) else str(wl)
        vis = "✓" if v.get("forest_visible") else "✗"
        obs = "✓" if v.get("forest_observable") else "✗"
        dla = "✓" if v.get("dla_detected") else "✗"
        notes = v.get("notes", "—")
        lines.append(f"| H{hi} | {wl_str} | {vis} | {obs} | {dla} | {notes} |")
    lines.append("")
    return "\n".join(lines)


def _build_composite_profile_section(feature_audit_verdict: dict) -> str:
    """Build a 'Composite Profile Results (from FeatureAuditor)' section.

    Forwards FA's emission–absorption composite profile verdicts.
    """
    verdicts = (feature_audit_verdict or {}).get("composite_profile_verdicts") or []
    if not verdicts:
        return ""

    lines = ["## Composite Profile Results (from FeatureAuditor)", ""]
    lines.append(
        "FeatureAuditor has already assessed whether co-located emission and "
        "absorption claims of the same species form a genuine composite profile. "
        "Use these pre-computed verdicts."
    )
    lines.append("")
    lines.append("| H | Species | λ_em (Å) | λ_abs (Å) | Composite | Morphology | Notes |")
    lines.append("|---|---------|----------|-----------|-----------|------------|-------|")
    for v in verdicts:
        hi = v.get("hypothesis_idx", "?")
        sp = v.get("species", "?")
        wl_em = v.get("wl_em", "?")
        wl_em_str = f"{float(wl_em):.1f}" if isinstance(wl_em, (int, float)) else str(wl_em)
        wl_abs = v.get("wl_abs", "?")
        wl_abs_str = f"{float(wl_abs):.1f}" if isinstance(wl_abs, (int, float)) else str(wl_abs)
        comp = "✓" if v.get("is_composite") else "✗"
        morph = v.get("morphology", "—")
        notes = v.get("notes", "—")
        lines.append(f"| H{hi} | {sp} | {wl_em_str} | {wl_abs_str} | {comp} | {morph} | {notes} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task instruction builders
# ---------------------------------------------------------------------------

def _build_verified_task(surviving: bool = True) -> str:
    """Task instructions for the post-FeatureAuditor synthesis path.

    Features have already been verified — no blind review needed.
    The verified matrix IS the Phase 2 read list.
    """
    if not surviving:
        return (
            "## Task\n\n"
            "FeatureAuditor has verified all claimed features and found NONE "
            "to be physically real. All detected features were classified "
            "as noise or artifacts. This is a strong signal that the spectrum "
            "is noise-dominated or that none of the proposed redshift "
            "hypotheses match real spectral features.\n\n"
            "**Deliver your verdict now.** Write the CSV and report (Phase 3 "
            "of your system prompt), then output `redshift=null`, "
            "`classification=\"Unknown\"`, `confidence=\"LOW\"` as the JSON "
            "verdict block (Phase 4 of your system prompt).\n\n"
            "Do NOT attempt Phase 2 reads — there are no surviving features "
            "to discriminate between."
        )

    return (
        "## Task\n\n"
        "FeatureAuditor has already independently verified every feature "
        "by reading the raw spectrum. Your job is now **cross-comparison**, "
        "not re-verification.\n\n"
        "### Phase 1: Review Verified Features\n\n"
        "1. **Orient yourself**: Read the Feature Audit Results above. Note "
        "spectrum quality, which features were removed/flagged, and any global "
        "issues (edge zone quality, OH/OI contamination).\n"
        "2. **Scan the Verified Feature Contradiction Matrix**: Identify rows "
        "where multiple hypotheses claim different line IDs at the same observed "
        "wavelength — these are your discriminating windows.\n"
        "3. **Check internal consistency per hypothesis** using only verified "
        "(KEEP + FLAG) features: redshift scatter, ionization consistency, "
        "surviving doublet ratios. Use `grep_kb` for physics rules.\n"
        "4. **Note FLAGGED features** (⚠): These are real but caveated. "
        "A hypothesis whose key anchor line is FLAGGED should be treated with "
        "extra caution.\n\n"
        "### Decision after Phase 1\n\n"
        "- If one hypothesis uniquely explains all verified features with tight "
        "physical consistency → deliver verdict directly (skip Phase 2).\n"
        "- If multiple hypotheses can explain the verified features → proceed to "
        "Phase 2 with targeted reads at the discriminating windows in the "
        "verified matrix.\n"
        "- If no hypothesis is credible → deliver UNKNOWN with LOW confidence.\n\n"
        "### Phase 2: Targeted Spectrum Investigation\n\n"
        "The Verified Feature Contradiction Matrix IS your Phase 2 read list. "
        "For each row where hypotheses disagree, use `read_spectrum_region` to "
        "examine the specific wavelength window and determine which line "
        "identification (if any) is physically correct.\n\n"
        "**Read as little data as possible** — target each read at a specific "
        "discriminating question. Batch your reads.\n\n"
        "### Phases 3 & 4\n\n"
        "First write the synthesis CSV (`write_synthesis_csv`) and report "
        "(`write_report`) per Phase 3 of your system prompt, then output the "
        "final JSON verdict block per Phase 4. REMOVED features must NOT "
        "appear in the CSV output."
    )


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _build_user_message(
    harness_results: list,
    harness_dir: str,
    wl: np.ndarray,
    fl: np.ndarray,
    feature_audit_verdict: dict,
    snr: np.ndarray | None = None,
    summaries: list[str] | None = None,
    mode: str = "nomad",
    report_path: str | None = None,
    csv_path: str | None = None,
) -> str:
    """Build the user prompt with spectrum metadata, harness reports, and
    pre-computed Dn4000 diagnostics.

    Parameters
    ----------
    summaries : list[str] or None
        Pre-built markdown summaries (one per hypothesis). If None, they
        are built on the fly via :func:`extract_harness_summary`.
    mode : str
        "nomad" or "redrock". Redrock-mode reports may contain fit-derived
        measurements alongside CWT features.
    feature_audit_verdict : dict or None
        FeatureAuditor verdict JSON.  When present, a Feature Audit Results
        section and a Verified Feature Contradiction Matrix replace the
        adopted catalog, and task instructions skip blind review.
    """

    # ── Build Dn4000 lookup ──
    dn4000_lookup = build_dn4000_lookup(wl, fl, harness_results)

    # ── Collect report sections ──
    report_sections = []
    for i, r in enumerate(harness_results):
        idx = r["hypothesis_idx"]
        report_text = r.get("report", "")
        if not report_text:
            _report_path = os.path.join(harness_dir, f"single_hypothesis/{idx}_report.md")
            if os.path.exists(_report_path):
                report_text = Path(_report_path).read_text(encoding="utf-8")

        # Use pre-built summary if available, otherwise fall back
        if summaries and i < len(summaries):
            report_sections.append(summaries[i])
        else:
            r_with_report = {**r, "report": report_text}
            report_sections.append(
                extract_harness_summary(r_with_report, dn4000_lookup)
            )

        # Full report in collapsible section for Phase 2 deep-dive
        if report_text and not r.get("error"):
            report_sections.append(
                "<details>\n<summary>Full report</summary>\n\n"
                + report_text
                + "\n</details>\n\n"
            )

    # ── Spectrum metadata ──
    spec_wl_range = f"{float(wl[0]):.0f} – {float(wl[-1]):.0f}"
    _snr = np.asarray(snr) if snr is not None else np.array([])
    snr_median = float(np.median(_snr)) if len(_snr) > 0 else None

    # ── Dn4000 diagnostics ──
    diagnostic_slices = prepare_diagnostic_slices(wl, fl, harness_results)

    # ── Mode-specific note ──
    _mode_note = ""
    if mode == "redrock":
        _mode_note = (
            "\n**Note**: These harness reports come from redrock-mode runs. "
            "Some line measurements may be fit-derived (fit_peak, fit_doublet) "
            "rather than CWT-adopted. Fit-derived lines have no ridge_length "
            "or cwt_snr but provide delta_chi2_per_n and local_snr quality "
            "metrics. Treat local_snr > 10 as roughly equivalent to "
            "cwt_snr > 10 + ridge_length >= 5. For doublet fits, the "
            "amplitude ratio provides the strongest validation — a correct "
            "ratio confirms that both components are physically associated "
            "rather than chance alignments.\n"
        )

    _output_paths = ""
    if report_path:
        _output_paths += f"Output Synthesis Report: {report_path}\n"
    if csv_path:
        _output_paths += f"Output Synthesis CSV: {csv_path}\n"

    # ── FeatureAuditor results (always present; FeatureAuditorFailed
    #    would have terminated the pipeline before reaching here) ───
    if feature_audit_verdict is None:
        logging.warning(
            "[synthesize] feature_audit_verdict is None — "
            "FeatureAuditor may not have run. Using empty verdict."
        )
        feature_audit_verdict = {"feature_verdicts": [], "global_issues": []}

    wl_left = float(wl[0])
    wl_right = float(wl[-1])
    full_matrix, full_doublets, full_stats = build_contradiction_matrix(
        harness_results, harness_dir, wl_left, wl_right,
    )
    filtered_matrix, filtered_doublets, filtered_stats = (
        _filter_contradiction_matrix(
            full_matrix, full_doublets, full_stats, feature_audit_verdict,
        )
    )
    audit_section = _build_feature_audit_summary(feature_audit_verdict)
    verified_matrix_section = _build_verified_matrix_section(
        filtered_matrix, filtered_stats, harness_results,
    )
    surviving_doublets_section = _build_surviving_doublets_section(
        filtered_doublets, feature_audit_verdict,
    )
    oii_morphology_section = _build_oii_morphology_section(feature_audit_verdict)
    lyalpha_forest_section = _build_lyalpha_forest_section(feature_audit_verdict)
    composite_profile_section = _build_composite_profile_section(feature_audit_verdict)
    task_text = _build_verified_task(surviving=len(filtered_matrix) > 0)

    # ── Per-hypothesis full line tables (for CSV population) ──
    line_tables = _build_line_tables(harness_results, harness_dir)

    return f"""## Spectrum

- Wavelength range: {spec_wl_range} Å
- Median SNR: {f'{snr_median:.1f}' if snr_median else 'N/A'}
- Number of hypotheses tested: {len(harness_results)}
{_mode_note}
{_output_paths}
## Harness Report Summaries

Each summary distills the key structured information from the per-hypothesis
harness run. Expand the <details> sections to read the full report when a
deeper look at a specific hypothesis is needed.

{''.join(report_sections)}

## Pre-computed Dn4000 Diagnostics

{diagnostic_slices}

{audit_section}{verified_matrix_section}{surviving_doublets_section}
{oii_morphology_section}{lyalpha_forest_section}{composite_profile_section}
{line_tables}

{task_text}
"""


# ---------------------------------------------------------------------------
# Formatting helpers (for streaming output)
# ---------------------------------------------------------------------------

def _clean_rec(rec: str) -> str:
    """Normalise an LLM-generated recommendation string (see AnalysisAuditor)."""
    rec = (rec or "").strip()
    for q in ('"', "'", "`"):
        if rec.startswith(q) and rec.endswith(q) and len(rec) >= 2:
            rec = rec[1:-1]
            break
    return rec.strip().upper()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def arun(
    harness_results: list,
    wl: np.ndarray,
    fl: np.ndarray,
    harness_dir: str,
    *,
    snr: np.ndarray | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.1,
    max_turns: int = 150,
    stream_md_path: str | None = None,
    summaries: list[str] | None = None,
    mode: str = "nomad",
    report_path: str | None = None,
    csv_path: str | None = None,
    feature_audit_verdict: dict,
) -> dict:
    """Run the LLM synthesis agent over multiple harness reports.

    Parameters
    ----------
    harness_results : list
        Results from per-hypothesis harness runs. Each dict has keys:
        hypothesis_idx, redshift, report, structured_output,
        feature_catalog, and optionally error.
    wl, fl : np.ndarray
        Wavelength and flux arrays for the spectrum.
    harness_dir : str
        Directory containing per-hypothesis ``{idx}_report.md`` files
        (used as fallback if report text is not in *harness_results*).
    snr : np.ndarray, optional
        SNR array. Used for the median SNR line in the user prompt.
    model, api_key, base_url : str, optional
        LLM configuration. Defaults to environment variables.
    temperature : float
        LLM temperature (default 0.1).
    max_turns : int
        Maximum agent turns / recursion_limit (default 150).
    stream_md_path : str, optional
        If set, stream the full conversation (system prompt + every LLM
        turn + tool calls + tool results) to this .md file in real time.

    Returns
    -------
    dict
        The parsed synthesis verdict with keys: redshift, anchor_line,
        anchor_wavelength, wavelength_error,
        classification, confidence, best_hypothesis_idx, primary_evidence,
        excluded_hypotheses, caveats.
    """
    model = model or os.environ.get("LLM_MODEL", "deepseek-v4-pro")
    api_key = api_key or os.environ.get("LLM_API_KEY")
    base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")

    system_prompt = _load_skill()
    user_prompt = _build_user_message(harness_results, harness_dir, wl, fl,
                                      feature_audit_verdict,
                                      snr=snr, summaries=summaries,
                                      mode=mode, report_path=report_path, csv_path=csv_path)

    # ── Closure over arrays for zero-copy slicing ────────────────
    _wl = wl
    _fl = fl

    @tool
    def read_spectrum_region(
        wl_min: float,
        wl_max: float,
        stride: int = 1,
    ) -> dict:
        """Read a raw slice of the spectrum for manual inspection.

        Use this to investigate specific wavelength windows identified in
        Phase 2 of the synthesis strategy — e.g. check for the 4000 Å
        break, Ca K/H doublet, Balmer decrement, or continuum shape at
        discriminating wavelengths. Read 100–300 Å windows centered on
        the features that differentiate competing hypotheses.

        Parameters
        ----------
        wl_min, wl_max : float
            Wavelength range of interest (Å).
        stride : int
            Downsampling step. Default 1 (no downsampling). Use 2–5 for
            larger regions to keep output manageable.

        Returns
        -------
        dict with keys:
            wl_range : [float, float]
            n : int
            data : list[[float, float]]  — [[wl, fl], ...] pairs
        """
        mask = (_wl >= wl_min) & (_wl <= wl_max)
        wl_slice = _wl[mask][::stride]
        fl_slice = _fl[mask][::stride]
        return {
            "wl_range": [wl_min, wl_max],
            "n": len(wl_slice),
            "data": [
                [round(float(w), 3), round(float(f), 4)]
                for w, f in zip(wl_slice, fl_slice)
            ],
        }

    # ── Build LLM (thinking disabled for multi-turn tool calling) ─
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

    @tool
    def detect_oii_slope_change(target_wl: float, search_window: float = 25.0) -> dict:
        """Detect the [O II] unresolved doublet slope-change signature.

        Call this when competing hypotheses disagree on whether the same
        observed feature is [O II] vs [O III]b (Phase 2a tiebreaker #6).
        """
        return _detect_oii_slope_change_core(_wl, _fl, target_wl, search_window)

    agent = create_agent(
        model=llm,
        tools=[read_spectrum_region, grep_kb, write_report, write_synthesis_csv],
        system_prompt=system_prompt,
    )

    config = {"recursion_limit": max_turns}

    # ── Streaming path ────────────────────────────────────────────
    if stream_md_path:
        os.makedirs(os.path.dirname(stream_md_path) or ".", exist_ok=True)

        md = open(stream_md_path, "w", encoding="utf-8")
        md.write("# Synthesis\n\n")
        md.write("---\n\n")
        md.write("<details>\n<summary>System Prompt</summary>\n\n")
        md.write(system_prompt)
        md.write("\n</details>\n\n---\n\n")
        md.write(f"### User\n\n{user_prompt}\n\n")
        md.flush()

        accumulated_messages = []
        turn = 0

        try:
            async for event in agent.astream(
                {"messages": [("user", user_prompt)]},
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
                            md.write("### Tool Result\n\n")
                            md.write(_format_tool_result(msg))
                            md.write("\n\n")
                            md.flush()
        except Exception as exc:
            md.write(f"\n\n> ❌ Synthesis streaming failed: {exc}\n\n")
            md.flush()
            raise
        finally:
            md.close()

        # ── Truncation detection (shared continuation loop) ─
        await run_continuation_streaming(
            agent,
            accumulated_messages,
            config=config,
            stream_md_path=stream_md_path,
            continuation_prompt=(
                "Your previous response was truncated. Continue EXACTLY "
                "from where you stopped. Do NOT repeat anything. Output "
                "ONLY the remaining content — do not wrap in markdown fences."
            ),
            log_prefix="Synthesis",
        )

        # Concatenate ALL AI messages (original + continuation if truncated)
        # so that partial JSON from the original + completion from continuation
        # form a parseable whole.
        raw_content = "\n".join(
            msg.content for msg in accumulated_messages
            if getattr(msg, "type", None) == "ai" and hasattr(msg, "content")
        )

        parsed = _extract_json_block(raw_content)
        if parsed is None:
            parsed = {
                "redshift": None,
                "anchor_line": None,
                "anchor_wavelength": None,
                "wavelength_error": None,
                "classification": "Unknown",
                "confidence": "LOW",
                "best_hypothesis_idx": None,
                "primary_evidence": "Failed to parse JSON from synthesis response.",
                "caveats": raw_content[:500],
            }
        return parsed

    # ── Non-streaming path ────────────────────────────────────────
    try:
        result = await agent.ainvoke(
            {"messages": [("user", user_prompt)]},
            config=config,
        )

        # ── Truncation detection (shared continuation loop) ─
        messages = result.get("messages", [])
        await run_continuation_ainvoke(
            agent,
            messages,
            config=config,
            continuation_prompt=(
                "Your previous response was truncated. Continue EXACTLY "
                "from where you stopped. Do NOT repeat anything. Output "
                "ONLY the remaining content."
            ),
            log_prefix="Synthesis",
        )

        raw_content = "\n".join(
            msg.content for msg in messages
            if getattr(msg, "type", None) == "ai" and hasattr(msg, "content")
        )

        parsed = _extract_json_block(raw_content)
        if parsed is None:
            parsed = {
                "redshift": None,
                "anchor_line": None,
                "anchor_wavelength": None,
                "wavelength_error": None,
                "classification": "Unknown",
                "confidence": "LOW",
                "best_hypothesis_idx": None,
                "primary_evidence": "Failed to parse JSON from synthesis response.",
                "caveats": raw_content[:500],
            }

    except Exception as exc:
        logging.warning(f"Synthesis agent failed: {exc}")
        parsed = {
            "redshift": None,
            "anchor_line": None,
                "anchor_wavelength": None,
                "wavelength_error": None,
            "classification": "Unknown",
            "confidence": "LOW",
            "best_hypothesis_idx": None,
            "primary_evidence": f"Synthesis failed: {exc}",
            "caveats": "LLM synthesis step errored.",
        }

    return parsed
