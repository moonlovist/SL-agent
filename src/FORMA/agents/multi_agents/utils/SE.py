"""
SelfEvolve utilities — failure analysis, recording, and batch processing.

These are exclusively consumed by ``SelfEvolve``.  Shared helpers (Dn4000,
harness summaries, JSON extraction) stay in ``RA.py``.
"""

import os
import json
import logging
import re
import numpy as np

from pathlib import Path

from FORMA.core.llm import _detect_vendor, _build_thinking_extra_body, _create_chat_openai
from FORMA.agents.multi_agents.utils.HA import (
    build_dn4000_lookup,
    _read_csv_lines,
    _build_line_table,
    format_structured_summary,
    _parse_extraction_json,
    extract_verdict_summary,
)


# ---------------------------------------------------------------------------
# Skill prompt loading
# ---------------------------------------------------------------------------

_HARNESS_DIR = Path(__file__).resolve().parent.parent / "harness" / "skills" / "SelfEvolve"


def _load_skill(filename: str) -> str:
    """Load a skill prompt from the SelfEvolve skills directory."""
    return (_HARNESS_DIR / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Failure analysis — two-round LLM root-cause identification
# ---------------------------------------------------------------------------

async def analyze_failure(
    synthesis_result: dict,
    harness_results: list,
    harness_dir: str,
    ground_truth: dict,
    mismatch_info: dict,
    *,
    model: str,
    api_key: str,
    base_url: str,
    wl: np.ndarray | None = None,
    fl: np.ndarray | None = None,
    temperature: float = 0.0,
    stream_md_path: str | None = None,
) -> dict:
    """Call an LLM to identify the root cause of a synthesis failure.

    Parameters
    ----------
    synthesis_result : dict
        The full rule_analysis dict from synthesis.
    harness_results : list
        Raw harness results (used to build summaries the synthesis saw).
    harness_dir : str
        Directory with harness reports and CSVs.
    ground_truth : dict
        ``{"z": float, "type": str}``.
    mismatch_info : dict
        ``{"z_mismatch": bool, "type_mismatch": bool}``.
    model, api_key, base_url : str
        LLM configuration.
    wl, fl : np.ndarray, optional
        Wavelength and flux arrays for Dn4000 computation. If None,
        a flat placeholder spectrum is used (Dn4000 will all be ~1.0).

    Returns
    -------
    dict
        Parsed failure analysis with keys: root_cause, explanation, suggested_fix.
    """
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from FORMA.agents.multi_agents.harness.tools import grep_kb

    # ── Build harness summaries (same as what synthesis saw) ──
    if wl is None or fl is None:
        wl = np.linspace(3600, 9824, 7000)
        fl = np.ones(7000) * 5.0
    dn4000_lookup = build_dn4000_lookup(wl, fl, harness_results)
    summaries = []
    for i, r in enumerate(harness_results):
        idx = r.get("hypothesis_idx", i + 1)
        csv_path = os.path.join(harness_dir, f"single_hypothesis/{idx}_lines.csv")
        csv_rows = _read_csv_lines(csv_path) if os.path.exists(csv_path) else []
        table_rows, nf_names, sp_names, z_scat = _build_line_table(csv_rows)
        structured = r.get("structured_output") or {}
        summaries.append(format_structured_summary(
            hypothesis_idx=idx,
            z_tested=r.get("redshift", 0),
            text_fields=structured,
            table_rows=table_rows,
            not_found_names=nf_names,
            spurious_names=sp_names,
            z_scatter=z_scat,
            dn4000_lookup=dn4000_lookup,
        ))

    # ── Extract key synthesis reasoning ──
    stream_path = os.path.join(harness_dir, "hypothesis_synthesis/stream.md")
    synthesis_reasoning = ""
    if os.path.exists(stream_path):
        text = Path(stream_path).read_text(encoding="utf-8")
        parts = re.split(r"### Assistant \(turn \d+\)", text)
        if len(parts) >= 2:
            synthesis_reasoning += "### Phase 1 Analysis\n\n"
            turn1 = parts[1]
            m = re.search(r"```json\s*\{.*?\}\s*```", turn1, re.DOTALL)
            if m:
                synthesis_reasoning += turn1[:m.start()].strip()[-3000:]
            else:
                synthesis_reasoning += turn1.strip()[-3000:]
        if len(parts) >= 3:
            synthesis_reasoning += "\n\n### Final Assessment\n\n"
            last = parts[-1].strip()[-3000:]
            synthesis_reasoning += last

    # ── Build mismatch description ──
    desc_parts = []
    if mismatch_info.get("z_mismatch"):
        desc_parts.append(
            f"redshift: expected {ground_truth['z']}, got {synthesis_result.get('redshift')}"
        )
    if mismatch_info.get("type_mismatch"):
        desc_parts.append(
            f"type: expected {ground_truth['type']}, got {synthesis_result.get('classification')}"
        )
    mismatch_desc = "; ".join(desc_parts)

    harness_text = "\n\n".join(summaries)

    # Always disable thinking — multi-turn tool calling can't pass back reasoning_content
    _vendor = _detect_vendor(base_url)
    _extra_body = _build_thinking_extra_body("disabled", _vendor) if _vendor != "unknown" else None
    llm = _create_chat_openai(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        extra_body=_extra_body,
    )

    async def _stream_one(md, label: str, prompt: str) -> str:
        """Stream a single LLM call into *md*, return accumulated text."""
        md.write(f"### {label}\n\n")
        md.write("<details>\n<summary>Prompt</summary>\n\n")
        md.write(prompt)
        md.write("\n</details>\n\n---\n\n")
        md.flush()

        accumulated = ""
        try:
            async for chunk in llm.astream([("user", prompt)]):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    accumulated += token
                    md.write(token)
                    md.flush()
        except Exception as exc:
            md.write(f"\n\n> ❌ {label} streaming failed: {exc}\n\n")
            md.flush()
            raise
        md.write("\n\n---\n\n")
        md.flush()
        return accumulated

    async def _invoke_one(prompt: str) -> str:
        resp = await llm.ainvoke([("user", prompt)])
        return resp.content if hasattr(resp, "content") else str(resp)

    def _fmt_tool_call(tc) -> str:
        name = tc.get("name", "unknown")
        args = tc.get("args", {})
        return f"**`{name}`**\n```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```"

    def _fmt_tool_result(msg) -> str:
        content = msg.content if hasattr(msg, "content") else str(msg)
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
            return f"```json\n{json.dumps(parsed, indent=2, ensure_ascii=False)}\n```"
        except (json.JSONDecodeError, TypeError):
            return str(content)

    async def _agent_stream(md, label: str, agent, prompt: str, config: dict) -> str:
        """Stream an agent call (with tool use) into *md*, return final text."""
        md.write(f"### {label}\n\n")
        md.write("<details>\n<summary>Prompt</summary>\n\n")
        md.write(prompt)
        md.write("\n</details>\n\n---\n\n")
        md.flush()

        accumulated = []
        turn = 0
        try:
            async for event in agent.astream(
                {"messages": [("user", prompt)]},
                config=config,
                stream_mode="updates",
            ):
                for _node_name, update in event.items():
                    msgs = update.get("messages", [])
                    for msg in msgs:
                        accumulated.append(msg)
                        msg_type = getattr(msg, "type", None)

                        if msg_type == "ai":
                            turn += 1
                            content = msg.content if hasattr(msg, "content") else ""
                            tool_calls = getattr(msg, "tool_calls", None)
                            if content:
                                md.write(f"### Assistant (turn {turn})\n\n{content.strip()}\n\n")
                            if tool_calls:
                                for tc in tool_calls:
                                    md.write(_fmt_tool_call(tc) + "\n\n")
                            md.flush()
                        elif msg_type == "tool":
                            md.write("### Tool Result\n\n")
                            md.write(_fmt_tool_result(msg) + "\n\n")
                            md.flush()
        except Exception as exc:
            md.write(f"\n\n> ❌ {label} streaming failed: {exc}\n\n")
            md.flush()
            raise
        md.write("\n\n---\n\n")
        md.flush()

        last_msg = accumulated[-1] if accumulated else None
        return last_msg.content if last_msg and hasattr(last_msg, "content") else ""

    # ── Round 1: blind review ────────────────────────────────────
    blind_prompt = _load_skill("blind_review_skill.md").format(
        harness_summaries=harness_text,
        synthesis_reasoning=synthesis_reasoning[:6000],
    )

    # ── Round 2: root cause with ground truth ────────────────────
    root_cause_prompt_tpl = _load_skill("root_cause_skill.md")

    # ── Round 2 agent tools ─────────────────────────────────────
    _wl = wl
    _fl = fl

    @tool
    def read_spectrum_region(wl_min: float, wl_max: float, stride: int = 1) -> dict:
        """Read a raw slice of the spectrum for manual inspection.

        Use this to verify whether harness-reported features are real at the
        claimed wavelengths, check noise levels, or examine discriminating
        regions that were overlooked by synthesis. Keep windows 50–200 Å.

        Parameters
        ----------
        wl_min, wl_max : float
            Wavelength range of interest (Å).
        stride : int
            Downsampling step. Default 1 (no downsampling).
        """
        mask = (_wl >= wl_min) & (_wl <= wl_max)
        wl_slice = _wl[mask][::stride]
        fl_slice = _fl[mask][::stride]
        return {
            "wl_range": [wl_min, wl_max],
            "n": len(wl_slice),
            "wl": [round(float(w), 3) for w in wl_slice],
            "fl": [round(float(f), 4) for f in fl_slice],
        }

    root_cause_agent = create_agent(
        model=llm,
        tools=[read_spectrum_region, grep_kb],
        system_prompt=root_cause_prompt_tpl,
    )
    agent_config = {"recursion_limit": 15}

    if stream_md_path:
        os.makedirs(os.path.dirname(stream_md_path) or ".", exist_ok=True)

        with open(stream_md_path, "w", encoding="utf-8") as md:
            md.write("# Failure Analysis (two-round)\n\n")
            md.write(f"**Ground truth**: z={ground_truth['z']}, type={ground_truth['type']}\n\n")
            md.write(f"**Synthesis**: z={synthesis_result.get('redshift')}, "
                     f"type={synthesis_result.get('classification')}, "
                     f"confidence={synthesis_result.get('confidence')}\n\n")
            md.write("---\n\n")

            # Round 1
            try:
                blind_text = await _stream_one(md, "Round 1 — Blind Review", blind_prompt)
            except Exception as exc:
                logging.warning(f"Failure analysis Round 1 stream failed: {exc}")
                return _fallback_analysis()

            blind_parsed = _parse_extraction_json(blind_text)
            blind_summary = (
                json.dumps(blind_parsed, indent=2, ensure_ascii=False)
                if blind_parsed
                else blind_text[:2000]
            )

            # Round 2 (agent with read_spectrum_region + grep_kb)
            root_prompt = root_cause_prompt_tpl.format(
                blind_review=blind_summary,
                ground_truth_z=ground_truth["z"],
                ground_truth_type=ground_truth["type"],
                in_scoring=mismatch_info.get("in_scoring", True),
                synthesis_z=synthesis_result.get("redshift"),
                synthesis_type=synthesis_result.get("classification"),
                synthesis_confidence=synthesis_result.get("confidence"),
                mismatch_desc=mismatch_desc,
                harness_summaries=harness_text,
            )

            try:
                root_text = await _agent_stream(
                    md, "Round 2 — Root Cause Analysis",
                    root_cause_agent, root_prompt, agent_config,
                )
            except Exception as exc:
                logging.warning(f"Failure analysis Round 2 stream failed: {exc}")
                return _fallback_analysis()

        parsed = _parse_extraction_json(root_text)
        if parsed:
            parsed["_blind_review"] = blind_parsed
            return parsed
        return _fallback_analysis()

    # ── Non-streaming path ────────────────────────────────────────
    try:
        blind_text = await _invoke_one(blind_prompt)
    except Exception as exc:
        logging.warning(f"Failure analysis Round 1 failed: {exc}")
        return _fallback_analysis()

    blind_parsed = _parse_extraction_json(blind_text)
    blind_summary = (
        json.dumps(blind_parsed, indent=2)
        if blind_parsed
        else blind_text[:2000]
    )

    root_prompt = root_cause_prompt_tpl.format(
        blind_review=blind_summary,
        ground_truth_z=ground_truth["z"],
        ground_truth_type=ground_truth["type"],
        in_scoring=mismatch_info.get("in_scoring", True),
        synthesis_z=synthesis_result.get("redshift"),
        synthesis_type=synthesis_result.get("classification"),
        synthesis_confidence=synthesis_result.get("confidence"),
        mismatch_desc=mismatch_desc,
        harness_summaries=harness_text,
    )

    try:
        result = await root_cause_agent.ainvoke(
            {"messages": [("user", root_prompt)]},
            config=agent_config,
        )
        messages = result.get("messages", [])
        last_msg = messages[-1]
        root_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    except Exception as exc:
        logging.warning(f"Failure analysis Round 2 failed: {exc}")
        return _fallback_analysis()

    parsed = _parse_extraction_json(root_text)
    if parsed:
        parsed["_blind_review"] = blind_parsed
        return parsed
    return _fallback_analysis()


def _fallback_analysis() -> dict:
    return {
        "root_cause": "ambiguous",
        "explanation": "LLM-based analysis failed to produce a parseable result.",
        "suggested_fix": {
            "target_file": None,
            "target_section": None,
            "proposed_change": None,
            "rationale": "Automatic analysis failed — requires manual review.",
        },
    }


# ---------------------------------------------------------------------------
# Batch failure accumulation & analysis
# ---------------------------------------------------------------------------

def _append_pending_failure(output_dir: str, record: dict) -> str:
    """Append a failure record to the pending batch queue.

    Returns the path to pending.jsonl.
    """
    failure_dir = os.path.join(output_dir, "failure")
    os.makedirs(failure_dir, exist_ok=True)
    pending_path = os.path.join(failure_dir, "pending.jsonl")
    with open(pending_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return pending_path


def _read_pending_failures(output_dir: str) -> list[dict]:
    """Read all pending failure records."""
    pending_path = os.path.join(output_dir, "failure", "pending.jsonl")
    if not os.path.exists(pending_path):
        return []
    records = []
    with open(pending_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _clear_pending_failures(output_dir: str) -> None:
    """Remove the pending queue after batch analysis."""
    pending_path = os.path.join(output_dir, "failure", "pending.jsonl")
    if os.path.exists(pending_path):
        os.remove(pending_path)


def _load_batch_skill() -> str:
    return _load_skill("batch_analysis_skill.md")


async def analyze_failure_batch(
    output_dir: str,
    batch_num: int,
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
) -> dict | None:
    """Run a collective LLM analysis on a batch of pending failures.

    Reads failures from ``output_dir/failure/pending.jsonl``, streams the
    LLM response to ``output_dir/failure/batch_{batch_num}_stream.md``,
    and clears the pending queue on success.

    Returns the parsed batch analysis dict, or None if no pending failures
    or the LLM call fails.
    """
    pending = _read_pending_failures(output_dir)
    if not pending:
        return None

    # Build compact failure summaries
    summary_lines = []
    for i, rec in enumerate(pending):
        gt = rec.get("ground_truth", {})
        sr = rec.get("synthesis_result", {})
        mm = rec.get("mismatch", {})
        summary_lines.append(
            f"**{i + 1}. {rec.get('spectrum_id', '?')}** — "
            f"expected z={gt.get('z')} type={gt.get('type')}, "
            f"got z={sr.get('redshift')} type={sr.get('classification')} "
            f"(confidence={sr.get('confidence')}); "
            f"in_scoring={mm.get('in_scoring')}, min_dz={mm.get('min_dz')}"
        )

    prompt = _load_batch_skill().format(
        n_failures=len(pending),
        failure_summaries="\n".join(summary_lines),
    )

    # Always disable thinking — multi-turn tool calling can't pass back reasoning_content
    _vendor = _detect_vendor(base_url)
    _extra_body = _build_thinking_extra_body("disabled", _vendor) if _vendor != "unknown" else None
    llm = _create_chat_openai(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        extra_body=_extra_body,
    )

    failure_dir = os.path.join(output_dir, "failure")
    os.makedirs(failure_dir, exist_ok=True)
    stream_path = os.path.join(failure_dir, f"batch_{batch_num}_stream.md")

    with open(stream_path, "w", encoding="utf-8") as md:
        md.write(f"# Batch Failure Analysis #{batch_num}\n\n")
        md.write(f"**Failures**: {len(pending)}\n\n")
        md.write("---\n\n")
        md.write("<details>\n<summary>Prompt</summary>\n\n")
        md.write(prompt)
        md.write("\n</details>\n\n---\n\n")
        md.write("## Response\n\n")
        md.flush()

        accumulated = ""
        try:
            async for chunk in llm.astream([("user", prompt)]):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    accumulated += token
                    md.write(token)
                    md.flush()
        except Exception as exc:
            md.write(f"\n\n> ❌ Batch analysis streaming failed: {exc}\n\n")
            md.flush()
            logging.warning(f"Batch analysis LLM stream failed: {exc}")
            return None

    # Also write a clean markdown version (no prompt dump)
    report_path = os.path.join(failure_dir, f"batch_{batch_num}.md")
    parsed = _parse_extraction_json(accumulated)
    if parsed:
        _write_batch_report(report_path, batch_num, pending, parsed)
        _clear_pending_failures(output_dir)
        return parsed

    # Parse failed — still write the raw response
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Batch Failure Analysis #{batch_num}\n\n")
        f.write(f"**Failures**: {len(pending)}\n\n")
        f.write("## Raw Response (parse failed)\n\n")
        f.write(accumulated)
    _clear_pending_failures(output_dir)
    return None


def _write_batch_report(
    report_path: str,
    batch_num: int,
    pending: list[dict],
    analysis: dict,
) -> None:
    """Write a formatted batch analysis report in markdown."""
    lines = [
        f"# Batch Failure Analysis #{batch_num}",
        "",
        f"**Failures reviewed**: {len(pending)}",
        "",
        f"## Diagnosis",
        "",
        analysis.get("batch_diagnosis", "(none)"),
        "",
        "---",
        "",
        "## Failure Groups",
        "",
    ]

    for gi, group in enumerate(analysis.get("groups", [])):
        rc = group.get("root_cause", "?")
        count = group.get("count", 0)
        ids = ", ".join(group.get("spectrum_ids", []))
        pattern = group.get("pattern", "")
        diagnosis = group.get("diagnosis", "")
        fix = group.get("suggested_fix") or {}

        lines.append(f"### {gi + 1}. {rc} ({count} failures)")
        lines.append(f"**Spectra**: {ids}")
        lines.append(f"**Pattern**: {pattern}")
        lines.append("")
        lines.append(diagnosis)
        lines.append("")

        target = fix.get("target_file") or "—"
        section = fix.get("target_section") or "—"
        change = fix.get("proposed_change") or "—"
        rationale = fix.get("rationale") or "—"
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Target file | {target} |")
        lines.append(f"| Target section | {section} |")
        lines.append(f"| Proposed change | {change} |")
        lines.append(f"| Rationale | {rationale} |")
        lines.append("")

    # Priority fix
    pf = analysis.get("priority_fix") or {}
    if pf:
        lines.append("---")
        lines.append("")
        lines.append("## Priority Fix")
        lines.append("")
        lines.append(f"**Group {pf.get('group_index', -1) + 1}** — {pf.get('reasoning', '')}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Failure record writer
# ---------------------------------------------------------------------------

def _record_failure(
    synthesis_result: dict,
    harness_dir: str,
    ground_truth: dict,
    mismatch_info: dict,
    analysis: dict,
    output_dir: str = "",
) -> str:
    """Write a structured failure record.

    Three outputs:
    1. Per-sample JSON written to ``{harness_dir}/failure_analysis.json``.
    2. Per-sample JSONL appended to ``{output_dir}/failure/pending.jsonl``
       (for batch accumulation).
    3. Per-sample JSONL appended to ``{output_dir}/failure_log.jsonl``
       (global log alongside harness dirs).

    Returns the pending path written to.
    """
    import datetime

    if not output_dir:
        output_dir = harness_dir

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "spectrum_id": os.path.basename(harness_dir).removesuffix('_harness'),
        "ground_truth": ground_truth,
        "synthesis_result": extract_verdict_summary(synthesis_result),
        "mismatch": mismatch_info,
        "root_cause": analysis.get("root_cause"),
        "explanation": analysis.get("explanation"),
        "suggested_fix": analysis.get("suggested_fix"),
        "harness_dir": harness_dir,
        "synthesis_stream": os.path.join(harness_dir, "hypothesis_synthesis/stream.md"),
        "failure_analysis_stream": os.path.join(harness_dir, "self_evolve/failure_analysis_stream.md"),
    }

    # Per-sample stream to harness dir (individual review)
    harness_failure_path = os.path.join(harness_dir, "self_evolve/failure_analysis.json")
    with open(harness_failure_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # Append to batch pending queue in output_dir/failure/
    pending_path = _append_pending_failure(output_dir, record)

    # Also write to global archive alongside harness dirs
    global_log_path = os.path.join(output_dir, "failure_log.jsonl")
    with open(global_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(f"Failure record written: harness={harness_failure_path}, pending={pending_path}")
    return pending_path
