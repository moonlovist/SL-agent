"""
Synthesis Host — Final Report Writing.

Runs AFTER the Result Auditor.  Calls the new harness
(``harness/report_writer.py``) to produce a structured 6-section final
report.  The LLM writes the report via ``write_report`` tool and outputs a
JSON comprehensive assessment block — no separate extraction step needed.
"""

import json
import logging
import os

from FORMA.agents.common.state import SpectroState
from FORMA.agents.common.base_agent import BaseAgent
from FORMA.core.runtime.runtime_container import RuntimeContainer


class ReportWriter(BaseAgent):
    """
    Synthesis Host: writes the final report and extracts structured summary.

    Calls ``report_writer.arun()`` which returns both the report Markdown
    and the parsed JSON comprehensive assessment.
    """

    agent_name = "ReportWriter"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    async def run(self, state: SpectroState) -> SpectroState:
        """Write final report and extract structured summary."""
        from FORMA.agents.multi_agents.harness import report_writer_arun

        harness_dir = state.get("harness_dir")
        if not harness_dir or not os.path.isdir(harness_dir):
            print("[ReportWriter] No harness_dir — skipping.")
            state["final_report"] = None
            state["in_brief"] = {}
            return state

        # ── Build LLM kwargs from runtime config ──
        llm_cfg = self.runtime.configs.model.llm
        llm_kwargs = {
            "model": llm_cfg.get("model", "deepseek-v4-pro"),
            "api_key": llm_cfg.get("api_key"),
            "base_url": llm_cfg.get("base_url"),
            "temperature": 0.3,
        }

        print("[ReportWriter] Writing final report ...")
        stream_path = os.path.join(harness_dir, "report_writer/stream.md")
        try:
            final_report, in_brief = await report_writer_arun(
                state, harness_dir, stream_md_path=stream_path, **llm_kwargs
            )
        except Exception as e:
            logging.warning(f"[ReportWriter] Report writing failed: {e}")
            state["final_report"] = None
            state["in_brief"] = {}
            return state

        state["final_report"] = final_report
        state["in_brief"] = in_brief or {}
        print(f"[ReportWriter] Report written. in_brief: {json.dumps(state['in_brief'], ensure_ascii=False) if state['in_brief'] else 'null'}")
        return state
