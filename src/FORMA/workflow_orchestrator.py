import asyncio
import logging
import traceback

from typing import Dict, Any, List, Optional, Set, Union

from langgraph.graph import StateGraph, END, START

from FORMA.core.runtime.runtime_container import RuntimeContainer
from FORMA.agents.common.state import SpectroState
from FORMA.agents.common.base_agent import _is_connection_error, _is_timeout_error
import copy

#########################
from FORMA.agents.multi_agents.VisualInterpreter import VisualInterpreter
from FORMA.agents.multi_agents.HypothesisAnalyst import HypothesisAnalyst
from FORMA.agents.multi_agents.SelfEvolve import SelfEvolve
from FORMA.agents.multi_agents.AnalysisAuditor import AnalysisAuditor, FeatureAuditor
from FORMA.agents.multi_agents.ReportWriter import ReportWriter
#########################


class WorkflowOrchestrator:
    """
    工作流编排器：管理整个智能体交互流程
    Workflow orchestrator: manage the entire agent interaction workflow
    """

    # 定义语言到代理类的映射
    AGENT_CLASSES = {
        'VisualInterpreter': VisualInterpreter,
        'HypothesisAnalyst': HypothesisAnalyst,
        'SelfEvolve': SelfEvolve,
        'AnalysisAuditor': AnalysisAuditor,
        'FeatureAuditor': FeatureAuditor,
        'ReportWriter': ReportWriter
    }

    def __init__(
            self,
            configs: Dict[str, Any],
        ):

        self.configs = configs

        # 初始化运行时容器
        self.runtime = RuntimeContainer(self.configs)

        # 初始化所有智能体
        self.spectro_agents = self._initialize_agents()

        # 创建状态图
        self.workflow = self._create_workflow()


        print("🚀 工作流编排器初始化完成")

    def _initialize_agents(self) -> Dict[str, Any]:
        """
        初始化所有智能体
        Initialize all agents
        """
        spectro_agents = {
            '_Visual_Interpreter': self.AGENT_CLASSES['VisualInterpreter'](self.runtime),
            '_Hypothesis_Analyst': self.AGENT_CLASSES['HypothesisAnalyst'](self.runtime),
            '_Self_Evolve': self.AGENT_CLASSES['SelfEvolve'](self.runtime),
            '_Analysis_Auditor': self.AGENT_CLASSES['AnalysisAuditor'](self.runtime),
            '_Feature_Auditor': self.AGENT_CLASSES['FeatureAuditor'](self.runtime),
            '_Report_Writer': self.AGENT_CLASSES['ReportWriter'](self.runtime)
        }
        print(f"Initialized {len(spectro_agents)} agents")
        return spectro_agents

    def _check_cancel(self):
        """
        检查是否需要取消分析
        Check if analysis needs to be canceled
        """
        if self.cancel_checker and callable(self.cancel_checker):
            if self.cancel_checker():
                raise asyncio.CancelledError("分析已被用户取消 Analysis canceled by user")

    async def _visual_interpreter_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 1: Visual Interpreter')
        result = await self.spectro_agents["_Visual_Interpreter"].run(state, plot=True)
        self._check_cancel()
        # Flag empty spectra for early exit
        peaks = result.get("peaks") or []
        troughs = result.get("troughs") or []
        result["_no_features"] = (len(peaks) == 0 and len(troughs) == 0)
        if result["_no_features"]:
            print("[VI] No emission or absorption features detected — aborting pipeline.")
        return result

    def _has_features(self, state: SpectroState) -> str:
        """Route: skip downstream agents if VI found no features."""
        if state.get("_no_features"):
            return "no_features"
        return "continue"

    async def _no_features_node(self, state: SpectroState) -> SpectroState:
        """Write placeholder report and JSON when no features were detected."""
        import os, json as _json
        self._check_cancel()
        print("Stage 1b: No features — writing placeholder report.")

        wl = state["spectrum"]["wavelength"]
        wl_left = float(wl[0])
        wl_right = float(wl[-1])
        snr_val = state["spectrum"].get("snr")
        import numpy as np
        snr_median = float(np.median(snr_val)) if snr_val is not None else None
        continuum = state.get("continuum") or {}
        continuum_desc = continuum.get("description", "No continuum data available.")

        harness_dir = state.get("harness_dir", "")
        file_name = state.get("file_name", "unknown")

        # ── Placeholder report ──
        report = f"""# Final Analysis Report

## §1: Spectrum Basic Information

- Wavelength coverage: {wl_left:.0f} – {wl_right:.0f} Å
- Median SNR: {f'{snr_median:.1f}' if snr_median else 'N/A'}
- Continuum shape: {continuum_desc}

**⚠ No emission or absorption features were detected in this spectrum.** The Visual Interpreter's CWT feature detection found zero peaks and zero troughs. The spectrum may be pure noise, or the signal is below the detection threshold.

## §2: Hypothesis Summary

No hypotheses were tested — the pipeline aborted after feature detection found no signal.

## §3: Synthesis & Audit Judgments

Skipped — no features to analyse.

## §4: Potential Issues

- **No signal detected**: CWT feature detection found no emission peaks or absorption troughs across the entire wavelength range.
- This spectrum may be noise-dominated, or the object may be too faint for DESI to detect at this exposure.

## §5: Comprehensive Assessment

1. **Final object type**: Unknown
2. **Recommended redshift**: null (no features to determine redshift)
3. **Confirmed lines**: none
4. **Signal clarity score**: 0 — no credible signal
5. **Confidence**: LOW
6. **Recommend human review**: Yes

## §6: Conclusion Summary

No spectral features were detected in this exposure. The spectrum appears to contain no detectable astrophysical signal above the noise floor. Deeper observation or re-observation at higher SNR is recommended to determine whether a real source is present.
"""

        # ── Write report ──
        if harness_dir:
            os.makedirs(harness_dir, exist_ok=True)
            report_path = os.path.join(harness_dir, "final_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

        # ── Placeholder JSON ──
        in_brief = {
            "type": "Unknown",
            "signal_clarity": 0,
            "redshift": None,
            "redshift_rms": None,
            "lines": [],
            "confidence": "LOW",
            "human_review": "Yes",
        }

        state["final_report"] = report
        state["in_brief"] = in_brief
        state["hypothesis_analysis"] = {"redshift": None, "classification": "Unknown", "confidence": "LOW"}
        state["auditor_verdict_json"] = {
            "verdict": "UNCERTAIN",
            "calibrated_confidence": "LOW",
            "has_real_peak": False,
            "confirmed_lines": [],
            "key_issues": ["No features detected by Visual Interpreter."],
        }
        state["feature_audit_verdict"] = {"spectrum_quality": "noise-dominated", "skipped": True}
        return state

    async def _hypothesis_analyst_search_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 2: Hypothesis Analyst — Single-Hypothesis Search')
        result = await self.spectro_agents["_Hypothesis_Analyst"].run(state)
        self._check_cancel()
        return result

    async def _feature_auditor_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 3: Feature Auditor — Cross-Hypothesis Verification')
        result = await self.spectro_agents["_Feature_Auditor"].run(state)
        self._check_cancel()
        return result

    async def _hypothesis_analyst_synthesize_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 4: Hypothesis Analyst — Synthesis')
        result = await self.spectro_agents["_Hypothesis_Analyst"].run_hypothesis_synthesis(state)
        self._check_cancel()
        return result

    async def _analysis_auditor_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 5: Analysis Auditor — Synthesis Audit')
        result = await self.spectro_agents["_Analysis_Auditor"].run(state)
        self._check_cancel()
        return result

    async def _self_evolve_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 2b: Self-Evolve — Ground-truth Check')
        result = await self.spectro_agents["_Self_Evolve"].run(state)
        self._check_cancel()
        return result

    async def _report_writer_node(self, state: SpectroState) -> SpectroState:
        self._check_cancel()
        print('Stage 6: Synthesis Host — Final Report & Info Extraction')
        result = await self.spectro_agents["_Report_Writer"].run(state)
        self._check_cancel()
        return result

    def _create_workflow(self) -> StateGraph:

        workflow = StateGraph(SpectroState)

        workflow.add_node("visual_interpreter", self._visual_interpreter_node)
        workflow.add_node("no_features", self._no_features_node)
        workflow.add_node("hypothesis_analyst_search", self._hypothesis_analyst_search_node)
        workflow.add_node("feature_auditor", self._feature_auditor_node)
        workflow.add_node("hypothesis_analyst_synthesize", self._hypothesis_analyst_synthesize_node)
        workflow.add_node("analysis_auditor", self._analysis_auditor_node)
        workflow.add_node("report_writer", self._report_writer_node)

        workflow.add_edge(START, 'visual_interpreter')
        workflow.set_entry_point("visual_interpreter")
        workflow.add_conditional_edges(
            "visual_interpreter",
            self._has_features,
            {"continue": "hypothesis_analyst_search", "no_features": "no_features"},
        )
        workflow.add_edge("hypothesis_analyst_search", "feature_auditor")
        workflow.add_edge("feature_auditor", "hypothesis_analyst_synthesize")
        workflow.add_edge("hypothesis_analyst_synthesize", "analysis_auditor")
        workflow.add_edge("analysis_auditor", "report_writer")
        workflow.add_edge("report_writer", END)
        workflow.add_edge("no_features", END)

        return workflow.compile()


    async def run_analysis_single(self, state, cancel_checker=None) -> SpectroState:

        print("🚀 Start LLM Spectro Agent")
        # 存储取消检查器
        self.cancel_checker = cancel_checker

        # 保存深拷贝的原始输入，用于重试时恢复干净状态
        initial_state = copy.deepcopy(state)

        max_tries = self.configs.max_tries or 3
        retry_delay = self.configs.retry_delay or 180

        current_state = initial_state

        for attempt in range(max_tries):
            try:
                # 检查取消状态
                self._check_cancel()

                # 运行工作流
                workflow_result = await self.workflow.ainvoke(current_state)
                final_state = workflow_result

                print("✅ 分析流程完成")
                return final_state

            except asyncio.CancelledError as e:
                print(f"⚠️ 分析流程已取消: {e}")
                return current_state

            except Exception as e:
                error_msg = str(e).lower()
                if attempt < max_tries - 1 and _is_connection_error(error_msg):
                    logging.warning(
                        f"🌐 工作流遇到连接错误，{retry_delay}秒后重试..."
                        f" (尝试 {attempt + 1}/{max_tries}): {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    # 重置中间结果，从干净状态重跑，避免脏数据（尤其是 List append 字段）污染
                    current_state = copy.deepcopy(initial_state)
                    logging.warning("🧹 已清除中间结果字段，将从头重新执行工作流")
                elif attempt < max_tries - 1 and _is_timeout_error(error_msg):
                    logging.warning(
                        f"⏱️ 工作流遇到超时，{retry_delay}秒后重试..."
                        f" (尝试 {attempt + 1}/{max_tries}): {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    current_state = copy.deepcopy(initial_state)
                    logging.warning("🧹 已清除中间结果字段，将从头重新执行工作流")
                else:
                    print(f"❌ 分析流程失败: {e}")
                    traceback.print_exc()
                    return current_state

        # 所有重试耗尽
        print(f"❌ 分析流程在 {max_tries} 次尝试后仍失败，放弃")
        return current_state
