from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState
from ..core.llm_client import failure_message


class ReportWriter(BaseAgent):
    """Render the already-audited assessment without adding scientific claims."""

    def __init__(self):
        super().__init__("ReportWriter")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("将已审查的专家判断写成可追溯报告")
        dossier = state.evidence_dossier
        assessment = state.expert_assessment
        audit = state.decision_audit
        if dossier is None or assessment is None or audit is None:
            raise ValueError("ReportWriter requires assessment and decision audit")
        publishable = audit.audit_passed and assessment.vote_category in {"A", "B", "C"}
        published_grade = assessment.vote_category if publishable else "unknown"
        publication_reason = (
            "provisional_q1_style_category_not_catalog_calibrated" if publishable
            else "audit_failed" if not audit.audit_passed
            else "non_lens_vote_X" if assessment.vote_category == "X"
            else "insufficient_evidence_or_unresolved_processing"
        )

        def fact_lines(items):
            return [
                f"- `{item.id}`: {item.statement} (Observation refs: {', '.join(item.observation_refs)})"
                for item in items
            ] or ["- None recorded."]

        missing_lines = [
            f"- {item.topic}: **{item.status}** - {item.reason}"
            for item in dossier.missing_information
        ] or ["- None recorded."]
        alternative_lines = [
            f"- `{item.id}`: **{item.status}** - {item.rationale}"
            for item in dossier.competing_explanations
        ]
        configuration_lines = [
            f"- `{item.id}`: **{item.status}** - {item.rationale}"
            for item in dossier.configuration_hypotheses
        ] or ["- None evaluated."]
        markdown = "\n".join(
            [
                f"# Strong-lens expert assessment: {state.object_id}",
                "",
                f"- Published grade: **{published_grade}** ({publication_reason})",
                "- 算法测量 + 固定规则；无多模态模型，无生成式补弧；尚未完成 Q1 目录校准。",
                "- 唯一正式评级来自一次证据裁决；核查用重放不是第二次投票。",
                "",
                "## Evidence dossier",
                "",
                f"- Signal reality: **{dossier.signal_reality.status}** - {dossier.signal_reality.rationale}",
                f"- Lens specificity: **{dossier.lens_specificity.status}** - {dossier.lens_specificity.rationale}",
                "",
                "### Observed facts used",
                "",
                *fact_lines(dossier.observed_facts),
                "",
                "### Competing explanations",
                "",
                *alternative_lines,
                "",
                "### Lens-configuration hypotheses",
                "",
                *configuration_lines,
                "",
                "### Missing information",
                "",
                *missing_lines,
                "",
                "### Processing cautions / 处理风险",
                "",
                *[f"- {item.id}: {item.description}" for item in dossier.processing_cautions],
                "",
                "## Provisional expert assessment",
                "",
                f"- Vote category: **{assessment.vote_category}**",
                f"- Score status: **{assessment.score_status}**",
                "- Official expert score: **null**",
                f"- Rationale: {assessment.assessment_rationale}",
                "",
                "## Independent decision audit",
                "",
                f"- Passed: **{audit.audit_passed}**",
                f"- Abstention recommended: **{audit.abstention_recommended}**",
                f"- Required corrections: {audit.required_corrections or 'none'}",
                "",
                "## Grading contract",
                "",
                "The current output is one categorical single-agent vote. It does not manufacture "
                "a numeric expert_score from one agent. "
                "A formal official_expert_score remains null until labelled Q1 calibration and "
                "independent validation are completed.",
                "",
                "Foreground selection uses the target position rather than the brightest neighbour. "
                "Curved structures have pixel-derived local morphology; comparison threshold occupancy "
                "is not a retained-flux fraction. Signed same-aperture flux checks model sensitivity, "
                "not independent observations. Missing measurements are unknown.",
            ]
        )
        candidate_evidence = state.candidate_evidence
        if candidate_evidence is not None:
            qualified = [item for item in candidate_evidence.grading_items if item.role != "excluded"]
            markdown = "\n".join([
                f"# Strong-lens expert assessment: {state.object_id}", "",
                "## 候选证据纳入情况", "",
                f"- 纳入评级的候选：{', '.join(item.candidate_id for item in qualified) or '无'}",
                f"- 构型关系：{len(candidate_evidence.configurations)} 组",
                f"- 竞争性非透镜指标：{'有' if candidate_evidence.competitive_non_lens_explanation else '无'}",
                "- model-chain 候选已经直接进入上方唯一的正式评级，不再另列基线与修订评级。", "",
                *markdown.splitlines()[2:],
            ])
        if state.require_observation().candidate_handoff_review is not None:
            from .candidate_review import render_candidate_review_cn

            markdown += "\n" + "\n".join(render_candidate_review_cn(state.require_observation()))
            markdown += "\n\n### 候选路线分组（搜索来源，不等同于评级构型）\n\n"
            markdown += "\n".join(
                f"- {item.id}: {', '.join(item.candidate_ids)}；路线只记来源，不作为独立票。"
                for item in dossier.candidate_configuration_review
            ) or "未形成可复核分组。"
        if candidate_evidence is not None:
            markdown += "\n\n### 正式评级采用的候选空间构型\n\n"
            markdown += "\n".join(
                f"- {item.id}: clear arc {item.primary_candidate_id} + companion {item.companion_candidate_id}; 方位分离 {item.azimuth_separation_deg:.1f}°，半径比 {item.radius_ratio:.2f}。"
                for item in candidate_evidence.configurations
            ) or "- 没有达到当前策略要求的 clear-arc + companion 构型。"
        if state.evidence_preparation:
            markdown += "\n\n## 确定性证据合并记录\n\n"
            markdown += "同一套最小像素/名义信号条件筛查所有来源；纳入表示可测量，不表示真实透镜。详细数值、掩膜指纹及规则参数见 summary.json 的 evidence_preparation。\n\n"
            markdown += "| 候选 | 进入本次判断 | 结构引用 | 原因 |\n|---|---|---|---|\n"
            for item in state.evidence_preparation.get("items", []):
                markdown += f"| {item['candidate_id']} | {'是' if item['included'] else '否'} | {item['structure_id'] or '—'} | {'; '.join(item['reasons'])} |\n"
            conditions = next((t["content"] for t in state.reasoning_trace if t["action"] == "decision_conditions"), {})
            markdown += "\n### 本次命中的裁决条件\n\n"
            markdown += "\n".join(f"- {key}: `{value}`" for key, value in conditions.items())
        if state.llm_assessment is not None:
            review = state.llm_assessment
            markdown = markdown.replace(
                "- 算法测量 + 固定规则；无多模态模型，无生成式补弧；尚未完成 Q1 目录校准。",
                "- 算法测量 + 固定规则票 + 实验性 DeepSeek 物理评审；尚未完成 Q1 目录校准。")
            markdown += "\n\n## DeepSeek 物理假设与裁决（实验性）\n\n"
            markdown += "数值测量 → 透镜/非透镜假设 → 引用及数值核查 → 模型裁决。两阶段使用同一模型，不是两位独立专家。\n\n"
            markdown += "模型未接收图像、目标身份、官方等级或规则票；引用核查不等于物理确认，尚未完成 Q1 标定或透镜方程检验。\n\n"
            markdown += f"- 请求模型：{review['requested_model']}\n- 运行状态：{review['status']}\n"
            if review["status"] == "completed":
                vote = review["assessment"]
                markdown += f"- 模型暂定等级：**{vote['vote_category']}**\n- 裁决：{vote['disposition']}\n"
                markdown += f"- 与规则票一致：{'是' if review['agrees_with_rules'] else '否，需人工复核'}\n"
                markdown += f"\n{vote['rationale']}\n\n### 模型采用的测量证据\n\n"
                for item in vote["supporting_evidence"]:
                    refs = ', '.join(c['observation_ref'] for c in item['measurements'])
                    markdown += f"- {item['statement']}（{refs}）\n"
                markdown += "\n### 模型限制与竞争解释\n\n"
                markdown += "\n".join(f"- {item}" for item in vote['limitations'] + vote['competing_explanations'])
            else:
                markdown += f"- 失败原因：{failure_message(review['error'])}（{review['error']}）；没有可用的模型等级。规则结果保留供查看。\n"
            if review.get("hypotheses"):
                markdown += "\n\n### 已比较的物理假设\n\n"
                for hypothesis in review['hypotheses']['hypotheses']:
                    markdown += f"- {hypothesis['id']}：{hypothesis['configuration']}（{hypothesis['family']}，{hypothesis['status']}）\n"
        state.final_report_markdown = markdown
        state.final_decision = state.to_dict()
        state.final_decision["published_grade"] = published_grade
        state.final_decision["publication_reason"] = publication_reason
        if state.llm_assessment is not None:
            review = state.llm_assessment
            state.final_decision["llm_grade"] = (
                review['assessment']['vote_category'] if review['status'] == 'completed' else None)
            state.final_decision["judgment_mode"] = "rules_and_deepseek_experimental"
        state.final_decision["report_markdown"] = markdown
        self.record_reasoning(
            state,
            "report_generation",
            {"scientific_reasoning_added": False, "audit_passed": audit.audit_passed},
        )
        state.final_decision["reasoning_trace"] = state.reasoning_trace
        return state
