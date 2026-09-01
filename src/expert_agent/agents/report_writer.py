from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.state import ExpertState


class ReportWriter(BaseAgent):
    def __init__(self):
        super().__init__("ReportWriter")

    def run(self, state: ExpertState) -> ExpertState:
        self.log("生成不含未标定分数的可审计报告")
        self.load_skill("report_generation_skill")
        document = state.require_observation()
        audit = state.audit_result
        grading = state.grade_assessment
        if audit is None or grading is None:
            raise ValueError("ReportWriter requires reasoning audit and grading status")

        structure_lines = [
            (
                f"- {item.id}: radius={item.radius_px:.2f} px, "
                f"PA={item.position_angle_deg:.1f}°, S/N={item.integrated_snr:.1f}, "
                f"extent={item.angular_extent_deg:.1f}°, "
                f"tangential misalignment={item.tangential_misalignment_deg:.1f}°"
            )
            for item in document.structures
        ] or ["- No candidate passed the structure-promotion audit."]
        candidate_lines = [
            (
                f"- {item.id}: origin={item.origin}, status={item.status}, "
                f"structural support={item.structural_support_level}, "
                f"partition={item.pixel_partition}, review priority={item.review_priority}, "
                f"parent SEP={item.parent_sep_component_id or 'none'}, "
                f"radius={item.radius_px:.2f} px, pixels={item.pixel_count}, "
                f"integrated S/N={item.integrated_snr:.1f}, "
                f"paths={','.join(item.path_ids) or 'none'}"
            )
            for item in document.model_candidates
        ] or ["- No model-chain candidate was retained."]
        candidate_audit_lines = [
            (
                f"- {item.id}: candidate={item.candidate_id}, verdict={item.verdict}, "
                f"structure={item.structure_id or 'none'}; {item.rationale}"
            )
            for item in document.candidate_structure_audits
        ] or ["- No candidate-structure audit was recorded."]
        system_lines = [
            (
                f"- {item.id}: path={item.path_id}, "
                f"members={','.join(item.member_candidate_ids)}, "
                f"SEP members={','.join(item.sep_candidate_ids) or 'none'}, "
                f"recovered members={','.join(item.recovered_candidate_ids) or 'none'}, "
                f"components={item.component_count}; candidate grouping only"
            )
            for item in document.candidate_systems
        ] or ["- No route-associated candidate system was assembled."]
        path_lines = [
            (
                f"- {item.id}: mode={item.mode}, "
                f"seeds={','.join(item.seed_candidate_ids) or 'none'}, "
                f"linked={','.join(item.linked_candidate_ids) or 'none'}, "
                f"recovered={','.join(item.recovered_candidate_ids) or 'none'}; "
                "search route only, not a ring claim"
            )
            for item in document.geometric_paths
        ] or ["- No geometry-guided search path was retained."]

        hypothesis_lines: list[str] = []
        for hypothesis in state.hypotheses:
            analysis = state.hypothesis_analyses[hypothesis.id]
            missing = (
                "; ".join(item.question_id for item in analysis.missing_evidence)
                or "none"
            )
            hypothesis_lines.append(
                f"- {hypothesis.id} `{hypothesis.name}`: status={analysis.status}; "
                f"support={','.join(analysis.supporting_evidence_refs) or 'none'}; "
                f"opposition={','.join(analysis.opposing_evidence_refs) or 'none'}; "
                f"missing={missing}"
            )
        comparison_lines = [
            (
                f"- {item.id}: {item.observed_fact} | "
                + "; ".join(
                    f"{effect.hypothesis_id}={effect.effect}" for effect in item.effects
                )
                + f" | discriminating={item.discriminating}"
            )
            for item in state.evidence_comparisons
        ] or ["- No cross-hypothesis comparison was available."]
        lens_support_lines = [
            f"- {item.id}: {item.claim} (refs: {', '.join(item.observation_refs)})"
            for item in state.evidence.get("H1", [])
            if item.status == "evaluated" and item.polarity == "support"
        ] or ["- No evaluated support item is currently recorded for H1."]
        contradiction_lines = [
            f"- {item.id} [{item.hypothesis_id}]: {item.description} "
            f"(refs: {', '.join(item.observation_refs)})"
            for items in state.contradictions.values()
            for item in items
        ] or ["- No explicit contradiction was identified."]
        limitation_lines = [f"- {item}" for item in document.limitations] or [
            "- No limitation was recorded."
        ]
        missing_grading_lines = [
            f"- {item}" for item in grading.missing_requirements
        ] or ["- None"]
        trace_labels = {
            "overview_diagnostic": "overview diagnostic",
            "snr_threshold_process": "S/N threshold process",
            "sep_consensus_diagnostic": "SEP consensus process",
            "model_chain_diagnostic": "SEP-to-geometry model-chain process",
            "expert_views_diagnostic": "Euclid-style VIS expert views",
            "foreground_model_comparison_diagnostic": "foreground-model comparison",
            "geometric_path_search_diagnostic": "geometry-guided search",
            "field_context_diagnostic": "full-field location audit",
            "rgb_context_diagnostic": "RGB-to-FITS location audit",
            "vision_manifest": "trace manifest",
            "vision_arrays": "numerical trace arrays",
        }
        trace_lines = [
            f"- {trace_labels[key]}: `{document.artifacts[key]}`"
            for key in trace_labels
            if key in document.artifacts
        ] or ["- No visual trace bundle was written for this run."]

        score_text = (
            f"{grading.expert_score:.3f}" if grading.expert_score is not None else "not issued"
        )
        grade_text = grading.grade or "not issued"
        a_plus_text = (
            "yes" if grading.a_plus is True else "no" if grading.a_plus is False else "not evaluated"
        )
        markdown = "\n".join(
            [
                f"# Strong-lens evidence report: {document.object_id}",
                "",
                "## Data reliability",
                "",
                f"- Usable: {document.data_quality.usable}",
                f"- Band: {document.band}",
                f"- Foreground model: {document.data_quality.model_method} "
                f"(success={document.data_quality.model_success})",
                f"- Supplied RMS map: {document.data_quality.supplied_rms}",
                f"- Observation audit: {state.observation_audit.quality_level} "
                f"(valid={state.observation_audit.valid})",
                "",
                "## Vision processing trace",
                "",
                *trace_lines,
                "",
                "## Candidate-structure handoff audit",
                "",
                *candidate_audit_lines,
                "",
                "## Audited structured observations",
                "",
                *structure_lines,
                "",
                "## Model-chain candidates",
                "",
                *candidate_lines,
                "",
                "## Route-associated candidate groups",
                "",
                *system_lines,
                "",
                "## Geometry-guided search routes",
                "",
                *path_lines,
                "",
                "## Competing hypothesis case files",
                "",
                *hypothesis_lines,
                "",
                "## Cross-hypothesis evidence comparison",
                "",
                *comparison_lines,
                "",
                "## Evidence supporting the strong-lens hypothesis",
                "",
                *lens_support_lines,
                "",
                "## Explicit contradictions",
                "",
                *contradiction_lines,
                "",
                "## Q1 grading status",
                "",
                f"- Scheme: `{grading.grading_scheme}`",
                f"- Status: **{grading.status}**",
                f"- expert_score: **{score_text}**",
                f"- Grade: **{grade_text}**",
                f"- A+ scientific-value flag: **{a_plus_text}**",
                f"- Rationale: {grading.rationale}",
                "- Missing requirements:",
                *missing_grading_lines,
                "",
                "## Remaining limitations",
                "",
                *limitation_lines,
                "",
                "No provisional numerical score, confidence value, or A/B/C label is "
                "issued before calibration. This is an evidence report, not a lens probability.",
            ]
        )
        state.final_report_markdown = markdown
        state.final_decision = {
            "object_id": document.object_id,
            "grading": grading.model_dump(mode="json"),
            "hypothesis_analyses": {
                key: value.model_dump(mode="json")
                for key, value in state.hypothesis_analyses.items()
            },
            "evidence_comparisons": [
                item.model_dump(mode="json") for item in state.evidence_comparisons
            ],
            "observation_audit": state.observation_audit.model_dump(mode="json"),
            "evidence_audit": state.evidence_audit.model_dump(mode="json"),
            "audit": audit.model_dump(mode="json"),
            "report_markdown": markdown,
        }
        self.record_reasoning(
            state,
            "report_generation",
            {
                "grading_status": grading.status,
                "grade": grading.grade,
                "expert_score": grading.expert_score,
                "audit_valid": audit.valid,
            },
        )
        return state
