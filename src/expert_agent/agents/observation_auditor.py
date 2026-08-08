from __future__ import annotations

from ..core.base_agent import BaseAgent
from ..core.config import ReasoningConfig
from ..core.state import ExpertState
from ..schemas.reasoning import AuditFinding, ObservationAuditResult


class ObservationAuditor(BaseAgent):
    """Assess whether measurements are usable without interpreting them as a lens."""

    def __init__(self, config: ReasoningConfig | None = None):
        super().__init__("ObservationAuditor")
        self.config = config or ReasoningConfig()

    def run(self, state: ExpertState) -> ExpertState:
        self.log("审计观测质量与测量可靠性")
        self.load_skill("observation_audit_skill")
        self.load_knowledge_bundle(["arc_morphology", "artifacts_subtraction"])
        document = state.require_observation()
        findings: list[AuditFinding] = []

        def add(fid: str, severity: str, category: str, description: str, action: str,
                refs: list[str] | None = None) -> None:
            findings.append(AuditFinding(
                id=fid, severity=severity, category=category, description=description,
                observation_refs=refs or [], recommended_action=action,
            ))

        finite_ok = document.data_quality.finite_fraction >= self.config.observation_min_finite_fraction
        if not finite_ok:
            add("OA-FINITE", "critical", "data_integrity", "The finite-pixel fraction is below the audit threshold.", "Repair or mask invalid pixels before scientific interpretation.")
        noise_ok = document.data_quality.background_rms > 0 and document.data_quality.residual_robust_rms > 0
        if not noise_ok:
            add("OA-NOISE", "critical", "noise_model", "A positive background and residual noise estimate is not available.", "Re-estimate the noise model.")
        if not document.data_quality.supplied_rms:
            add("OA-RMS", "warning", "noise_model", "No supplied RMS map was used; noise was estimated from the image.", "Treat marginal S/N structures cautiously.")
        model_ok = document.data_quality.model_success
        if not model_ok:
            add("OA-MODEL", "warning", "foreground_subtraction", "Foreground-light modelling did not report success.", "Inspect the original and residual images together.")
        low_reliability = [m.id for m in document.measurements if m.reliability < self.config.observation_min_reliability]
        if low_reliability:
            add("OA-RELIABILITY", "warning", "measurement_reliability", "Some measurements have very low reported reliability.", "Do not let these measurements drive a decision.", low_reliability)
        measurement_ids = {m.id for m in document.measurements}
        structure_refs_ok = all(set(s.measurement_ids).issubset(measurement_ids) for s in document.structures)
        if not structure_refs_ok:
            add("OA-PROVENANCE", "critical", "provenance", "A structure references an unknown measurement.", "Regenerate the Observation document with intact provenance.")
        psf_available = bool(document.alternative_indicators and document.alternative_indicators.psf_information_available)
        if not psf_available:
            add("OA-PSF", "warning", "instrument_context", "PSF information is unavailable.", "Keep PSF-shaped residuals as an unresolved artifact explanation.")

        critical = any(item.severity == "critical" for item in findings)
        warning = any(item.severity == "warning" for item in findings)
        valid = document.data_quality.usable and not critical
        quality = "usable" if valid and not warning else "limited" if valid else "unusable"
        checks = {
            "data_marked_usable": document.data_quality.usable,
            "finite_pixel_fraction": finite_ok,
            "positive_noise_estimate": noise_ok,
            "foreground_model_success": model_ok,
            "structure_provenance": structure_refs_ok,
        }
        state.observation_audit = ObservationAuditResult(
            valid=valid,
            quality_level=quality,
            findings=findings,
            checks=checks,
            reliability_summary=f"Observation quality is {quality}; {len(findings)} audit finding(s) were recorded.",
            recommendation=("Measurements may enter reasoning with the stated limitations." if valid else "Do not issue a scientific classification; retain only an indeterminate result."),
        )
        self.record_reasoning(state, "observation_quality_audit", {
            **state.observation_audit.model_dump(mode="json"),
            "knowledge_topics": ["arc_morphology", "artifacts_subtraction"],
        })
        return state
