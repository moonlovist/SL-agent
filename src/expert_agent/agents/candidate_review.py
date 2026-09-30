"""Traceable review of candidate measurements used by integrated grading."""
from __future__ import annotations

from ..schemas.observation import ObservationDocument
from ..schemas.reasoning import CandidateConfigurationReview, TraceableFact


def review_candidate_handoff(document: ObservationDocument):
    bundle = document.candidate_handoff_review
    if bundle is None:
        return [], []
    facts, groups = [], []
    entries = {e.candidate_id: e for e in bundle.entries}
    for entry in bundle.entries:
        measured = entry.measurement
        if measured is None:
            statement = f"{entry.candidate_id}: measurement unavailable; " + "; ".join(entry.uncertainty_notes)
        else:
            statement = (
                f"{entry.candidate_id}: {measured.pixel_count} measured pixels; "
                f"nominal diagonal-RMS aperture SNR={measured.integrated_snr:.3f}; "
                f"angular span={measured.angular_extent_deg:.2f} deg; "
                f"comparison above-threshold pixel fraction={measured.foreground_comparison_survival_fraction}; "
                f"comparison aperture flux={entry.comparison_aperture_flux}; "
                f"comparison detection={entry.comparison_detection}. "
                "Physical origin remains unresolved; grading uses the full morphology and spatial relation, not this measurement alone."
            )
        facts.append(TraceableFact(
            id=f"CRF_{entry.candidate_id}", statement=statement,
            observation_refs=[f"model_candidates.{entry.candidate_id}", f"candidate_handoff_review.entries.{entry.candidate_id}"],
            source_types=["primary_residual_measurement"] if measured is not None else [],
            reliability="unknown",
        ))
    linked = set()
    for path in document.geometric_paths:
        ids = [c.id for c in document.model_candidates if path.id in c.geometry_consistent_path_ids and c.id in entries and entries[c.id].measurement is not None]
        if not ids:
            continue
        linked.update(ids)
        groups.append(CandidateConfigurationReview(
            id=f"CRG_{path.id}", path_id=path.id, candidate_ids=ids,
            unresolved_candidate_ids=[cid for cid in ids if entries[cid].formal_structure_id is None],
            observation_refs=[path.id, *[f"candidate_handoff_review.entries.{cid}" for cid in ids]],
            rationale="These measured candidates share a searched geometric route. This is a review grouping, not an established lens configuration. Overlapping routes and recovered pixels are not independent votes. Local morphology, model sensitivity and non-lens controls still require validation.",
        ))
    # Off-route measurements remain visible individually; they cannot dilute the
    # formal scene statistics or get lost because no path happens to link them.
    for cid, entry in entries.items():
        if cid in linked or entry.measurement is None:
            continue
        groups.append(CandidateConfigurationReview(
            id=f"CRG_{cid}", candidate_ids=[cid],
            unresolved_candidate_ids=[cid] if entry.formal_structure_id is None else [],
            observation_refs=[f"candidate_handoff_review.entries.{cid}"],
            rationale="No measured route group is available for this candidate. Preserve its complete morphology for review without imposing a common circular radius or contributing a lens vote.",
        ))
    return facts, groups


def render_candidate_review_cn(document: ObservationDocument) -> list[str]:
    bundle = document.candidate_handoff_review
    if bundle is None:
        return []
    lines = [
        "", "## 完整候选交接与正式评级", "",
        "下表补齐所有可测候选；统一的像素、信号、切向形态和空间关系直接进入唯一正式评级。"
        "旧图的隔离类别只保留为处理来源记录，不代表候选的物理来源。", "",
        "比较超阈值比例 = 原候选掩膜中，在另一前景扣除残差上达到像素检测阈值的像素数 / 掩膜像素数。"
        "它不是通量保留率，也不是真实概率；0 只表示没有像素超过阈值，不证明整段弧消失。"
        "两个前景模型使用同一张图，不是独立观测。", "",
        "| 候选 | 旧交接类别（留档） | 新测量状态 | 像素 | 名义孔径SNR | 比较超阈值比例 | 比较孔径通量 | 比较检测状态 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    def fmt(value):
        return "未测得" if value is None else f"{value:.4g}"
    statuses = {"formal_reference": "正式结构对照", "measured_for_review": "完整测量待复核", "unmeasurable": "像素测量不可用"}
    detections = {"detected": "有像素超阈值", "below_threshold": "低于像素阈值", "unavailable": "比较不可用"}
    for entry in bundle.entries:
        m = entry.measurement
        lines.append(f"| {entry.candidate_id} | {entry.legacy_handoff_class or '无记录'} | {statuses[entry.status]} | {m.pixel_count if m else '—'} | {fmt(m.integrated_snr if m else None)} | {fmt(m.foreground_comparison_survival_fraction if m else None)} | {fmt(entry.comparison_aperture_flux)} | {detections[entry.comparison_detection]} |")
    if not bundle.entries:
        lines.extend(["", "该历史观测未提供 model-chain 候选，无法据此恢复缺失候选；不能视为已验证完整检出。"])
    for entry in bundle.entries:
        if entry.measurement is None:
            lines.extend(["", f"- {entry.candidate_id} 无法测量：{'；'.join(entry.uncertainty_notes)}"])
    lines.extend(["", "名义孔径SNR仅用输入RMS的对角项计算；未校准像素相关噪声、前景拟合误差和搜索选择效应。"
                  "路线、阈值检测及形态来自同一批像素，不能重复计作独立证据。天空坐标未在回放时重算。"])
    path = document.artifacts.get("candidate_handoff_review")
    if path:
        lines.extend(["", f"![完整候选像素对照](<{path.replace(chr(92), '/')}>)"])
    return lines
