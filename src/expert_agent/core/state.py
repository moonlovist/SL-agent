from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ExpertState:
    """
    Expert Reasoning Agent 的共享状态。

    目标：

    模拟人类专家分析过程。


    流程：

    observation
        ↓
    hypotheses
        ↓
    evidence_plan
        ↓
    evidence
        ↓
    contradiction
        ↓
    confidence
        ↓
    audit
        ↓
    decision
    """


    # =========================
    # 输入信息
    # =========================

    image_path: Optional[str] = None

    object_id: Optional[str] = None



    # =========================
    # 第一阶段：
    # 专家观察到的事实
    # =========================

    observations: List[Dict[str, Any]] = field(
        default_factory=list
    )



    # =========================
    # 第二阶段：
    # 候选解释
    # =========================

    hypotheses: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # =========================
    # 第三阶段：
    # 证据规划
    #
    # EvidencePlannerAgent输出：
    #
    # 哪些证据最值得优先检查
    # 为什么选择这些证据
    # 针对哪些假设
    # =========================

    evidence_plan: Optional[Dict[str, Any]] = None

    # =========================
    # 第四阶段：
    # 支持证据
    # =========================

    evidence: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )



    # =========================
    # 第五阶段：
    # 反证 / 替代解释
    # =========================

    contradictions: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )



    # =========================
    # 第六阶段：
    # 可信度校准
    #
    # ConfidenceAgent输出：
    # - 最终候选解释
    # - 可信度等级
    # - 支持因素
    # - 限制因素
    # - 剩余不确定性
    # =========================

    confidence_assessment: Optional[Dict[str, Any]] = None



    # =========================
    # 第七阶段：
    # 推理审计
    # =========================

    audit_result: Optional[Dict[str, Any]] = None



    # =========================
    # 第八阶段：
    # 最终专家报告
    #
    # ReportWriter最终输出
    # =========================

    final_decision: Optional[Dict[str, Any]] = None



    # =========================
    # 推理历史
    #
    # 保存每个Agent执行记录
    # =========================

    reasoning_trace: List[Dict[str, Any]] = field(
        default_factory=list
    )