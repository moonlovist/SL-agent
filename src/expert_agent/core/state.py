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
    evidence
        ↓
    contradiction
        ↓
    audit
        ↓
    decision
    """

    # =========================
    # 输入
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
    # 支持证据
    # =========================

    evidence: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )


    # =========================
    # 第四阶段：
    # 反证 / 异常
    # =========================

    contradictions: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )


    # =========================
    # 第五阶段：
    # 推理审计
    # =========================

    audit_result: Optional[Dict[str, Any]] = None


    # =========================
    # 最终判断
    # =========================

    final_decision: Optional[Dict[str, Any]] = None


    # =========================
    # 推理历史
    # =========================

    reasoning_trace: List[Dict[str, Any]] = field(
        default_factory=list
    )