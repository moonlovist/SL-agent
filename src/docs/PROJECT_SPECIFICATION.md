# LensAgent 项目说明 v1.0

## 1. 项目名称

LensAgent：

A Multi-Agent Expert Reasoning System for Strong Gravitational Lens Analysis

---

# 2. 项目目标

开发一个能够模拟天文学专家分析强引力透镜候选目标过程的智能体系统。

核心目标：

不是替代专家直接分类。

而是复现专家：

* 观察；
* 推理；
* 验证；
* 质疑；
* 决策。

---

# 3. 核心思想

传统方法：

```
Image
 ↓
Classifier
 ↓
Lens / Non-lens
```

本项目：

```
Image
 ↓
Observation Agent
 ↓
Hypothesis Agent
 ↓
Evidence Agent
 ↓
Counter Evidence Agent
 ↓
Reasoning Auditor
 ↓
Report Writer
 ↓
Expert-like Decision
```

---

# 4. Agent组成

## ObserverAgent

负责：

模拟专家第一眼观察。

输出：

* 形态；
* 位置；
* 可见结构。

---

## HypothesisAgent

负责：

提出多个可能解释。

避免：

过早判断。

---

## EvidenceAgent

负责：

寻找支持当前假设的证据。

---

## CounterEvidenceAgent

负责：

主动寻找反例。

模拟专家自我怀疑。

---

## ReasoningAuditor

负责：

检查：

* 推理是否合理；
* 是否存在逻辑跳跃；
* 是否忽略反证。

---

## ReportWriter

负责：

生成最终专家报告。

---

# 5. 技术路线

参考：

FORMA 多 Agent 架构思想。

但不复制其结构。

根据任务需求设计。

---

# 6. 当前开发状态

Minimal Working Agent 已完成：

* FITS/PNG 输入与一次预处理、多测量器共享的 VisionPipeline；
* 带版本、测量 ID、可靠性、相关性和 provenance 的 Observation JSON；
* Observation → Hypothesis → Evidence Planning → Evidence/Counter-evidence →
  Decision/Abstain → Confidence → Audit → Report 的完整流程；
* 结论到 measurement ID 的可追溯引用；
* 零证据、平局和非特异证据不能默认选择强透镜；
* CLI、JSON/Markdown 结果落盘、单元测试和真实 FITS 端到端测试。

下一阶段是科学能力扩展与数据集校准，包括多波段颜色、PSF、WCS、并合形态、
hard negatives，以及只能引用现有证据、不能绕过确定性约束的可选 LLM 解释层。

---

# 7. 长期目标

形成一个：
可解释；
可审计；
具有专家推理过程；
能够辅助天文学研究的 Agent 系统。
