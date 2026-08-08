# Lens Expert Agent 当前版本说明

> 当前版本：`0.3.0`  
> Observation schema：`1.1`  
> 阶段主题：分层科学审计与 Knowledge / Skill 分离

## 本版解决了什么问题

v0.2 已经能够让四种解释竞争，但原来的 `ReasoningAuditor` 主要检查“程序有没有按步骤执行、引用是否存在”。它不能充分回答三个科学问题：原始测量是否可靠、证据是否被重复或夸大、最终结论是否应该主动降级。

v0.3 将这三个问题拆成独立角色：

```text
Observation → ObservationAuditor
            → 多假设与证据分析
            → EvidenceAuditor
            → Decision + Confidence
            → DecisionAuditor
            → ReasoningAuditor（结构完整性）
            → Report
```

## 三层科学审计

### ObservationAuditor

检查有限像素比例、噪声估计、前景光减除状态、测量可靠性、结构到测量的引用，以及 RMS/PSF 信息是否缺失。它只判断这些测量能否进入推理，不判断目标是不是透镜。

### EvidenceAuditor

检查证据是否引用真实测量、缺失证据是否保持中性、同源证据是否可能重复计数、低区分力事实是否被过度解释，以及每项事实是否真正跨 H1–H4 比较。

### DecisionAuditor

像一个专门唱反调的复核专家：检查替代解释是否仍在场、上游审计是否通过、选择阈值是否满足、关键证据是否缺失、反证是否尚未解决。输出只有 `confirm`、`downgrade`、`abstain`；它不能提高置信度。

原 `ReasoningAuditor` 继续负责程序结构、schema、引用和审计执行完整性。科学数据不可用但系统正确弃权时，结构审计仍可为真，这表示“推理流程正确”，不是表示“目标已被科学确认”。

## Knowledge / Skill 分离

- `knowledge/`：透镜几何、弧形态、本征星系形态、并合、伪影、颜色与多波段等领域知识；
- `skills/`：Agent 应按什么步骤完成观察审计、证据审计和决策审计；
- Python：负责测量、验证、执行阈值、保存结构化结果。

知识文件不能创造 Observation 中不存在的事实，也不能绕过测量引用。这为以后接入 LLM 留出知识接口，但本版没有接入 LLM、VLM 或 Self-Evolve。

## 输出变化

`reasoning.json` 与 `summary.json` 新增：

- `observation_audit`；
- `evidence_audit`；
- `decision_audit`。

最终 `classification` 使用决策审计后的有效结论。原始 `decision` 和原始 `confidence` 仍完整保留，便于追查“初判是什么、审计为什么改变它”。命令行同时显示决策审计 verdict 和结构审计状态。

## 验证结果

- 自动化测试：8 项全部通过；
- Observation 1.0 兼容推理：通过；
- 不可用 Observation 强制弃权：通过；
- 真实 FITS 端到端回归：通过；
- 回归样本最终为 `strong_lens_candidate`、有效置信度 `medium`、决策审计 `confirm`、结构审计通过。

## 当前仍不代表什么

`strong_lens_candidate` 仍是候选评估，不是经过光谱红移、质量模型或人工专家确认的物理透镜。PNG 仍是保守辅助输入；单波段数据不能产生颜色证据；PSF、多波段和标定数据集仍是后续工作。
