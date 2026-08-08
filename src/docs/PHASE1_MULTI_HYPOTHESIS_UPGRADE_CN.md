# LensAgent v0.2 第一阶段升级记录

## 目标

本阶段吸收 FORMA 的多假设竞争思想，但不接入 LLM/VLM，不引入自动
Self-Evolve，也不改变视觉层与推理层的边界。

核心变化是把：

```text
Observation → 面向强透镜的证据打分
```

升级为：

```text
Observation
→ 四种竞争假设
→ 同一事实对 H1-H4 的影响比较
→ 每个假设的独立案卷
→ 假设综合、弃权、可信度和审计
```

## Observation 1.1

新增三组结构化观察：

1. `foreground_galaxy`
   - 中心位置、峰值、SEP 积分通量、椭率、位置角和轴比；
   - 仅描述中心光分布，不判断其是否为透镜星系。
2. `scene_geometry`
   - 结构数量、共同半径变异系数、总环向覆盖、平均切向偏差、
     切向对齐比例。
3. `alternative_indicators`
   - 与内区连接的结构数、次级核心数、180° 残差不对称、
     核心残差比例、坏像素比例和 PSF 信息可用性。

替代解释指标被明确标记为 screening proxy。特别是：

- 高残差不对称可能由真实透镜弧本身产生，不能独立支持并合；
- 多核心只在减除敏感的中心尺度内筛选，避免把环带弧误认为核心；
- 没有 PSF 模型时只记录“无法检查”，不能据此确认或排除伪影。

旧的 Observation 1.0 仍可由 `reason` 命令读取，因为新字段均为可选字段。

## 新推理阶段

### EvidenceComparisonAgent

把引用同一组测量的证据合并为一行，对 H1-H4 分别记录：

- `support`；
- `oppose`；
- `neutral`；
- `unknown`。

每行保留测量引用、证据引用、相关组和是否具有区分力。

### HypothesisAnalysisAgent

为每个假设生成独立案卷：

- 支持证据；
- 反对证据；
- 缺失证据；
- 当前不确定性；
- 已解释的观察；
- 尚未解释的相关观察；
- `viable`、`weakened` 或 `insufficient` 暂态。

这些暂态不是最终分类。最终结果仍由 DecisionAgent 在比较全部假设后产生。

## 新工作流

```text
ObserverAgent
→ HypothesisAgent
→ EvidencePlannerAgent
→ EvidenceAgent
→ EvidenceComparisonAgent
→ CounterEvidenceAgent
→ HypothesisAnalysisAgent
→ DecisionAgent
→ ConfidenceAgent
→ ReasoningAuditor
→ ReportWriter
```

## 审计增强

当前 ReasoningAuditor 新增检查：

- 跨假设比较是否存在；
- 每条比较是否引用真实测量；
- 每条比较是否覆盖全部四种假设；
- 是否为每个假设生成完整案卷。

分层的 ObservationAuditor、EvidenceAuditor 与 DecisionAuditor 属于第二阶段，
本阶段没有提前把它们混入一个大规模重构。

## 输出兼容性

原有字段继续保留。新增内容出现在：

- `observation.json`：三组 Observation 1.1 字段；
- `reasoning.json`：`evidence_comparisons` 与 `hypothesis_analyses`；
- `summary.json`：新增同名摘要字段；
- `report.md`：新增逐假设分析与跨假设证据比较章节。

CLI 命令和原有六类输出文件名不变。

## 验证

- 自动测试覆盖四份假设案卷、缺失证据和四假设比较矩阵；
- 真实 FITS 样本完成端到端运行；
- 回归样本输出 `strong_lens_candidate`、中等可信度且审计通过；
- 代理指标误把候选弧视为次级核心的问题已通过收紧核心范围修正。
