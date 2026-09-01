# LensAgent 当前版本说明

> 当前版本：`0.15.0`
>
> Observation schema（视觉观察文档格式）：`1.13`，未改变
>
> 本次主题：退出未经标定的旧评分，建立 Euclid Q1 评级接口

## 本次改变的原因

旧版为了先跑通完整 Agent，使用人工设定的证据加权、假设分数、分数差和
`confidence.score`。这些数字能够演示流程，但没有使用 Euclid Q1 的
`expert_score` 做标定，因此既不是透镜概率，也不能严谨地对应 A/B/C。

本版删除了活动流程中的旧数值判决链。视觉层保持 v0.14.0 的冻结状态，原始
测量、多假设、支持证据、反证、缺失证据和证据来源继续保留。

## 当前推理流程

```text
Observation（观察事实）
↓
Observation Audit（观察质量审计）
↓
H1–H4 多假设与证据计划
↓
Evidence / Counter-evidence（支持证据与反证）
↓
Cross-hypothesis Comparison（跨假设比较）
↓
Per-hypothesis Case Files（每个假设的独立案卷）
↓
Evidence Audit（证据来源、相关性和区分力审计）
↓
Reasoning Audit（推理结构完整性审计）
↓
Grading Readiness（评级准备状态）
```

在标定工作完成前，程序不会用临时权重生成 A/B/C。

## 评级状态

`summary.json`、`reasoning.json` 和 `report.md` 现在使用统一的 `grading`：

| 状态 | 含义 |
|---|---|
| `calibration_pending` | 数据和证据链可以进入未来评分，但评分器尚未完成标定 |
| `insufficient_data` | 数据本身不可用于科学评级；这不等于 X |
| `blocked_by_audit` | 证据来源或推理结构存在阻断问题，修复前不能评级 |
| `graded` | 只为未来经过标定并通过独立验证的结果保留 |

未评级时：

```json
{
  "grading_scheme": "euclid_q1_paper",
  "status": "calibration_pending",
  "expert_score": null,
  "grade": null,
  "a_plus": null
}
```

这里用 `null` 明确表示“没有评估”，而不是分数为零或等级为 X。

## 唯一保留的等级映射

本版只实现项目选定的 Euclid Q1 论文口径。论文原文明确给出 A 和 B 的分界，
并将低于 B 分界的结果描述为 C 或更低；详见
[Euclid Q1 Strong Lensing Discovery Engine论文](https://arxiv.org/abs/2503.15324)：

```text
A：expert_score > 2.0
B：1.5 < expert_score <= 2.0
C：expert_score <= 1.5
```

映射函数已经通过边界测试，但只有未来经过标定的评分器才能向它提供
`expert_score`。Q1论文没有规定`expert_score <= 1.0`自动映射为X的目录分箱，
因此本项目不发明该边界。程序不再输出第二套 Zenodo 分箱。

## A+ 如何表示

`A+` 不再被当成另一个分数区间，而是 A 级结果上的独立科学价值标记：

- 只能附着在 A 级结果上；
- 必须同时给出科学价值理由；
- 未进行专门审查时为 `null`，不能默认写成 `false`。

## 本版明确没有做什么

- 没有编写评分前的最终量化审计表；
- 没有猜测各项证据权重；
- 没有使用 A/B/C 文件夹反推临时阈值；
- 没有接入 LLM 或 VLM；
- 没有改变 v0.14.0 的视觉检测、椭圆路线或候选四路交接。

下一步需要使用带有 Q1 `expert_score` 的标定数据，建立量化证据表、确定缺失值
语义、校准评分器，并在完全独立的样本上验证后，才允许产生正式等级。
