# LensAgent 当前版本说明

> 当前版本：`0.16.0`
>
> Observation schema：`1.13`（视觉层未改变）
>
> 主题：统一专家判断层与轻量独立决策审查

## 活动架构

```text
冻结的视觉层
-> Observation JSON
-> ExpertAssessmentAgent
-> DecisionAudit
-> provisional expert assessment
-> Euclid Q1 论文等级暂定映射
-> JSON / Markdown report
```

`ExpertAssessmentAgent` 是唯一的主要科学判断层。它一次性完成事实选择、信号
真实性与透镜特异性分离、多假设比较、支持与反对证据、缺失信息、处理风险、
相关证据组、整体专家投票锚点和暂定等级判断。

`DecisionAudit` 只检查事实是否来自 Observation、缺失信息是否被误当成反证、
相关证据是否被重复计票、测量是否被写成透镜证明、恢复候选是否被自动惩罚、
替代解释是否遗漏、是否发生外部答案泄漏、文字与分数是否冲突，以及是否应当
弃权。它不重跑多假设分析，也不生成第二套分数。

旧的 `HypothesisAgent`、`EvidenceAgent`、`CounterEvidenceAgent`、
`HypothesisAnalysisAgent`、`EvidenceAuditor`、`ReasoningAuditor` 和
`GradingPreparationAgent` 文件仍保留以便 Git 回退和历史追溯，但已从活动
`ExpertWorkflow` 停用。

## Evidence Dossier

输出包含：

- `observed_facts`：实际参与判断的事实及 measurement、structure 或字段引用；
- `signal_reality`：信号是否像真实天体/残差结构，而不是噪声；
- `lens_specificity`：真实信号是否形成具有透镜区分力的联合空间关系；
- 支持证据、真正冲突的证据和五类竞争解释；
- 缺失信息、处理风险、相关证据组和未使用字段；
- 当前未执行的 null tests 与未来工具请求。

高 S/N、SEP 稳定、多阈值存活和椭圆路线一致只能帮助评价信号或搜索来源，
不能单独证明强透镜。恢复通道只记录发现来源，不自动降级。缺失颜色、反像、
PSF、光谱或质量模型保持 `unknown` / `not_measured` / `unavailable`，不写成
颜色不一致或不存在反像。

## 暂定评分契约

第一版输出：

- `expert_vote_anchor`: `0 / 1 / 2 / 3 / null`；
- `provisional_expert_score`: `float / null`；
- `provisional_score_range`: `[low, high] / null`；
- `provisional_grade`: `X / C / B / A / insufficient_for_grading / null`；
- `score_status`: `provisional_uncalibrated`；
- `calibration_required`: `true`；
- `official_expert_score`: `null`。

锚点对应单个专家的整体视觉判断：X=0、C=1、B=2、A/A+=3。程序不对几何、
颜色、S/N 等子分做加和或平均。C 与 B 之间无法诚实选择时输出 `[1.0, 2.0]`
区间，并让点分和暂定等级保持 `null`。数据或关键测量不可用时输出
`insufficient_for_grading`，不会自动写成 X。

Q1 论文暂定映射为 A：`score > 2.0`，B：`1.5 < score <= 2.0`，C：
`score <= 1.5`。正式映射只允许作用于未来经过标定的聚合 `expert_score`。
A+ 不增加分值；它在当前 schema 中表示独立的 `scientific_value_flags`，没有
可靠依据时保持空列表。

## 视觉层冻结

版本 0.16.0 没有修改 FITS 读取、前景星系建模和减除、SEP 多参数检测、
多 S/N 阈值、圆/椭圆几何路线、路线内补充搜索、结构恢复、候选四路交接或
诊断图。两个验证样本直接读取既有 Observation，不批量重跑视觉层。

## 标定仍需的数据

正式 `official_expert_score` 仍需：

1. 与本项目输入可合法匹配的 Q1 连续聚合 `expert_score`；
2. 每个对象的专家投票数、投票分布和专家偏差校正说明；
3. 独立训练/校准/盲测划分，包含高质量正例和困难负例；
4. 预先冻结的评价指标、弃权策略和误差分析方案。

这些标签只能在当前独立视觉评估之后用于校准和 post-hoc evaluation，不能
反向调整单个样本规则。
