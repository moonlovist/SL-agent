# LensAgent 当前版本说明

> 当前版本：`0.30.0` 可交付快照，Observation `1.18`。安装、入口、包含范围与已知限制见 [v0.30 交付说明](DELIVERY_V0300_CN.md)。桌面跑批、结果浏览、人工闭环与独立干扰区 mask 见 [v0.29修订说明](SCIENTIFIC_REVISION_V0290_UI_CN.md)；人工选择后的拟合质量门仍以 [v0.28修订说明](SCIENTIFIC_REVISION_V0280_CN.md) 为准。下文旧版本段落留作历史说明。
>
> Observation schema：`1.18`（在 1.17 基础上增加输入径向对比度、宽度/PSF 尺度与可观测性诊断字段）
>
> 主题：自动单目标路由加可追溯人工覆盖、连续闭合极小值边界、PSF卷积前景模型库；仍只输出一个对象的一套结果和单专家类别

## v0.19.0 历史变更

本版修改了视觉测量与后端判断，不能再套用下文 v0.18.0 的“底层测量不变”边界。
默认以切图中心为目标位置参考，限制前景中心选择，歧义时停止评级；支持显式中心坐标。
增加真实掩膜的角向占据、径向宽度和局部切向测量；独立模型存活率按参与构型的结构
分别检查，不再让无关结构的混合比例否决强弧。低混合比例仍记录为处理风险。
弱但可测、具有几何关系且在比较模型中存活的待复核候选支持 C，不冒充正式结构或 A。

完整规则、边界与回归结果见 [v0.19.0 修改说明](V0190_CENTER_STRUCTURE_CHANGE_CN.md)。

## 活动架构

```text
底层视觉测量（目标中心选择、前景减除、SEP、S/N阈值与几何测量）
-> 候选信号真实性审计（完整交接可信结构）
-> Observation JSON
-> ExpertAssessmentAgent
-> DecisionAudit
  -> one categorical X/C/B/A vote
-> JSON / Markdown report
```

`ExpertAssessmentAgent` 是唯一的主要科学判断层。它一次性完成事实选择、信号
真实性与透镜特异性分离、多假设比较、支持与反对证据、缺失信息、处理风险、
相关证据组、多类透镜构型假设和一个专家类别。

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
- `configuration_hypotheses`：环/部分环、切向弧、折叠弧、尖点弧、径向弧、
  紧致多像、非对称多像和单像假设。

高 S/N、SEP 稳定、多阈值存活和椭圆路线一致只能帮助评价信号或搜索来源，
不能单独证明强透镜。恢复通道只记录发现来源，不自动降级。缺失颜色、反像、
PSF、光谱或质量模型保持 `unknown` / `not_measured` / `unavailable`，不写成
颜色不一致或不存在反像。

## 暂定投票契约

当前输出：

- `vote_category`: `X / C / B / A / insufficient_for_grading / null`；
- `plausible_vote_categories`: 仅包含本次单专家选择的一个类别；
- `vote_code`: `null`（兼容字段，不再生成机械数字）；
- `expert_vote_anchor`: `null`；
- `provisional_expert_score`: `null`；
- `provisional_score_range`: `null`；
- `provisional_grade`: `X / C / B / A / insufficient_for_grading / null`；
- `score_status`: `provisional_uncalibrated`；
- `calibration_required`: `true`；
- `official_expert_score`: `null`。

`vote_category` 是主输出。一个 Agent 模拟一个专家，只选择 X/C/B/A，不制造
数字分、区间或 `B-A` 伪类别。旧数字字段仅为 schema 兼容而保留为空。
程序不对几何、颜色、S/N 等子分做加和或平均。数据或关键测量不可用时输出
`insufficient_for_grading`，不会自动写成 X。

Q1 论文明确的 catalogue 分界为 A：`score > 2.0`，B：
`1.5 < score <= 2.0`；§8.3 给出 C：`1.0 < score <= 1.5`。
`score <= 1.0` 记为 `below_C_cut`，不冒充官方 X 汇总界限。正式映射只允许作用于未来经过标定的聚合
`expert_score`。
A+ 不增加分值；它在当前 schema 中表示独立的 `scientific_value_flags`，没有
可靠依据时保持空列表。

## v0.18.0 结构交接与构型判断升级

- 稳定、连通、正信号的路线外候选不再因缺少椭圆路线而被挡住；它以
  `recovery_candidate` 来源生成完整正式结构并进入规则层。
- 接近减除核心或主前景模型失败只增加风险；若独立比较模型仍保留信号，
  不再标红隔离。只有比较模型明确擦除候选时才标记前景污染。
- A/B/C/X 按系统级构型判断，不再依赖统一的“结构数、切向角、60度弧长”组合。
- 显式评估第21页所示相关构型：径向弧、折叠弧、尖点弧、紧致多像/爱因斯坦
  十字，以及切向弧、部分环和高度不对称系统。
- 稳健的单条或部分切向弧可以支持 A；短弧单独不足时仍为 B，但折叠/尖点或
  其他结构关系可使短弧系统达到 A。
- 内连数、潮汐不对称和核心残差为相关代理；不足以单独压过清晰、跨模型存活
  的透镜构型。
- 单专家只输出一个类别，所有未校准数字字段保持 `null`。

## 视觉层变更边界

版本 0.18.0 没有修改 FITS 读取、前景星系建模和减除、SEP 多参数检测、
多 S/N 阈值、圆/椭圆几何路线、路线内补充搜索或像素恢复。修改范围仅为候选
信号审计后的交接、交接图说明和后端构型判断。两个问题样本重新完整运行，
验证青色长弧进入正式结构、被误标红的独立存活结构得到恢复。

## 标定仍需的数据

正式 `official_expert_score` 仍需：

1. 与本项目输入可合法匹配的 Q1 连续聚合 `expert_score`；
2. 每个对象的专家投票数、投票分布和专家偏差校正说明；
3. 独立训练/校准/盲测划分，包含高质量正例和困难负例；
4. 预先冻结的评价指标、弃权策略和误差分析方案。

标签不能作为单对象输入或产生 ID 特判。开发集可用于发现总体门控偏差，但仍需
困难负例和独立盲测集评估放宽后可能增加的假阳性。
