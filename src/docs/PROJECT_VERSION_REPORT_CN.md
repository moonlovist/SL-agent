# 强引力透镜专家推理 Agent 项目版本报告
## 项目简介
本项目目标是设计一个能够模拟天文学专家分析过程的智能 Agent 系统。
不同于传统图像分类任务，本项目关注：

- 专家如何观察目标；
- 如何提出多个解释；
- 如何寻找支持证据；
- 如何考虑反例；
- 如何调整可信度；
- 如何形成最终判断。

项目核心目标：
> 构建一个能够记录专家推理过程，而不仅仅输出分类结果的强引力透镜分析 Agent。
---

# Version 1.0

## 基础专家 Agent 框架


## 版本目标

v1.0阶段主要目标：

建立强引力透镜专家分析系统的基础架构。


主要思想：

将传统：

```
输入图像
    ↓
直接分类
    ↓
输出结果
```

转变为：

```
观察
 ↓
假设
 ↓
证据
 ↓
判断
```

的专家式分析流程。


---

# v1.0主要完成内容


## 1. 建立Agent架构


创建基础 Agent 模块：

包括：

- ObserverAgent
- HypothesisAgent
- EvidenceAgent
- CounterEvidenceAgent
- ReasoningAuditor
- ReportWriter


每个 Agent 负责一个专家分析阶段。


---

## 2. 建立共享状态系统


创建：

ExpertState


用于保存：

- 观察结果；
- 候选假设；
- 证据；
- 反证；
- 推理记录；
- 最终结果。


实现多个 Agent 之间的信息传递。


---

## 3. 建立专家推理流程


初始流程：

```
Observation

↓

Hypothesis Generation

↓

Evidence Evaluation

↓

Counter Evidence

↓

Reasoning Audit

↓

Report Generation
```


---

# v1.0存在的问题


随着测试和设计深入，发现初始架构存在一些限制。


## 1. 假设生成容易变成固定枚举


早期方式：

```
观察

↓

生成H1-H4

↓

分别检查
```


问题：

- 所有假设被同等处理；
- 容易产生大量无效分析；
- 不符合专家实际判断过程。


---

## 2. 缺少证据优先级


系统知道：

“需要证据”

但是不知道：

“哪些证据最值得检查”。


---

## 3. 缺少明确的不确定性表达


专家判断不仅需要结果，还需要：

- 可信程度；
- 支持原因；
- 限制因素。


---

# Version 1.1

## 专家推理规则增强


## 版本目标

v1.1主要引入：

- Skill规则系统；
- Reasoning Trace；
- 更清晰的Agent职责。


---

# v1.1主要完成内容


## 1. 引入Skill系统


将专家知识从代码中分离。


例如：

- hypothesis_skill
- evidence_evaluation_skill
- confidence_calibration_skill


使 Agent 可以根据专业规则执行推理。


---

## 2. 增加推理轨迹记录


系统开始保存：

Agent：

- 做了什么；
- 为什么做；
- 使用了什么规则。


提高系统可解释性。


---

## 3. 优化Hypothesis设计


开始强调：

假设不是答案。

而是：

> 当前阶段需要验证的可能解释。


---

---

# V2.0 Expert Reasoning Framework Upgrade

## 版本目标

V2.0 对原有基础 Agent 流程进行重新设计。

核心目标：

从简单的任务执行流程，

升级为模拟专家分析过程的推理框架。


核心原则：

Observation does not equal conclusion.

观察结果不直接等于最终判断。

系统需要：

- 保留多个候选解释；
- 分析支持证据；
- 寻找反例；
- 评估判断可信程度。


---

## 推理流程升级


V2.0 新增完整专家推理链：


Observation

↓

Hypothesis Generation

↓

Evidence Evaluation

↓

Contradiction Analysis

↓

Confidence Calibration

↓

Reasoning Audit

↓

Final Report


---

## 主要改进


### 1. Hypothesis Generation升级

新增竞争假设机制。

系统不直接判断目标是否为强引力透镜，而是同时维护多个可能解释：

- strong gravitational lens;
- intrinsic galaxy structure;
- galaxy merger;
- imaging artifact.


每个假设包含：

- supporting observations;
- required evidence;
- possible contradictions。


---

### 2. Evidence Evaluation升级

EvidenceAgent不再直接输出结论。

改为：

根据候选假设生成需要验证的证据。


每个证据记录：

- 关联假设；
- 需要验证的问题；
- 当前评估状态；
- 不确定性。


---

### 3. Counter Evidence机制

新增反证分析流程。

目标：

降低确认偏差。


系统主动寻找：

- 哪些观察可能削弱当前假设；
- 哪些替代解释仍然成立。


---

### 4. Confidence Calibration

新增可信度校准阶段。


区别：

判断：

“哪个解释更合理”

与

相信程度：

“这个判断有多可靠”。


系统记录：

- confidence level;
- supporting factors;
- limiting factors;
- remaining uncertainties。


---

### 5. Reasoning Audit

新增推理审计阶段。


检查：

- 是否生成候选假设；
- 是否分析证据；
- 是否考虑反证；
- 是否记录不确定性。


---

# V2.1 Evidence Planning Upgrade

## 版本目标

解决 V2.0 中存在的问题：

所有候选假设进行完整枚举验证，

导致：

- 验证效率低；
- 无效推理增加；
- 不符合专家实际分析流程。


---

## 核心改进

从：

全部证据检查


升级为：

优先验证高价值证据。


---

## 新增组件


### EvidencePlannerAgent

新增证据规划阶段。


流程升级：

Hypothesis

↓

Evidence Planning

↓

Evidence Evaluation


负责：

- 选择优先验证的问题；
- 根据假设竞争关系规划证据；
- 减少无效检查。


---

## 新增Skill

新增：

evidence_planning_skill


用于定义：

- 证据优先级；
- 验证目标；
- 信息价值。


---

## State升级

新增：

evidence_plan


用于保存：

- 优先验证问题；
- 目标假设；
- 规划原因。


---

## Agent升级


### EvidenceAgent

升级：

从读取全部证据需求，

变为读取 EvidencePlanner 生成的验证计划。


新增：

planned_evidence。


---

### CounterEvidenceAgent

升级：

从生成全部可能反例，

变为针对关键验证问题生成风险检查。


新增：

planned_contradiction。


---

### ConfidenceAgent

升级：

不再固定选择第一个假设。

增加基础候选比较机制。


---

### ReasoningAuditor

新增：

检查证据规划是否执行。


---

### ReportWriter

新增输出：

evidence_plan。


---

# V2.2 Evidence Scoring Framework

## 版本目标

解决 V2.1 中的问题：

系统能够选择验证什么，

但还不能评价证据强弱。


---

## 核心改进

从：

证据存在性判断


升级为：

证据权重评价框架。


---

## 新增Skill

新增：

evidence_scoring_skill


定义：

- evidence strength;
- reliability;
- specificity;
- support score。


为未来接入视觉模型和LLM评分提供接口。


---

## EvidenceAgent升级


证据结构增加：

- strength;
- reliability;
- specificity;
- support_score;
- evaluation_reason。


使证据由：

“待验证项目”

升级为：

“可评分知识单元”。


---

## ConfidenceAgent升级


评分方式由：

support_count - contradiction_count


升级为：

基于证据权重的评分框架。


新增：

- hypothesis_scores;
- evidence_score_breakdown。


用于记录：

- 支持证据贡献；
- 反证影响；
- 假设综合评分。


---

## ReasoningAuditor升级


新增检查：

- 证据是否包含评分结构；
- 可信度判断是否使用证据评分。


---

## ReportWriter升级


最终报告新增：

evidence_scoring


用于展示：

- 证据权重；
- 假设评分；
- 可信度依据。


---

# 当前版本状态

当前版本：

V2.2


当前专家推理流程：

Observation

↓

Hypothesis Generation

↓

Evidence Planning

↓

Evidence Evaluation

↓

Evidence Scoring

↓

Contradiction Analysis

↓

Confidence Calibration

↓

Reasoning Audit

↓

Expert Report


---

# V2.3 Decision Separation

## 版本目标

解决 V2.2 中存在的职责耦合问题：

ConfidenceAgent 同时承担：

- 选择最佳假设；
- 判断可信程度。


这种设计导致：

“选择什么解释”

和

“相信这个解释多少”

两个不同的专家任务混合。


V2.3 将两个过程进行拆分，使推理流程更加符合真实专家分析过程。


---

# 核心改进


## Decision 与 Confidence 分离


原流程：

Evidence

↓

ConfidenceAgent

↓

Final judgement


其中 ConfidenceAgent 同时负责：

- hypothesis selection；
- confidence evaluation。



升级后：

Evidence

↓

DecisionAgent

↓

ConfidenceAgent

↓

Final judgement



职责变为：


### DecisionAgent

回答：

“当前哪个解释最符合已有证据？”


负责：

- 比较候选假设；
- 综合证据评分；
- 考虑反证影响；
- 输出假设排序。


不负责：

- 可信度等级；
- 不确定性分析。



---

### ConfidenceAgent

回答：

“对于当前选择，我应该有多大信心？”


负责：

- confidence level；
- uncertainty；
- supporting factors；
- limiting factors。


不再负责：

- 选择最终假设。


---

# 新增组件


## DecisionAgent


新增：

`DecisionAgent`


作用：

根据：

- evidence scoring；
- contradiction analysis；

选择当前最佳解释。


输出：

- selected hypothesis；
- hypothesis ranking；
- decision reasoning；
- remaining competitors。



---

# State升级


新增：

```text
decision_result
```


用于保存 DecisionAgent 输出。


新的状态流程：


Observation

↓

Hypothesis

↓

Evidence Planning

↓

Evidence Evaluation

↓

Contradiction Analysis

↓

Decision

↓

Confidence

↓

Audit

↓

Final Report



---

# Workflow升级


ExpertWorkflow 新增：

DecisionAgent。


流程由：

Evidence

↓

Counter Evidence

↓

Confidence


升级为：


Evidence

↓

Counter Evidence

↓

Decision

↓

Confidence



---

# ConfidenceAgent升级


主要变化：

删除：

- 根据 hypothesis_scores 自动选择最佳假设；
- 使用 max score 直接决定最终解释。


新增：

读取：

```text
decision_result
```


由 DecisionAgent 提供：

当前最佳解释。


ConfidenceAgent 专注：

可信度评估。


---

# ReasoningAuditor升级


新增检查：


## Decision检查

验证：

- 是否完成最佳解释选择；
- 是否存在 selected_hypothesis；
- 是否存在 hypothesis ranking。


## Confidence依赖检查

验证：

Confidence阶段是否基于 Decision 结果。


---

# ReportWriter升级


最终报告新增：

```text
decision_result
```


使最终报告能够完整记录：

- 为什么选择该假设；
- 候选假设如何排序；
- 可信度如何评估。


---

# V2.3意义


V2.3 的核心不是增加新的分析能力，

而是优化专家推理结构。


系统从：

“一个 Agent 同时负责选择和判断”


升级为：

“多个专家角色分别负责不同认知任务”。


形成更加清晰的专家决策链：


观察事实

↓

提出解释

↓

规划验证

↓

评估证据

↓

分析反例

↓

选择最佳解释

↓

评估可信程度

↓

审计推理

↓

生成报告



---

# 当前版本状态

当前版本：

V2.3


当前 Expert Reasoning Framework：


Observation

↓

Hypothesis Generation

↓

Evidence Planning

↓

Evidence Evaluation

↓

Evidence Scoring

↓

Contradiction Analysis

↓

Decision Making

↓

Confidence Calibration

↓

Reasoning Audit

↓

Final Report


---

# 下一阶段规划


## V2.4 LLM Integration Preparation


目标：

为接入真实模型能力准备接口。


主要方向：

- Agent Prompt结构设计；
- LLM调用层；
- Skill与模型推理结合；
- 输入输出格式标准化。


当前系统已经完成：

专家推理流程骨架。


下一阶段将逐步替换：

模拟专家逻辑

↓

模型驱动专家推理。
```

---

更新完后可以 Git：

建议 commit：

```text
V2.3 Decision Separation completed
```

或者：

```text
V2.3：决策与可信度分离完成
```

---

# Lens Expert Agent v0.2.0（2026-08-08）

> 说明：上文的 V1.0–V2.3 是早期 Expert Reasoning Framework 的内部演进编号；
> 从可安装、可运行的 `lens-expert-agent` 开始，项目统一采用语义化版本号。
> 本节对应软件包版本 `0.2.0` 和 Observation schema `1.1`。

## 本次目标

吸收 FORMA 的多假设竞争思想，把现有流程从“围绕强透镜收集证据”升级为：

```text
观察事实
→ 四种解释分别分析
→ 同一事实跨假设比较
→ 综合选择或主动弃权
```

本版本不接入 LLM、VLM 或自动 Self-Evolve，继续保持视觉测量与专家推理分离。

## Observation 1.1

新增：

- 中心前景光分布的位置、峰值、通量、椭率、位置角和轴比；
- 整体共同半径、环向覆盖、平均切向偏差和切向对齐比例；
- 结构与内区连接性；
- 次级核心筛查；
- 180° 残差不对称代理；
- 核心残差比例、坏像素比例和 PSF 信息可用性。

所有替代解释指标都保持为 Observation 或 screening proxy，不直接输出透镜、
并合或伪影结论。

## 多假设推理

新增 `EvidenceComparisonAgent`：

- 将同一组测量对 H1–H4 的影响并排记录；
- 区分支持、反对、中性和未知；
- 标记该事实是否真正具有跨假设区分力。

新增 `HypothesisAnalysisAgent`：

- 为每个假设建立独立案卷；
- 记录支持证据、反对证据、缺失证据和不确定性；
- 记录已解释及尚未解释的相关观察；
- 产生 `viable`、`weakened` 或 `insufficient` 暂态，供最终综合使用。

## 新工作流

```text
Observer
→ Hypothesis Generation
→ Evidence Planning
→ Evidence Evaluation
→ Cross-Hypothesis Evidence Comparison
→ Counter Evidence
→ Per-Hypothesis Analysis
→ Hypothesis Synthesis / Decision
→ Confidence
→ Reasoning Audit
→ Report
```

## 审计与兼容性

- 审计器新增跨假设矩阵覆盖检查；
- 审计器检查每个假设是否拥有完整案卷；
- 原有 CLI 和六类输出文件名保持不变；
- Observation 1.0 仍可由 `lens-agent reason` 读取；
- 新字段写入 `observation.json`、`reasoning.json`、`summary.json` 和 `report.md`。

## 验证

- 自动测试：6 项全部通过；
- Python 依赖检查：无损坏项；
- 旧 Observation 1.0 兼容推理通过；
- 真实 FITS 端到端回归通过；
- 回归样本结果为 `strong_lens_candidate`、中等可信度、审计通过；
- 软件源码和安装元数据均为 `0.2.0`。

## 下一阶段

- 拆分 ObservationAuditor、EvidenceAuditor 与 DecisionAuditor；
- 建立正式的强透镜 Knowledge / Skill 分层；
- 暂不接入 LLM/VLM，直至审计和知识接口稳定。

---

# Lens Expert Agent v0.3.0（2026-08-09）

> 本节对应软件包版本 `0.3.0`，Observation schema 继续为 `1.1`。

## 升级目标

本版完成第二阶段升级：将单一流程完整性审计拆成 Observation、Evidence、Decision 三层科学审计，同时建立正式的 Knowledge / Skill 分离。继续坚持测量提供事实、Agent 解释事实，不接入 LLM、VLM 或 Self-Evolve。

## 新工作流

```text
Observer
→ Observation Audit
→ Multiple Hypotheses
→ Evidence Planning / Evaluation / Comparison
→ Counter Evidence / Per-Hypothesis Analysis
→ Evidence Audit
→ Decision / Confidence
→ Decision Audit
→ Structural Reasoning Audit
→ Report
```

## 分层审计

- `ObservationAuditor`：检查数据可用性、噪声、前景减除、测量可靠性、PSF/RMS 局限和引用来源；
- `EvidenceAuditor`：检查证据引用、重复/相关证据、缺失证据中性原则、区分力和过度解释；
- `DecisionAuditor`：检查替代解释、上游审计、阈值、关键缺失证据和反证，输出 `confirm`、`downgrade` 或 `abstain`；
- `ReasoningAuditor`：保留为结构审计，并验证阻断性科学审计是否被正确执行。

科学数据不可用但系统正确弃权时，结构审计可以通过。两者含义不同：前者评价科学结论，后者评价推理流程。

## Knowledge / Skill

新增六类领域知识：透镜几何、弧形态、本征星系形态、并合/相互作用、伪影/减除、颜色/多波段。新增三个审计技能。`BaseAgent` 提供知识主题与知识包加载接口，安装包元数据同时包含 `knowledge/*.md`。

## 输出与兼容性

- `reasoning.json` 和 `summary.json` 新增 `observation_audit`、`evidence_audit`、`decision_audit`；
- 最终分类采用决策审计后的有效结果，同时保留原始决策和置信度；
- CLI 显示有效置信等级、决策审计 verdict 和结构审计状态；
- Observation schema 仍为 1.1，旧 1.0 Observation 继续可读。

## 验证

- 自动化测试从 6 项增加到 8 项，全部通过；
- 新增不可用 Observation 强制弃权测试；
- 新增相关证据组与审计输出测试；
- 真实 FITS `102018667_NEG587746535508057234` 端到端回归通过；
- 回归结果：`strong_lens_candidate`，有效置信度 `medium`，Decision Audit 为 `confirm`，结构审计通过。

## 下一阶段

保持当前审计与知识接口稳定，下一阶段再建设独立领域 Plugin 输入/输出契约和 General Agent 对接边界；在验证集和错误案例体系建立前，仍不启用自动 Self-Evolve。

---

# 后续版本

（每次完成并验证一个可运行版本后，在这里继续追加。）


---

# 项目长期目标


最终构建一个能够模拟天文学专家分析强引力透镜候选体过程的智能系统。


系统不仅回答：

> “它是什么？”


还回答：

> “为什么这样判断？”

> “有哪些证据？”

> “还有哪些可能？”

> “结论有多可信？”

```

---
