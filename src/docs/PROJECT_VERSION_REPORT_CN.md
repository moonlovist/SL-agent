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

# 下一阶段规划

## V2.3 Decision Separation

目标：

拆分：

假设选择

与

可信度判断。


新增：

DecisionAgent


职责：

- 比较候选假设；
- 根据证据评分选择最佳解释。


ConfidenceAgent：

专注：

- 判断可信程度；
- 记录不确定性；
- 分析风险。


---


# Version 3.0

（待更新）


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
