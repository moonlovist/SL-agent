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

# Version 2.0

## 专家推理流程重构


## 版本目标


v2.0是一次重要架构升级。


核心变化：

从：

> 固定假设检查系统


升级为：

> 模拟专家判断过程的推理系统。


---

# v2.0主要完成内容


## 1. Hypothesis Generation升级


旧流程：

```
观察

↓

列出所有可能

↓

逐个检查
```


升级：

```
观察事实

↓

生成竞争解释

↓

提出验证问题

↓

寻找关键证据
```


新增：

- 初始可信度；
- 支持证据需求；
- 可能反例。


---

## 2. Evidence Evaluation升级


EvidenceAgent不再只是生成证据占位。


升级为：

根据假设：

分析：

- 哪些证据支持该解释；
- 哪些证据能够区分不同解释。


---

## 3. CounterEvidence增强


新增专家反思机制：

> 如果当前判断错误，可能是什么原因？


用于：

- 降低确认偏差；
- 主动寻找替代解释。


---

## 4. 新增Confidence Calibration


加入：

ConfidenceAgent


负责：

综合：

- 证据；
- 反证；
- 不确定性。


输出：

- 判断；
- 可信度；
- 支持因素；
- 限制因素。


---

## 5. Reasoning Audit升级


增强推理检查。


检查：

- 是否存在无证据结论；
- 是否考虑替代解释；
- 是否记录不确定性。


---

# v2.0最终推理流程


```
Observation

↓

Hypothesis Generation

↓

Evidence Evaluation

↓

Counter Evidence Analysis

↓

Confidence Calibration

↓

Reasoning Audit

↓

Expert Report
```


---

# v2.0解决的问题


相比v1.0：

|问题|改进|
|-|-|
|固定假设枚举|竞争解释|
|所有证据全部检查|强调关键证据|
|只寻找支持|加入反证分析|
|结果导向|推理过程导向|
|缺少可信度|加入Confidence Calibration|


---
## v2.0测试状态

v2.0核心推理流程已经完成初步运行测试。

验证内容：

- Agent流程完整运行；
- ExpertState完成模块化重构；
- Confidence结果与最终报告分离；
- ReasoningAudit能够正确检查完整推理流程。

当前限制：

- Observer仍等待真实视觉模型；
- Evidence仍为规则驱动；
- Hypothesis选择仍需要未来引入动态决策机制。

# 后续版本记录


# Version 2.1
## V2.1 Evidence Planning Upgrade

新增：
- EvidencePlannerAgent
- evidence_plan state
- Evidence-guided evaluation workflow

改进：
- 从穷举式证据检查升级为优先级证据选择
- 减少无效假设验证
- 提高专家推理效率

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
