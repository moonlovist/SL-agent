# Lens Expert Agent 中文使用说明书

> 适用版本：`lens-expert-agent 0.15.0`
>
> 当前状态：视觉层已暂时冻结；旧评分已退出；Q1 评分等待标定数据

## 1. 当前版本一句话说明

程序可以把 FITS/PNG 转换为可追溯的视觉测量，建立四种竞争解释，并列出支持
证据、反证、缺失信息和审计结果；但在 Q1 标定完成前，它不会假装给出
`expert_score` 或 A/B/C。

## 2. 实际处理流程

```text
FITS / PNG
↓
前景光建模与减除
↓
残差 S/N、多阈值、SEP、椭圆路线和候选交接
↓
Observation JSON（只记录看到了什么）
↓
观察审计
↓
H1 强透镜 / H2 星系自身结构 / H3 并合相互作用 / H4 伪影
↓
支持证据、反证、缺失证据、跨假设比较
↓
证据审计与推理结构审计
↓
评级准备状态
```

视觉层不输出“这是透镜”。它只提供测量。推理层也不再用未经标定的旧公式把
测量强行变成等级。

## 3. 安装

在项目根目录执行：

```powershell
python -m pip install -e ".[dev]"
```

项目要求 Python 3.11 或更高版本。主要依赖包括 Astropy、SEP、Photutils、
NumPy、SciPy、Matplotlib、Pillow 和 Pydantic。

## 4. 运行完整流程

```powershell
lens-agent analyze "G:\Astronomy\New data\A\样本文件夹名" --output outputs
```

也可以输入单个 FITS 或 PNG：

```powershell
lens-agent analyze "G:\Astronomy\sample.fits" --output outputs
lens-agent analyze "G:\Astronomy\sample.png" --output outputs
```

PNG 没有原始 FITS 的通量、RMS、PSF 和 WCS 信息，因此只适合降级观察，不能与
科学 FITS 证据完全等价。

## 5. 只运行推理层

```powershell
lens-agent reason "outputs\样本名\observation.json" --output "outputs\reasoning_only"
```

该命令只读取 `observation.json`，不读取原图或像素数组，用于验证视觉层与推理层
的边界。

## 6. 主要输出文件

| 文件 | 内容 |
|---|---|
| `observation.json` | 正式机器可读视觉测量 |
| `observation_summary_cn.md` | 视觉测量的中文说明，不含评级 |
| `vision_diagnostic.png` | 输入、前景模型、残差和候选总览 |
| `vision_snr_threshold_process.png` | 多个 S/N 阈值下结构怎样存活 |
| `vision_sep_consensus.png` | SEP 多参数检测痕迹 |
| `vision_model_chain.png` | SEP、路线支持和路线恢复的关系 |
| `vision_signal_audit.png` | 候选信号质量与交接结果 |
| `vision_artifacts.npz` | 像素级数值溯源；VS Code 普通文本编辑器不能直接阅读 |
| `vision_manifest.json` | 输入指纹、运行参数和产物路径 |
| `reasoning.json` | 完整多假设证据链与评级状态 |
| `summary.json` | 面向程序使用的精简结果 |
| `report.md` | 面向人工阅读的专家证据报告 |

## 7. 四种视觉候选交接

| 类别 | 含义 |
|---|---|
| `primary_candidate` | 当前最适合进入后端解释的可测残差信号 |
| `recovery_candidate` | 可能有用但证据不完整，保留待复核 |
| `foreground_contaminated` | 容易受前景减除误差污染 |
| `trace_candidate` | 弱线索，只保留追溯 |

这些名称只描述“信号怎样交接”，不等于弧、环、多重像或透镜标签。

## 8. 四种竞争假设

| 编号 | 解释 |
|---|---|
| H1 | 强引力透镜 |
| H2 | 前景星系自己的旋臂、环或星形成结构 |
| H3 | 星系并合、相互作用或潮汐结构 |
| H4 | 噪声、PSF、坏像素或前景减除伪影 |

同一观察事实会对四个假设分别记录 `support`（支持）、`oppose`（反对）、
`neutral`（中性）或 `unknown`（尚无解释规则）。缺失测量不能当作反对证据。

## 9. 为什么不再显示旧分数和 confidence

旧版的 `weighted_score`、假设分数、分数差和 `confidence.score` 是为了验证流程而
编写的人工公式，没有使用 Euclid Q1 的连续 `expert_score` 标定。它们既不是
真实透镜概率，也不能严谨对应 A/B/C，因此已从活动代码删除。

证据中的 `strength`、`reliability` 和 `specificity` 目前只保留为证据描述及审计
元数据，不会相乘产生最终等级。未来量化审计完成后，应优先读取原始测量值、
误差、可观测性和相关证据组，而不是把这些旧描述值直接当成 Q1 分数。

## 10. 怎样阅读 grading

当前 `summary.json` 包含：

```json
{
  "grading": {
    "grading_scheme": "euclid_q1_paper",
    "status": "calibration_pending",
    "expert_score": null,
    "grade": null,
    "a_plus": null,
    "rationale": "...",
    "missing_requirements": ["..."]
  }
}
```

状态含义：

| 状态 | 应怎样理解 |
|---|---|
| `calibration_pending` | 流程正常，但评分器还没有用 Q1 数据标定 |
| `insufficient_data` | 数据无法支持科学评级，不等于非透镜 |
| `blocked_by_audit` | 证据来源或推理结构有问题，修复前不能评级 |
| `graded` | 未来标定并验证完成后才会出现 |

`null` 表示“没有进行这项判断”。它和数字 0、等级 X、`false` 完全不同。

## 11. Q1 论文等级映射

项目只保留一套论文口径映射。论文原文明确给出 A 和 B 的分界，并将低于 B
分界的结果描述为 C 或更低；来源见
[Euclid Q1 Strong Lensing Discovery Engine论文](https://arxiv.org/abs/2503.15324)：

```text
A：expert_score > 2.0
B：1.5 < expert_score <= 2.0
C：expert_score <= 1.5
```

映射本身已经实现并测试，但目前没有任何活动模块获准生成 `expert_score`。
Q1论文没有规定`expert_score <= 1.0`自动映射为X的目录分箱，因此当前实现
不会自行增加该边界。

A+ 是 A 级上的独立科学价值标记，不是更高的数值区间。只有先得到 A，并完成
专门科学价值审查后，才能写 `a_plus=true`，同时必须记录理由。

## 12. 低质量数据为什么不是 X

X 的意思是“现有可用信息更支持非透镜”。如果图像损坏、噪声模型无效或有效
像素不足，专家其实无法判断。程序会输出 `insufficient_data`，防止把“看不清”
错误写成“确定不是”。

## 13. 当前审计和未来评分前审计的区别

当前已经存在：

- Observation Audit：检查数据、噪声、前景模型和测量来源；
- Evidence Audit：检查证据引用、重复信息、相关性和低区分力；
- Reasoning Audit：检查四个假设案卷和推理引用是否完整。

尚未实现的是“评分前量化审计表”。它需要把中心关系、切向关系、径向一致性、
颜色/形态相容性、前景模型敏感性和替代解释等字段统一整理，并使用 Q1 标注数据
校准。当前版本只为它留出清晰接口，没有猜测阈值或权重。

## 14. 数据整理建议

未来标定表最好包含：

```text
sample_id
Q1 expert_score
论文标准 grade
原始 FITS 路径
是否有人工作出的结构位置
最重要的支持理由
最有竞争力的非透镜解释
```

如果只有 A/B/C 而没有连续 `expert_score`，可以先研究等级区分，但不能严格
重建论文连续分数。应当保留一批完全不参与定阈值的独立验证样本，避免过拟合。

## 15. 测试

```powershell
python -m pytest
```

测试覆盖视觉测量、WCS、前景模型、几何搜索、候选交接、推理引用、不可评级
状态、Q1 分界点和 A+ 约束。

## 16. 当前科研边界

当前版本可以可靠回答：

- 程序测量到了什么；
- 测量来自哪些像素和处理步骤；
- 各观察事实支持或反对哪些解释；
- 哪些信息缺失或存在冲突；
- 数据和推理链是否具备进入评分阶段的条件。

当前版本不能可靠回答：

- 该目标的正式 Q1 `expert_score` 是多少；
- 它应当被分为 A/B/C 中的哪一级；
- 某个内部数字是否是真实透镜概率。

这些限制不是程序故障，而是为了在标定完成前避免用主观数字冒充科研结论。
