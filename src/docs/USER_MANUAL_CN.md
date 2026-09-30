# Lens Expert Agent 中文使用说明书

> 适用版本：`lens-expert-agent 0.29.0`
>
> 当前状态：已升级前景中心选择、弯曲结构测量与逐结构比较；输出单专家暂定类别；正式 Q1 聚合分数等待标定

本版具体变更见 [v0.19.0 修改说明](V0190_CENTER_STRUCTURE_CHANGE_CN.md)。默认输入是目标居中的切图。
若目标不在切图中央，可在 `analyze` 命令同时提供 `--center-x` 和 `--center-y`，以图像数组的
零起始像素坐标指定目标参考位置；不要把图中文字的上下方向当成数组坐标。中心歧义时会弃权。

## 1. 当前版本一句话说明

程序把 FITS/PNG 转换为可追溯视觉测量，由唯一的 `ExpertAssessmentAgent`
建立证据档案、比较八类透镜构型与五类竞争解释，并给出明确标为未校准的
单专家类别；独立
`DecisionAudit` 只检查错误。正式 `official_expert_score` 在 Q1 标定前保持空值。

## 2. 实际处理流程

```text
FITS / PNG
↓
前景光建模与减除
↓
残差 S/N、多阈值、SEP和椭圆路线
↓
信号真实性审计：可信绿色与青色结构完整交接
↓
Observation JSON（只记录看到了什么）
↓
ExpertAssessmentAgent：证据档案 + 八类构型 + 五类竞争解释 + A/B/C/X单选
↓
DecisionAudit：轻量独立错误检查
↓
暂定 Q1 映射与报告
```

视觉层不输出“这是透镜”。专家层不把分项相加，也不把暂定分数解释成概率。

## 3. 安装

安装桌面界面及测试依赖：

```powershell
python -m pip install -e ".[dev,ui]"
```

启动完整桌面流程：

```powershell
lens-agent ui
```

主界面支持单 FITS/文件夹导入、后台顺序跑批、A/B/C/X/无法判断分组、过程图与判定依据浏览、人工校正后的版本化重跑、系统完成提醒，以及日间/夜间主题。普通批处理不会因前景失败自动弹窗；需要时由用户点击“人工校正”。

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

### 人工指定前景星系

在正式前景拟合前打开原生像素标注窗口：

```powershell
lens-agent analyze "G:\Astronomy\sample.fits" --interactive-foreground --output outputs
```

最简操作只需三步：

1. 把唯一的绿色点拖到正确前景星系附近；
2. 点击“按绿色点选择最近对象并生成第2层保守核心”；程序不会再复制整个 SEP 岛；
3. 检查第 2 层位于正确星系的明亮、连续核心，然后点击“接受人工前景并继续完整重跑”。

第 2 层不是完整前景外缘，也不是要直接减除的区域；人只需确认对象及可信核心，不必沿
星系轮廓精细抠图。只有自动范围明显吞入 arc/环/源像时，才用画笔或橡皮修正
第 2 层，或在第 3 层涂出明确禁止参与前景拟合的结构。编号比标签颜色更可靠；不同
napari 版本的标签颜色可能不同。

人工中心会在不超过 3 像素的范围内按亮核精调。人工改换目标后，拟合掩膜与残差检测
掩膜会围绕新中心重新生成；不会继承旧自动目标的遮罩。若所有前景模型未通过背景零点、
宽环残留及核心残差质量检查，视觉候选检测停止并返回无法评级，而不是把失败残差评为 X。

若完整运行后才发现前景错误，可从旧结果重新标注并写到新目录：

```powershell
lens-agent foreground-review "outputs\样本名" --output outputs\human_foreground_rerun
```

人工标注保存为带输入哈希的 `foreground_override.json` 和原生网格 FITS 掩膜；它只改变
前景选择与拟合输入，后续检测和评级必须完整重跑。旧结果不会被覆盖。

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
| `reasoning.json` | Evidence Dossier、五类解释、暂定专家判断与DecisionAudit |
| `summary.json` | 面向程序使用的精简结果 |
| `report.md` | 面向人工阅读的专家证据报告 |

## 7. 四种视觉候选交接

| 类别 | 含义 |
|---|---|
| `primary_candidate` | 当前最适合进入后端解释的可测残差信号 |
| `recovery_candidate` | 路线外稳定信号或恢复信号；可信者带完整测量进入评级 |
| `foreground_contaminated` | 独立比较模型明确擦除的候选，隔离但保留追溯 |
| `trace_candidate` | 弱线索，只保留追溯 |

这些名称只描述“信号怎样交接”，不等于弧、环、多重像或透镜标签。
接近核心、主前景模型失败或未命中椭圆路线只形成风险/来源标记，不能单独删除
独立模型仍保留的真实信号。

专家层分别检查八种构型：环/部分环、切向弧、折叠弧、尖点弧、径向弧、
紧致多像（包括爱因斯坦十字）、高度不对称多像和单像。它们是互相替代的解释
路径，不是八个可相加的分项。短弧、径向弧或紧致多像不再被统一的60度弧长、
切向角或结构数门槛排除。

## 8. 五种竞争解释

| 标识 | 解释 |
|---|---|
| `strong_gravitational_lens` | 强引力透镜 |
| `intrinsic_galaxy_morphology` | 前景星系自身形态 |
| `merger_or_interaction` | 星系并合或相互作用 |
| `foreground_subtraction_or_psf_artifact` | 前景减除或 PSF 伪影 |
| `noise_bad_pixel_or_projected_source` | 噪声、坏像素或随机投影源 |

每个解释记录支持事实、冲突事实、缺失信息和理由。缺失测量不能当作反对证据。

## 9. 为什么不再显示旧分数和 confidence

旧版的 `weighted_score`、假设分数、分数差和 `confidence.score` 是为了验证流程而
编写的人工公式，没有使用 Euclid Q1 的连续 `expert_score` 标定。它们既不是
真实透镜概率，也不能严谨对应 A/B/C，因此已从活动代码删除。

证据中的 `strength`、`reliability` 和 `specificity` 目前只保留为证据描述及审计
元数据，不会相乘产生最终等级。未来量化审计完成后，应优先读取原始测量值、
误差、可观测性和相关证据组，而不是把这些旧描述值直接当成 Q1 分数。

## 10. 怎样阅读 expert_assessment

当前 `summary.json` 包含：

```json
{
  "expert_assessment": {
    "vote_category": "A",
    "plausible_vote_categories": ["A"],
    "vote_code": null,
    "expert_vote_anchor": null,
    "provisional_expert_score": null,
    "provisional_score_range": null,
    "provisional_grade": "A",
    "score_status": "provisional_uncalibrated",
    "calibration_required": true,
    "official_expert_score": null,
    "scientific_value_flags": []
  }
}
```

状态含义：

| 输出 | 应怎样理解 |
|---|---|
| `vote_category` | Agent 可以明确选择的单个专家类别 |
| `plausible_vote_categories` | 与单专家选择一致的单元素类别列表 |
| `vote_code` | 兼容字段，当前保持空值；单专家不制造数字分 |
| `provisional_uncalibrated` | 当前结果是透明规则层的暂定专家判断，不等价于人类聚合分数 |
| `provisional_score_range` | 兼容字段，当前保持空值 |
| `insufficient_for_grading` | 数据无法支持科学评级，不等于非透镜 |
| `official_expert_score: null` | 尚未进行 Q1 标定与独立验证 |

`null` 表示“没有进行这项判断”。它和数字 0、等级 X、`false` 完全不同。

## 11. Q1 论文等级映射

项目只把论文明确给出的 A 和 B 分界用于正式 catalogue 分箱；来源见
[Euclid Q1 Strong Lensing Discovery Engine论文](https://arxiv.org/abs/2503.15324)：

```text
A：expert_score > 2.0
B：1.5 < expert_score <= 2.0
below_B_cut：expert_score <= 1.5（论文未给连续 X/C 完整边界）
```

活动模块只能生成明确标为 provisional 的单专家类别，不能生成正式聚合
`official_expert_score`。X=0、C=1、B=2、A/A+=3来自论文的单次专家投票编码；
正式目录等级必须等待多专家聚合、专家偏差校正和本项目校准。

A+ 是 A 级上的独立科学价值标记，不是更高的数值区间。只有先得到 A，并完成
专门科学价值审查后，才能写 `a_plus=true`，同时必须记录理由。

## 12. 低质量数据为什么不是 X

X 的意思是“现有可用信息更支持非透镜”。如果图像损坏、噪声模型无效或有效
像素不足，专家其实无法判断。程序会输出 `insufficient_data`，防止把“看不清”
错误写成“确定不是”。

## 13. 当前 DecisionAudit 与未来标定审计的区别

当前 `DecisionAudit` 检查事实引用、缺失当反证、相关证据重复、测量当证明、
恢复自动降级、替代解释遗漏、答案泄漏、文字分数冲突和弃权需要。它不重跑多
假设分析。未来标定审计需要用 Q1 标注数据验证类别投票和弃权策略，不能
从单个案例猜测权重。

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

测试覆盖视觉测量、WCS、前景模型、几何搜索、候选交接，以及多构型专家判断边界、
不可评级状态、Q1 分界点和 A+ 约束。

## 16. 当前科研边界

当前版本可以可靠回答：

- 程序测量到了什么；
- 测量来自哪些像素和处理步骤；
- 各观察事实如何影响五类竞争解释；
- 哪些信息缺失或存在冲突；
- 当前单专家 A/B/C/X 类别，以及 DecisionAudit 是否通过。

当前版本不能可靠回答：

- 该目标的正式 Q1 `expert_score` 是多少；
- 它的暂定等级是否等价于 Q1 多专家正式目录等级；
- 某个内部数字是否是真实透镜概率。

这些限制不是程序故障，而是为了在标定完成前避免用主观数字冒充科研结论。
