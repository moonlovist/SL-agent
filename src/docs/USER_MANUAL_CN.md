# Lens Expert Agent 全面介绍与使用说明书

> 适用版本：`lens-expert-agent 0.12.0`（候选结构审计与后端交付版 MWA）
>
> v0.3 阅读提示：系统现在有三位科学“复核员”。ObservationAuditor 检查测量能不能信，EvidenceAuditor 检查证据有没有重复或夸大，DecisionAuditor 检查最终结论应当确认、降级还是弃权。原 ReasoningAuditor 仍只检查程序结构和引用完整性。`summary.json` 中的最终分类以 DecisionAuditor 审核后的结果为准，原始初判仍会保留，方便追溯。
> 适用系统：Windows + PowerShell；Python 3.11 或更高版本  
> 本手册依据当前实际代码编写。凡是尚未实现的功能，都会明确标为“当前不支持”或“未来扩展”。

---

## 1. 一句话认识本项目

Lens Expert Agent 是一个用于**强引力透镜候选体检查**的、以证据为中心的专家推理系统。

它不只是回答“是透镜”或“不是透镜”，而是先从天文图像中测量事实，再比较多种可能解释，检查支持证据和反证，最后给出带有不确定性说明、审计结果和可追溯依据的报告。

它的核心流程是：

```text
FITS / PNG 图像
       ↓
视觉测量层：图像中实际测到了什么？
       ↓
Observation JSON：结构化观察记录
       ↓
专家推理层：哪一种解释最符合这些事实？
       ↓
结论 + 可信度 + 证据 + 反证 + 局限性 + 审计
```

可以把它想象成一家医院中的两类角色：

- 视觉测量层像检验科，只负责产生化验结果，不负责下诊断；
- 专家推理层像医生，依据化验结果比较不同疾病解释，再给出诊断意见；
- 推理审计器像会诊审核，检查医生是否跳过了关键步骤或引用了不存在的检查结果。

这种分工是本项目最重要的设计原则。

---

## 2. 为什么强引力透镜识别不能只看“有没有弧”

### 2.1 什么是强引力透镜

当前景的大质量天体（通常是星系或星系团）位于更遥远的背景天体前方时，它的引力会弯曲背景天体发出的光。若位置和质量条件合适，背景源可能被观测成：

- 拉长的弧；
- 多个相似像；
- 接近圆环的爱因斯坦环；
- 围绕前景星系近似切向排列的结构。

通俗地说，前景星系像一个由引力形成的“宇宙玻璃透镜”。但它不像普通玻璃那样规则，所以形成的像可能不完整、不对称，也可能很微弱。

### 2.2 为什么“看到弧”仍然不能直接下结论

弧形或细长结构还可能来自：

- 星系自身的旋臂、环或恒星形成区；
- 两个星系并合或相互作用产生的潮汐结构；
- 前景星系光模型减除不完整留下的残差；
- 噪声、坏像素、衍射结构或图像处理伪影。

所以本系统同时维护四种竞争假设：

| 编号 | 假设 | 通俗解释 |
|---|---|---|
| H1 | `strong_gravitational_lens` | 强引力透镜 |
| H2 | `intrinsic_galaxy_morphology` | 星系自身结构，例如旋臂或环 |
| H3 | `merger_or_interaction` | 星系并合、相互作用或潮汐结构 |
| H4 | `observation_or_subtraction_artifact` | 噪声、观测伪影或星系减除残差 |

项目遵循的不是：

```text
看到弧 → 判为透镜
```

而是：

```text
测到弧状结构
→ 检查它是否围绕中心、是否近似切向、是否位于共同半径
→ 检查信噪比和检测稳定性
→ 比较透镜、星系内部结构、并合和伪影四种解释
→ 证据不足时主动不下结论
```

---

## 3. 当前版本能做什么、不能做什么

### 3.1 当前已经实现

当前 MWA 已经能够：

- 读取单个 FITS 图像；
- 读取包含一个目标 FITS 的目标文件夹；
- 读取单个 PNG 图像，并以保守的降级模式处理；
- 从指定 FITS 波段扩展中读取 flux 和 RMS；
- 估计背景和噪声；
- 定位中心亮源；
- 对前景星系光进行等光度椭圆建模，失败时使用径向模型回退；
- 生成星系减除残差和残差信噪比图；
- 使用多信噪比阈值检查结构稳定性；
- 使用多组 SEP 去混叠参数检查检测稳定性；
- 提取环带内候选弧像素和结构；
- 测量半径、位置角、角向延展、轴比、切向错位角和积分信噪比；
- 生成严格、版本化的 Observation JSON；
- 在不读取图像的条件下，仅根据 Observation JSON 进行专家推理；
- 比较四种竞争假设；
- 生成支持证据、反证、排名、结论和可信度；
- 对证据引用和推理流程进行自动审计；
- 输出 JSON、Markdown 报告、诊断图和像素级溯源数组；
- 在证据不足、分数过低或竞争假设差距太小时主动输出 `indeterminate`。

### 3.2 当前尚未实现

当前版本还**不支持**：

- 对多个波段进行联合颜色推理；
- 自动比较不同透镜像的颜色一致性；
- PSF 感知的精细前景星系建模；
- 使用 WCS 将像素距离可靠换算为角秒；
- 专门的多核心、潮汐尾和并合形态测量；
- 透镜质量模型拟合或爱因斯坦半径物理反演；
- 批量递归分析整个数据集；
- 经过标注数据校准的透镜概率；
- 在线大语言模型或视觉语言模型参与决策；
- 用训练好的 CNN 直接分类。

特别重要：当前的 `confidence.score` 是内部推理可信度指标，**不是“成为真实透镜的概率”**。例如 `0.64` 不能解释为“64% 是透镜”。

---

## 4. 当前软件架构

### 4.1 两层边界

```text
┌─────────────────────────────────────────────┐
│ Vision Observation Layer                    │
│ 输入：FITS / PNG                            │
│ 输出：测量事实、质量标记、方法和溯源         │
│ 禁止：输出 lens / not lens 或透镜概率        │
└──────────────────────┬──────────────────────┘
                       │ ObservationDocument 1.0
┌──────────────────────▼──────────────────────┐
│ Expert Reasoning Layer                      │
│ 输入：Observation JSON                      │
│ 输出：竞争假设、证据、反证、判断、可信度、审计│
│ 禁止：重新读取 FITS、PNG 或像素数组          │
└─────────────────────────────────────────────┘
```

这样做有三个好处：

1. **可解释**：每条结论都能追溯到测量编号；
2. **可替换**：未来可替换视觉工具，而不必重写整个推理系统；
3. **可审计**：推理层不能悄悄从原图中“猜”出 Observation 中不存在的事实。

### 4.2 实际运行的代码目录

当前命令 `lens-agent` 的入口是：

```text
src/expert_agent/
```

其中：

```text
src/expert_agent/
├── cli.py                 命令行入口
├── vision/                图像读取、预处理和结构测量
├── schemas/               Observation 与 Reasoning 数据契约
├── agents/                各推理阶段
├── core/                  配置、状态和工作流编排
└── skills/                专家规则设计文档
```

仓库中的 `src/FORMA/` 是较早阶段的另一套代码来源或历史架构参考，**不是当前 `lens-agent` 命令安装和执行的包**。当前打包配置只包含 `expert_agent*`。

---

## 5. 视觉测量层如何工作

### 5.1 读取图像

对 FITS，程序优先寻找：

```text
<BAND>_FLUX
<BAND>_RMS
```

默认 `BAND=VIS`，所以默认查找：

```text
VIS_FLUX
VIS_RMS
```

如果找不到 `<BAND>_FLUX`，但 FITS 主 HDU 是二维图像，程序会读取主 HDU，并从图像本身估计噪声。

对 PNG，程序会转为灰度图，并从像素估计噪声。PNG 通常是为了显示而渲染的图像，没有可靠的原始通量、误差图和仪器标定，因此测量精度低于 FITS。

### 5.2 前景星系光减除

前景透镜星系往往非常亮，而背景弧很暗。若直接看原图，弧可能像站在探照灯旁边的一只萤火虫。

程序先建立前景星系光模型，再执行：

```text
残差图 = 原始图像 - 前景星系模型
```

当前优先使用 Photutils 等光度椭圆模型；如果模型无法成功收敛，则回退到径向亮度模型。模型是否成功、使用了哪种方法，都会记录在 `observation.json` 中。

减除并不保证完美。靠近星系中心的结构尤其容易受到模型误差影响，因此系统会标记 `near_core_subtraction_sensitive`。

### 5.3 残差信噪比

信噪比可以粗略理解为：某个信号比背景随机起伏明显多少。

- 信噪比低：可能只是噪声；
- 信噪比高：说明结构更可能真实存在；
- 但信噪比高并不自动说明它是透镜，也可能是旋臂或减除残差。

因此信噪比只回答“这个结构是否显著”，不回答“这个结构是什么”。

### 5.4 多阈值稳定性

程序会使用一组不同的信噪比阈值重复检测。如果某结构只在一个非常特殊的阈值下出现，它更可疑；如果在多种阈值下持续存在，它更稳定。

默认 FITS 阈值为：

```text
2.00, 2.58, 3.34, 4.31, 5.57, 7.19, 9.29, 12.00
```

### 5.5 SEP 参数共识

SEP 是 Source Extractor 思路的 Python 实现之一。程序使用多组去混叠参数反复检测结构，检查结果是否过度依赖某一次参数选择。

这类似于把同一张图交给多位使用略有不同放大和分割标准的检验员：如果大家都能找到大致相同的结构，结果更稳定。

但这些检测都使用同一张星系减除残差图，因此它们并不是完全独立的证据。系统通过 `correlation_group` 标记这种相关性，并在决策时按相关组聚合，避免把同一个事实重复计分。

### 5.6 弧结构测量

对于每个候选结构，系统会记录：

| 字段 | 含义 | 如何理解 |
|---|---|---|
| `centroid_xy` | 结构中心像素坐标 | 它位于图像哪里 |
| `radius_px` | 到星系中心的半径 | 它离中心多远 |
| `position_angle_deg` | 位置角 | 它位于中心的哪个方向 |
| `pixel_count` | 像素数量 | 结构覆盖多少像素 |
| `integrated_snr` | 积分信噪比 | 整体显著程度 |
| `angular_extent_deg` | 围绕中心延展的角度 | 是小块还是长弧 |
| `axis_ratio` | 长短轴比 | 是否细长 |
| `tangential_misalignment_deg` | 与理想切向方向的错位 | 越小越接近绕中心的弧 |
| `threshold_survival_fraction` | 多阈值存活比例 | 对阈值是否稳定 |
| `sep_survival_fraction` | SEP 参数网格存活比例 | 对分割参数是否稳定 |
| `near_core` | 是否靠近中心 | 是否易受减除误差影响 |

其中“切向”可以想象为圆周上某一点的行进方向。如果结构像一段围绕中心的圆弧，它通常近似沿圆的切线方向伸展；若它直指中心，则更像径向结构。

---

## 6. 专家推理层如何工作

当前推理流程按固定顺序运行：

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

### 6.1 ObserverAgent：整理观察事实

它把 Observation 文档整理成便于后续推理使用的观察摘要，并明确记录：

```text
classification_performed = false
```

也就是说，观察阶段没有偷偷进行分类。

### 6.2 HypothesisAgent：保留竞争解释

它建立 H1–H4 四个候选解释。假设不是结论，而是需要证据检验的“可能答案”。

### 6.3 EvidencePlannerAgent：决定先检查什么

它规划具有区分力的问题，例如：

- 结构是否近似切向？
- 是否具有足够的角向延展？
- 多个结构是否位于共同半径？
- 结构是否在不同检测设置下稳定？
- 前景星系模型是否成功？
- 是否存在多核心或潮汐结构？
- 多波段颜色是否一致？

当前测量层无法回答的问题不会被伪装成反证，而会保留为“未回答问题”。

### 6.4 EvidenceAgent：形成带引用的证据

每条证据包含：

- 支持或反对哪个假设；
- 具体主张；
- 引用了哪些测量 ID；
- 方向：支持、反对或中性；
- 强度 `strength`；
- 可靠性 `reliability`；
- 特异性 `specificity`；
- 加权分数；
- 相关性分组；
- 解释理由；
- 是否已有数据可评估。

加权分数的基本形式是：

```text
方向 × 强度 × 可靠性 × 特异性
```

“特异性”非常重要。例如，高信噪比只能说明结构真实，却不专属于透镜；共同半径和切向几何通常对透镜解释更有区分力。

### 6.5 EvidenceComparisonAgent：比较同一事实对四种解释的影响

它把引用同一组测量的证据放到一行中，明确记录这个事实对 H1–H4
分别是支持、反对、中性还是当前未知。这样可以区分“事实很明显”和
“事实真正能够区分假设”这两个不同概念。

### 6.6 CounterEvidenceAgent：主动找反例

它不会只罗列支持 H1 的事实，还会把反对某个假设的证据转化成明确的矛盾项。

这样可降低“确认偏差”：一旦先喜欢上某个解释，人往往只继续寻找支持它的信息，而忽略反例。

### 6.7 HypothesisAnalysisAgent：为每个假设建立独立案卷

它分别汇总每个假设的支持证据、反对证据、缺失证据、不确定因素、
已解释观察和尚未解释的相关观察。`viable`、`weakened` 和
`insufficient` 只是进入综合决策前的案卷状态，不是最终分类。

### 6.8 DecisionAgent：比较解释并决定是否弃权

DecisionAgent 先按相关证据组聚合，再计算每个假设的：

```text
最终分数 = 支持证据分数 - 反证惩罚
```

当前默认决策门槛包括：

- 最高最终分数至少为 `0.12`；
- 第一名与第二名差距至少为 `0.06`。

对 H1 还有额外的“透镜特异几何门”：至少需要一条特异性不低于 `0.6`、来自透镜几何或场景几何组的支持证据。仅凭“高信噪比”或“有一个细长结构”不能强行通过。

可能输出三种分类：

| 分类 | 含义 |
|---|---|
| `strong_lens_candidate` | 现有证据更支持强透镜候选解释 |
| `not_lens_preferred` | 非透镜解释在当前证据下更优 |
| `indeterminate` | 证据不足、竞争太接近或审计不能支持明确结论 |

`strong_lens_candidate` 表示“值得作为候选体进一步检查”，不等于“已被科学确认的强透镜”。

### 6.9 ConfidenceAgent：评价这次判断有多可靠

判断与可信度是两个不同问题：

- 判断问：“哪个解释目前排名第一？”
- 可信度问：“我们有多大把握相信这个排名？”

可信度综合考虑：

- 首选假设分数；
- 与第二名的差距；
- 证据计划完成度；
- 数据质量；
- 已知限制和未回答问题。

可信度等级为：

```text
high / medium / low / insufficient
```

### 6.10 ReasoningAuditor：检查推理是否合法

审计器会检查：

- Observation 是否存在且符合结构；
- 测量 ID 是否唯一；
- 是否保留多个竞争假设；
- 是否真正评估了证据；
- 证据引用是否能追溯到测量；
- 反证引用是否有效；
- 跨假设比较是否覆盖全部四种解释；
- 是否为每个假设形成了完整案卷；
- 决策是否满足约束；
- 是否生成可信度结果。

若审计失败，ReportWriter 会把有效分类降为 `indeterminate`，防止输出未经审计支持的明确判断。

### 6.11 ReportWriter：只总结，不创造新事实

报告生成器把已有测量和推理结果组织成人类可阅读的 Markdown 报告。它不应加入 Observation 和推理记录中不存在的新天文事实。

---

## 7. 安装与环境准备

### 7.1 环境要求

- Python 3.11 或更高版本；
- Windows PowerShell；
- 足够的磁盘空间保存 NPZ 和诊断图；
- 运行分析不需要联网，也不需要 LLM API Key。

核心依赖包括：

- `astropy`：读取 FITS；
- `photutils`：星系等光度椭圆建模；
- `sep`：源检测和去混叠；
- `numpy`、`scipy`：数值计算；
- `matplotlib`：生成诊断图；
- `Pillow`：读取 PNG；
- `pydantic`：严格验证 JSON 数据结构；
- `pytest`：运行测试。

### 7.2 当前电脑上的运行方式

当前已验证的 Python 解释器是：

```text
G:\Python\Python312\python.exe
```

当前项目不依赖项目目录中的 `.venv`。VS Code 应选择上述解释器。

检查命令：

```powershell
Get-Command python
Get-Command lens-agent
python --version
lens-agent --help
```

### 7.3 在干净环境中安装（推荐的可复现方式）

进入项目目录：

```powershell
Set-Location "G:\Astronomy\LensAgent - 副本"
```

创建虚拟环境：

```powershell
py -3.12 -m venv .venv
```

激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装项目及测试依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 使用 `-e .[dev]`，真正的依赖声明以 `pyproject.toml` 为唯一来源。`-e` 表示可编辑安装：修改源码后通常不需要重新安装包。

验证：

```powershell
python -m pip check
python -m pytest
lens-agent --help
```

预期测试结果为：

```text
4 passed
```

注意：版本号使用下限约束而非完全锁定，因此当前项目能够重新安装，但还不是“任意时间安装都得到完全相同依赖版本”的锁文件级复现。科研生产环境建议以后增加锁文件和持续集成测试。

### 7.4 VS Code 解释器设置

若 VS Code 尝试激活一个不存在的 `.venv`：

1. 按 `Ctrl+Shift+P`；
2. 选择 `Python: Select Interpreter`；
3. 选择 `G:\Python\Python312\python.exe`，或选择你刚创建的 `.venv\Scripts\python.exe`；
4. 关闭旧终端并新建终端；
5. 必要时运行 `Developer: Reload Window`。

---

## 8. 输入格式与选择规则

### 8.1 单个 FITS 文件

支持直接传入：

```text
.fits
.fit
.fts
```

FITS 图像数据必须最终是二维数组。程序支持以下两种组织方式：

1. 扩展名为 `<BAND>_FLUX`，可选 `<BAND>_RMS`；
2. 主 HDU 本身包含二维图像。

### 8.2 包含 FITS 的目标文件夹

支持传入一个目标目录，但目录解析规则比较严格：

1. 优先寻找 `目录名.fits`；
2. 若不存在，则要求目录根层级恰好只有一个 `*.fits`；
3. 若找到零个或多个 `*.fits`，会报错；
4. 不会递归搜索子目录；
5. 目录模式当前只解析 `.fits`，不会在目录中自动选择 PNG、`.fit` 或 `.fts`。

例如：

```text
example/
└── 102018667_NEG587746535508057234/
    ├── 102018667_NEG587746535508057234.fits
    ├── rgb_0.png
    └── rgb_1.png
```

把该目录传给 `analyze` 时，会优先选择同名 FITS，不会选择两个 PNG。

### 8.3 单个 PNG 文件

支持直接传入 `.png` 文件。PNG 会：

- 转为灰度；
- 从像素估计噪声；
- 使用更保守、更小的检测参数网格；
- 在 Observation 中记录没有 supplied RMS；
- 记录 PNG 模式的限制；
- 因数据缺少定标而降低可解释能力。

PNG 更适合演示、初筛和流程测试，不应代替带误差图的科学 FITS 分析。

### 8.4 多波段 FITS

当前 `--band` 可以选择一个波段，例如：

```powershell
lens-agent analyze "target.fits" --band VIS
lens-agent analyze "target.fits" --band H
```

第二条命令会寻找 `H_FLUX` 和 `H_RMS`。

但这仍是**一次选择一个波段进行单波段分析**。当前版本不会同时读取 VIS、Y、J、H 等波段，也不会联合比较颜色。若需要分析多个波段，可以分别运行并人工比较报告，但不能把这种方式称为已经实现的多波段联合推理。

### 8.5 当前不支持的输入

- JPEG、TIFF、BMP；
- FITS 数据立方体；
- 同一目录中的多个候选体批处理；
- 一个目录中多个 FITS 的自动选择；
- 仅传入包含 PNG 的目录；
- URL、数据库对象 ID 或远程数据地址。

---

## 9. 命令行使用方法

### 9.1 查看帮助

```powershell
lens-agent --help
lens-agent analyze --help
lens-agent reason --help
```

当前有两个子命令：

| 命令 | 作用 |
|---|---|
| `lens-agent analyze` | 图像测量 + 专家推理的完整流程 |
| `lens-agent reason` | 只从已有 Observation JSON 运行推理 |

### 9.2 `analyze` 参数

```text
lens-agent analyze INPUT
                   [--output OUTPUT]
                   [--flat-output]
                   [--band BAND]
                   [--ring-inner PIXELS]
                   [--ring-outer PIXELS]
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `INPUT` | 必填 | 单个 FITS、单个 PNG 或符合规则的目标文件夹 |
| `--output` | `outputs` | 输出根目录 |
| `--flat-output` | 关闭 | 直接把文件写进输出目录，不再建立对象子目录 |
| `--band` | `VIS` | 要读取的 FITS 波段扩展前缀 |
| `--ring-inner` | `4.0` | 搜索环带内半径，单位为像素 |
| `--ring-outer` | `25.0` | 搜索环带外半径，单位为像素 |

通常不要随意改变环带参数。环带过小可能漏掉较远的弧，过大可能引入大量邻近天体和噪声。参数必须结合图像像素尺度、裁剪尺寸和预期爱因斯坦半径调整。

### 9.3 `reason` 参数

```text
lens-agent reason OBSERVATION_JSON [--output OUTPUT]
```

它只读取 `observation.json`，不读取原始图像或 `vision_artifacts.npz`。这条命令可用于：

- 验证视觉层和推理层确实分离；
- 在规则更新后重新运行推理；
- 用人工或其他测量工具生成的合规 Observation 测试推理；
- 不重复昂贵图像处理的情况下复查结论。

---

## 10. 三个实际运行例子

所有例子都假设当前终端位于：

```text
G:\Astronomy\LensAgent - 副本
```

### 示例一：分析一个目标文件夹

```powershell
lens-agent analyze ".\example\102018667_NEG587746535508057234" --output ".\outputs"
```

程序会自动选择：

```text
example\102018667_NEG587746535508057234\102018667_NEG587746535508057234.fits
```

结果写入：

```text
outputs\102018667_NEG587746535508057234\
```

### 示例二：直接分析单个 FITS，并选择波段和环带

```powershell
lens-agent analyze ".\example\102018667_NEG587746535508057234\102018667_NEG587746535508057234.fits" `
  --band VIS `
  --ring-inner 4 `
  --ring-outer 25 `
  --output ".\outputs"
```

这是默认设置的显式写法，适合记录实验参数。

### 示例三：分析单个 PNG

```powershell
lens-agent analyze ".\example\102018667_NEG587746535508057234\rgb_0.png" `
  --output ".\outputs\png_test"
```

由于输入文件名是 `rgb_0.png`，默认结果目录为：

```text
outputs\png_test\rgb_0\
```

### 附加示例：仅重新运行推理层

```powershell
lens-agent reason ".\outputs\102018667_NEG587746535508057234\observation.json" `
  --output ".\outputs\reasoning_only"
```

### 附加示例：不建立对象子目录

```powershell
lens-agent analyze ".\example\102018667_NEG587746535508057234" `
  --output ".\outputs\single_run" `
  --flat-output
```

使用 `--flat-output` 时，请为每次运行提供独立输出目录，否则同名结果文件可能被后一次运行覆盖。

---

## 11. 一次完整运行会产生什么

默认目录结构为：

```text
outputs/
└── <object_id>/
    ├── observation.json
    ├── observation_summary_cn.md
    ├── vision_artifacts.npz
    ├── vision_diagnostic.png
    ├── vision_field_context.png
    ├── vision_rgb_context.png（存在可匹配RGB时）
    ├── vision_snr_threshold_process.png
    ├── vision_sep_consensus.png
    ├── vision_model_chain.png
    ├── vision_expert_views.png
    ├── vision_foreground_model_comparison.png
    ├── vision_manifest.json
    ├── reasoning.json
    ├── summary.json
    ├── report.md
    └── .matplotlib/
```

### 11.1 `observation.json`

这是视觉层的正式输出，也是视觉层与推理层之间的契约。

它包含：

- 对象和输入来源；
- 输入格式、波段、图像尺寸；
- 中心位置；
- 数据质量；
- 检出的结构；
- 原子测量及可靠性；
- 测量方法和单位；
- 相关证据分组；
- 生成产物路径；
- 已知限制；
- 管线版本和运行参数。

它**不包含最终透镜分类**。

### 11.2 `vision_artifacts.npz`

这是供科研复核和开发调试使用的 NumPy 压缩数组，包含：

- `data`：输入图像；
- `rms`：RMS 图；
- `model`：前景星系模型；
- `residual`：减除残差；
- `residual_snr`：残差信噪比；
- `source_mask`：源掩膜；
- `fit_mask`：模型拟合掩膜；
- `initial_segmentation`：初始分割图；
- `residual_segmentation`：残差结构分割；
- `sep_footprint`：SEP 检测覆盖；
- `residual_structure_mask`：未分类残差结构像素掩膜；
- `arc_mask`：为旧分析脚本保留的兼容别名，内容与上一项相同；
- `threshold_level_map`：多阈值检测层级；
- `nucleus_yx`：中心坐标。
- `comparison_psf_mge_model`：只供审计的PSF-MGE对照前景模型（拟合成功时存在）；
- `comparison_psf_mge_residual`：对照模型减除后的残差（拟合成功时存在）。

它不是普通图片，可用 NumPy 读取：

```python
import numpy as np

artifact = np.load("vision_artifacts.npz")
print(artifact.files)
residual_snr = artifact["residual_snr"]
```

### 11.3 `vision_diagnostic.png`

诊断图包含四个面板：

1. 输入图；
2. 前景星系模型；
3. 残差信噪比图；
4. 经过候选结构审计、已晋级为可测量结构的像素。

青色十字标记系统识别的中心。

检查时重点看：

- 模型是否大体重现中心星系；
- 残差是否是明显弧，而不是中心附近的大块正负残差；
- 晋级结构像素是否落在值得后端继续解释的真实残差上；
- 中心是否定位正确；
- 搜索环带是否合适。

### 11.3.1 视觉处理过程图与清单

v0.4.1 保存与每个样本实际计算有关的处理过程：

- `vision_snr_threshold_process.png`：每个 S/N 阈值的真实掩膜、掩膜残差、相邻层壳和最终存活层级；
- `vision_sep_consensus.png`：SEP 参数网格中稳定出现的结构及存活次数；
- `vision_model_chain.png`：程序先把粘连的原始SEP块按路线局部分割；橙色表示路线外降级留痕，绿色表示路线内原SEP重点候选，紫色表示路线内第二次SEP补充恢复的重点候选；绿色和紫色复核优先级相同但来源不同，颜色不是最终置信度；
- `vision_expert_views.png`：Euclid风格VIS arcsinh/MTF全视场与中央100×100观察图；
- `vision_foreground_model_comparison.png`：比较正式前景模型与PSF-MGE审计模型；它只用于发现“换一种减除方法就消失”的模型敏感结构，不参与正式检测；
- `vision_manifest.json`：输入文件指纹、运行参数、处理步骤、数组含义和过程文件路径。

固定的视觉架构流程不随样本变化，只在 `src/docs/VISION_PROCESS_FLOW_CN.md` 中维护一份，不再复制到每个样本输出目录。

`observation_summary_cn.md` 是由固定模板从 `observation.json` 自动生成的中文解释。它列出实际测量值、结构支持等级、候选结构审计、路线内候选系统和已知限制，不调用 LLM，也不输出透镜分类或概率。科研计算仍应读取 `observation.json`；中文摘要用于快速人工复核。

`observation.json`中的`candidate_structure_audits`是视觉候选与后端推理之间的交接记录。每个候选会得到`promoted`（晋级为可测量结构）、`hold_for_review`（暂缓人工复核）或`trace_only`（只保留追溯）之一，并保存逐项检查、支持事实、风险和理由。只有`promoted`候选会出现在`structures`中；这仍不表示它已经被判为弧、环、多重像或强透镜。

`vision_field_context.png` 用完整FITS视场说明中央裁剪来自哪里，并显示搜索环、得到几何支持的候选编号、中心天球坐标、像素尺度和N/E方向。`vision_rgb_context.png` 会先自动比较RGB与FITS的方向相关性，再把相同标记映射到彩色图；匹配相关性不足时不会生成，避免伪造坐标对应。未解释的原始SEP块只在SEP与模型链过程图中留痕，不会伪装成已采纳的全视场候选标记。

FITS诊断图按WCS原生像素网格使用左下原点显示。PNG保持普通栅格图方向。Manifest会明确记录显示原点、是否重投影和WCS状态。

这些 PNG 是给人看的审计视图，`vision_artifacts.npz` 才是精确数值记录。当前正式管线没有小波去噪，使用的是前景星系减除后的原始残差 S/N；清单中会明确记录这一点。

### 11.4 `reasoning.json`

这是最完整的机器可读推理记录，包括：

- 原始 Observation；
- 观察摘要；
- 四个假设；
- 证据计划；
- 每个假设的证据；
- 每个假设的矛盾项；
- 决策和排名；
- 可信度；
- 审计结果；
- 每个 Agent 的推理轨迹。

研究复现、算法比较和程序调试应优先保存这个文件。

### 11.5 `summary.json`

这是较精简的最终结果，包括：

- `classification`；
- 完整决策摘要；
- 可信度；
- 审计；
- 报告 Markdown 文本。

若要把结果接入后续程序，通常先读取 `summary.json`；若要追查每一步，读取 `reasoning.json`。

### 11.6 `report.md`

这是人类可读的专家报告，通常包含：

- 数据可靠性；
- 结构化观察；
- 竞争假设排名；
- 首选解释的支持证据；
- 各假设反证；
- 最终判断；
- 可信度；
- 审计结果；
- 剩余不确定性。

---

## 12. 如何正确阅读结果

建议按以下顺序，而不是只看第一行分类。

### 第一步：检查 `Audit valid`

- `True`：推理链在结构和引用层面通过检查；
- `False`：不要采用明确分类，先查看 `issues` 和 `recommendation`。

审计通过只说明推理过程内部一致，不代表天体已经得到科学确认。

### 第二步：检查数据质量

重点关注：

- `usable` 是否为真；
- 是否提供真实 RMS；
- 前景模型是否成功；
- `finite_fraction` 是否接近 1；
- `flags` 中是否有减除或无结构警告。

### 第三步：查看诊断图

JSON 数字看起来合理，并不保证像素层面没有错误。诊断图是连接“数字结果”和“真实图像”的关键人工质检步骤。

### 第四步：查看假设排名和分差

不仅看谁第一，还要看：

- 第一名最终分数；
- 第二名是谁；
- 两者差距；
- 哪些替代解释仍然存在。

### 第五步：核对证据引用

例如报告写：

```text
S1 is approximately tangential ...
(refs: M_S1_tangential_misalignment)
```

就可以在 `observation.json` 中找到该测量编号，查看数值、可靠性、方法和相关组。

### 第六步：阅读 Remaining uncertainty

这部分不是客套话，而是决定下一步观测或开发任务的清单。例如：

- 单波段无法检查颜色一致性；
- 尚未检查多核心或潮汐结构；
- 像素几何还没有 WCS 物理标定；
- 多个检测器共享同一残差，证据并不独立。

---

## 13. 三种分类的实际含义

### 13.1 `strong_lens_candidate`

含义：现有测量中，透镜解释在竞争假设中排名第一，通过分数、分差和透镜特异几何门槛。

正确后续动作：

- 人工检查诊断图和原图；
- 获取或分析多波段数据；
- 检查候选像的颜色和形态一致性；
- 进行更精细的前景减除和 PSF 建模；
- 必要时开展光谱或透镜质量模型验证。

错误解读：

```text
系统已经证明它一定是强引力透镜。
```

### 13.2 `not_lens_preferred`

含义：在当前有限测量下，非透镜解释比透镜解释更符合证据。

这不是数学意义上的永久排除。若输入质量差、环带设置不合适或弧被前景减除破坏，真实透镜也可能没有被正确测到。

### 13.3 `indeterminate`

含义：系统认为目前不应强行选择。

常见原因：

- 没有显著环带结构；
- 只有非特异性证据；
- 第一名分数太低；
- 前两名差距太小；
- 数据质量或证据计划完成度不足；
- 推理审计未通过。

主动弃权不是失败，而是安全机制。一个永远给出肯定或否定答案的系统，通常会隐藏真实的不确定性。

---

## 14. 常见问题与排错

### 14.1 提示找不到 `lens-agent`

先检查：

```powershell
Get-Command lens-agent
python -m pip show lens-expert-agent
```

若未安装，在项目目录执行：

```powershell
python -m pip install -r requirements.txt
```

也可临时通过模块入口运行：

```powershell
python -m expert_agent --help
```

### 14.2 VS Code 报 `.venv\Scripts\Activate.ps1` 不存在

这是 VS Code 缓存了已删除的解释器路径。重新选择有效解释器并新建终端即可，通常与项目源码无关。

### 14.3 PowerShell 禁止激活脚本

可只为当前 PowerShell 进程临时允许：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

然后重新激活虚拟环境。该设置只影响当前终端进程。

### 14.4 目录输入提示找到 0 个或多个 FITS

目录模式要求：

- 存在与目录同名的 `.fits`；或
- 根目录恰好只有一个 `.fits`。

否则请直接传入你想分析的 FITS 文件完整路径。

### 14.5 FITS 提示没有二维图像或指定扩展

使用 Astropy 查看 HDU：

```python
from astropy.io import fits

fits.info("target.fits")
```

确认：

- 是否存在 `VIS_FLUX`；
- 或使用 `--band` 指定正确前缀；
- 或主 HDU 是否是二维图像。

### 14.6 输出没有检测到结构

可能原因包括：

- 图中确实没有达到门槛的环带结构；
- 弧位于默认 4–25 像素环带之外；
- 中心定位偏移；
- 弧被前景模型吸收；
- RMS 过大或不准确；
- 弧太弱、太碎或被掩膜；
- 输入 PNG 的保守模式过滤了不稳定结构。

先检查 `vision_diagnostic.png`，再结合已知像素尺度考虑是否调整 `--ring-inner` 和 `--ring-outer`。不要仅为了得到期待结论而反复调参。

### 14.7 PNG 与肉眼看起来很明显，但结果仍然弃权

这是预期的保守行为。渲染 PNG 可能经过拉伸、裁剪、色彩合成或动态范围调整，缺少真实 RMS。系统不会把显示效果当成定量天文测量。

### 14.8 中文终端中角度符号显示异常

报告文件使用 UTF-8。若 PowerShell 或编辑器显示乱码，请把文件编码设置为 UTF-8。优先用 VS Code 打开 `report.md`，不要因为终端编码显示问题误认为数值损坏。

### 14.9 如何检查依赖是否完整

```powershell
python -c "import astropy, photutils, sep; print(astropy.__version__, photutils.__version__, sep.__version__)"
python -m pip check
python -m pytest
```

---

## 15. 科研使用中的注意事项

### 15.1 当前结果属于候选体评估，不是最终确认

科学确认通常还需要更充分的数据和分析，例如：

- 多波段一致性；
- 光谱红移；
- 多像对应关系；
- 透镜质量建模；
- 更高分辨率成像；
- 专家人工复核。

### 15.2 保留完整产物

科研记录至少应保存：

- 原始输入；
- `observation.json`；
- `reasoning.json`；
- `vision_artifacts.npz`；
- `vision_diagnostic.png`；
- 运行命令；
- 软件版本和依赖环境。

只保存最终分类会丢失最有价值的证据链。

### 15.3 参数调整必须留痕

若修改波段或搜索环带，应在实验记录中保留命令。Observation 的 `provenance.parameters` 也会记录实际视觉参数。

### 15.4 避免数据泄漏和选择性报告

不要根据已知标签专门调节某个样本的环带，直到输出想要的结论。应在独立开发集上制定参数，再在未见过的验证集上评价。

### 15.5 评价系统需要正样本和困难负样本

未来校准不能只使用“明显透镜”和“空白图”。困难负样本应包括：

- 环星系；
- 蓝色旋臂；
- 并合和潮汐尾；
- 星系减除失败案例；
- 衍射和成像伪影；
- 弱、碎裂或部分遮挡的真实透镜。

---

## 16. 面向开发者的扩展说明

### 16.1 替换或增加视觉测量工具

可以接入 SourceXtractor++、新的 SEP 配置、PSF 建模或其他天文算法，但输出必须转换为 `ObservationDocument`，并满足：

- 不在视觉层生成分类；
- 每个测量具有唯一 ID；
- 记录单位、方法和可靠性；
- 标记共享数据来源造成的相关性；
- 记录限制和参数；
- 保持 schema 版本可追踪。

### 16.2 接入 LLM

未来可让 LLM 负责自然语言解释、提出补充检查建议或生成更贴近专家表达的报告，但建议采用以下安全边界：

- LLM 只能读取结构化 Observation 和有编号的证据；
- LLM 不能修改原始测量；
- LLM 不能绕过决策门槛；
- LLM 的每个事实性主张必须引用测量或证据 ID；
- 最终输出仍需通过 schema 验证和 ReasoningAuditor；
- LLM 不可把内部评分描述成校准概率。

### 16.3 人工或外部工具生成 Observation

`lens-agent reason` 接受严格的 `ObservationDocument 1.0`。由于 Pydantic 配置为 `extra="forbid"`：

- 缺少必要字段会报错；
- 多余字段会报错；
- 范围不合法会报错；
- schema 版本不匹配会报错。

这种严格性有助于防止接口悄悄漂移。

### 16.4 测试覆盖的核心行为

当前自动测试验证：

- 典型双弧几何可支持 `strong_lens_candidate`；
- 空 Observation 会主动弃权；
- 单个非切向结构不能强行触发透镜判断；
- 跨越 0°/360° 的角度范围计算正确。

运行：

```powershell
python -m pytest
```

在增加新测量或规则时，应同时增加正例、反例、边界条件和退化输入测试。

---

## 17. 推荐的标准工作流程

### 新手日常使用

```text
1. 优先准备带 RMS 的科学 FITS
2. 运行 lens-agent analyze
3. 确认终端没有报错
4. 先看 vision_diagnostic.png
5. 再看 report.md
6. 确认 Audit valid=True
7. 阅读支持证据、竞争假设和剩余不确定性
8. 将结果视为候选体检查意见，而不是最终科学确认
```

### 科研复核

```text
1. 固定软件和依赖版本
2. 记录完整命令与输入校验值
3. 检查 Observation 测量和质量标记
4. 检查 NPZ 中的模型、残差和掩膜
5. 追踪报告证据 ID 到具体测量
6. 检查相关证据是否被重复计算
7. 对比困难替代解释
8. 结合多波段、光谱或透镜建模做外部验证
```

---

## 18. 快速参考表

### 最常用命令

```powershell
# 完整分析
lens-agent analyze "输入路径" --output ".\outputs"

# 仅推理
lens-agent reason ".\outputs\目标名\observation.json" --output ".\outputs\reasoning_only"

# 查看帮助
lens-agent analyze --help

# 检查环境
python -m pip check

# 运行测试
python -m pytest
```

### 最重要的结果文件

| 想做什么 | 看哪个文件 |
|---|---|
| 快速阅读结论 | `report.md` |
| 程序读取最终结果 | `summary.json` |
| 检查视觉测量 | `observation.json` |
| 追查完整推理链 | `reasoning.json` |
| 肉眼检查减除和弧掩膜 | `vision_diagnostic.png` |
| 像素级科研复核 | `vision_artifacts.npz` |

### 结果采用前的最低检查

```text
□ 输入文件和波段正确
□ 数据质量 usable=True
□ 前景模型和中心定位合理
□ 诊断图中的弧掩膜合理
□ Audit valid=True
□ 证据引用可追溯
□ 已阅读替代解释和剩余不确定性
□ 没有把 confidence 当作真实概率
```

---

## 19. 最后的理解框架

这版智能体最值得强调的能力，不是它能输出一个标签，而是它把“看见什么”和“如何判断”拆成了两套可检查的记录：

```text
图像不是结论；图像产生观察。
观察不是结论；观察支持或反对假设。
最高分不是绝对真理；还要检查分差、数据质量和反证。
可信度不是概率；它描述当前证据链有多完整、稳定和可追溯。
证据不足时不下结论，是科学推理的一部分。
```

因此，Lens Expert Agent 当前最准确的定位是：

> 一个面向强引力透镜候选检验的、可解释、可追溯、会保留竞争解释并能主动弃权的最小可运行专家推理系统。

它已经建立了“测量事实 → 竞争假设 → 证据与反证 → 判断 → 可信度 → 审计 → 报告”的完整骨架；下一阶段的重点，应是在保持这条证据边界的前提下，提升多波段、PSF、并合形态和真实数据校准能力，并谨慎接入受约束的 LLM 解释层。
