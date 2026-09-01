# LensAgent 数据运行新任务交接 Prompt

> 用途：把本文件中“可复制 Prompt”一节完整复制到新的 Codex 任务中，再补充本次
> 输入目录、输出目录和样本范围。
>
> 基线版本：LensAgent `0.15.0`，Git commit `a1aaa1e`

## 使用前需要填写

```text
输入根目录：<由用户填写>
输出根目录：<由用户填写，必须使用新的空目录>
样本范围：<例如全部样本、每级前20个、指定编号>
运行方式：<完整 analyze / 只运行已有 Observation 的 reason>
已知标签文件：<如有 expert_score 表格则填写；没有就写“无”>
资源或停止条件：<如有则填写；没有就写“发生异常立即停止”>
```

---

## 可复制 Prompt

你现在接手的是强引力透镜专家推理项目 LensAgent 的数据运行与标定数据整理任务。
请先完整阅读本 Prompt，再进行任何操作。除非我在新任务中明确改变要求，否则以
这里的边界为准。

### 一、项目目标和当前阶段

本项目不是普通的图片二分类器，而是模拟天文学专家的检查过程：

```text
图像测量事实
→ 多种可能解释
→ 支持证据与反证
→ 证据审计
→ expert_score
→ Euclid Q1 等级
```

视觉层回答“图中测到了什么”，不能直接判断透镜。后端同时维护：

- H1：强引力透镜；
- H2：前景星系自身结构；
- H3：星系并合或相互作用；
- H4：噪声、PSF或前景减除伪影。

当前基线为 LensAgent `0.15.0`，Git commit `a1aaa1e`。v0.14.0 的视觉层已经暂时
冻结；v0.15.0 删除了未经Euclid Q1数据标定的旧假设分数、旧
`confidence.score`和旧阈值判决。

因此当前正常输出应当是：

```json
{
  "grading_scheme": "euclid_q1_paper",
  "status": "calibration_pending",
  "expert_score": null,
  "grade": null,
  "a_plus": null
}
```

这不是程序失败，而是科研边界：在标定完成前禁止制造临时分数或等级。

### 二、本次任务

本次只负责：

1. 核对环境和项目基线；
2. 先运行一个代表样本作为试运行；
3. 检查输出目录、文件和状态是否正常；
4. 再按我给出的范围顺序批量运行；
5. 保存每个样本的完整可追溯产物；
6. 汇总成功、失败、跳过和不可评级样本；
7. 若存在真实`expert_score`标签，建立“输入标签—运行结果”的独立索引，但绝不
   把标签提前传入Agent推理过程。

### 三、严格禁止事项

批量运行期间不得：

- 修改视觉层参数、SEP参数、S/N阈值、椭圆路线或前景模型；
- 修改推理规则、审计规则或Q1等级映射；
- 恢复已删除的旧分数、旧confidence或旧Decision链；
- 因为某个样本结果不好就边跑边调参；
- 把A/B/C文件夹名称或已知`expert_score`输入给Agent作为推理证据；
- 把`unsuccess`文件夹当作非透镜/X真值；该目录表示建模或处理未成功，不能据此
  推断天体一定不是强透镜；
- 把`calibration_pending`改写成C、X、低可信度或程序失败；
- 把`insufficient_data`解释成非透镜；
- 覆盖已有输出目录、删除历史结果或把大批数据/outputs提交到Git；
- 在没有独立验证集的情况下宣称准确率提高。

如果发现必须修改代码才能继续，立即停止批量运行，报告具体错误、受影响样本和
最小复现方式，等待用户单独授权。不要在数据运行任务中顺手修代码。

### 四、开始前检查

在项目根目录执行只读检查：

```powershell
git status --short
git log -1 --oneline
python -m pip show lens-expert-agent
lens-agent --help
```

期望看到：

- 最新提交为`a1aaa1e`或它之后经用户确认的新提交；
- 安装版本为`0.15.0`或之后经确认的版本；
- 现有未跟踪的outputs、`.tmp`和数据目录属于历史产物，不得擅自删除、移动或提交。

若可编辑安装元数据不是当前版本，但源码基线正确，可以执行：

```powershell
python -m pip install -e . --no-deps
```

不要创建或替换`.venv`，不要无理由升级依赖。

### 五、试运行

完整流程命令：

```powershell
lens-agent analyze "<一个样本文件或样本文件夹>" --output "<新的输出根目录>"
```

只对已有Observation运行后端：

```powershell
lens-agent reason "<observation.json>" --output "<新的输出目录>"
```

试运行后必须检查：

- `observation.json`能正常读取；
- `reasoning.json`、`summary.json`和`report.md`存在；
- 完整运行时视觉诊断图和`vision_manifest.json`存在；
- `summary.json.grading.status`为`calibration_pending`、`insufficient_data`或
  `blocked_by_audit`之一；
- 未出现旧字段`confidence`、旧假设总分或旧`decision_audit`；
- `expert_score`和`grade`在未标定阶段必须为`null`。

先把试运行位置和检查结果告诉用户；如果用户已明确要求无需再次确认，则可继续
批量运行。

### 六、批量运行原则

1. 使用全新的输出根目录，例如：

   ```text
   outputs_v0150_<数据集名称>_<日期>
   ```

2. 每个样本独立运行，不把多个目标写入同一个样本目录。
3. 默认顺序运行，避免多个天文图像任务同时抢占大量内存。
4. 每个样本结束后记录：样本ID、输入路径、输出路径、成功/失败状态、
   `grading.status`、错误摘要。
5. 已成功写完且输入指纹一致的样本可以在恢复任务时跳过；不得仅凭目录存在就认定
   已完成。
6. 某个样本损坏时记录失败并根据用户要求继续或停止；若出现连续异常、磁盘空间
   异常、Python进程异常或系统明显卡顿，立即停止。
7. 用户给出Codex用量停止线时严格遵守，不要为了跑完而越过限制。

样本文件夹逐个运行的PowerShell结构可以是：

```powershell
$sampleRoot = "<输入根目录>"
$resultRoot = "<新输出根目录>"
Get-ChildItem -LiteralPath $sampleRoot -Directory | ForEach-Object {
    lens-agent analyze $_.FullName --output $resultRoot
    if ($LASTEXITCODE -ne 0) {
        throw "LensAgent failed for $($_.FullName)"
    }
}
```

实际执行前必须先确认输入层级确实是“一份目标一个子文件夹”。如果不是，应根据
真实目录结构调整遍历对象，但不能修改科学程序。

### 七、必须保存的产物

完整运行时，每个样本至少保留：

- `observation.json`；
- `observation_summary_cn.md`；
- `vision_manifest.json`；
- `vision_artifacts.npz`；
- `vision_diagnostic.png`；
- `vision_snr_threshold_process.png`；
- `vision_sep_consensus.png`；
- `vision_model_chain.png`；
- `vision_signal_audit.png`；
- 该样本实际生成的其他视觉定位和对照图；
- `reasoning.json`；
- `summary.json`；
- `report.md`。

不要因为暂时看不懂NPZ二进制文件就删除它，它保存像素级溯源数据。

### 八、标签与标定数据的隔离

真实标签应放在运行结果之外的独立表中，建议字段为：

```text
sample_id
input_path
expert_score
paper_grade
label_source
expert_structure_annotation_available
expert_support_reason
expert_competing_explanation
split
```

其中：

- `expert_score`是主要连续标签；
- `paper_grade`是由项目选定的论文口径映射得到的离散字段；
- `label_source`必须说明来自论文目录、Zenodo字段、专家复核还是人工整理；
- `split`必须区分`calibration`、`validation`和可选的`test`；
- 同一目标的重复切图不能跨越不同数据划分，防止数据泄漏；
- 不允许使用validation/test样本调整阈值或权重。

当前项目选定的Q1论文口径是：

```text
A：expert_score > 2.0
B：1.5 < expert_score <= 2.0
C：expert_score <= 1.5
```

Q1论文没有规定`expert_score <= 1.0`自动映射为X的目录分箱，因此不要自行制造
X边界。A+是A上的独立科学价值标记，也不能由分数自动生成。

如果只有A/B/C目录而没有连续`expert_score`，必须在汇总中明确写出：这些数据可
用于初步等级研究，但不足以严格重建论文的连续专家分数。

### 九、运行结束后的汇报格式

最终向用户报告：

```text
项目版本与Git提交：
输入目录：
输出目录：
计划样本数：
成功数：
失败数：
跳过数：
calibration_pending数：
insufficient_data数：
blocked_by_audit数：
失败样本与原因：
是否修改源码：否
是否覆盖或删除历史结果：否
```

如果拥有标签，可以报告运行覆盖率和各状态分布，但在正式量化审计与评分器完成前，
不得报告Agent的A/B/C准确率，因为当前版本根本没有产生正式等级。

### 十、推理纪律

对复杂事实明确区分：

- 已由文件或运行结果验证的事实；
- 根据结果作出的合理推断；
- 尚未验证的假设；
- 缺失信息。

不编造论文结论、样本真值、专家意见、准确率、阈值或输出文件。遇到标签含义不明
时先查元数据和来源，无法确认就明确写“未知”，不要用文件夹名称猜测。

### 十一、本次用户填写的信息

```text
输入根目录：<填写>
输出根目录：<填写>
样本范围：<填写>
运行方式：<填写>
已知标签文件：<填写>
资源或停止条件：<填写>
```

请从基线检查开始，先做一个样本试运行；除非用户在本任务中明确允许，否则不要
修改任何源码、参数、规则或已有文件。

---

## 给当前用户的说明

这份Prompt专门服务于“跑数据和整理标定输入”，不是下一阶段评分规则开发Prompt。
等数据运行结束后，应另开一个任务，根据标定集建立量化审计表；不要让批量运行任务
同时承担调参和评分开发，否则容易产生数据泄漏与边跑边迎合标签的问题。
