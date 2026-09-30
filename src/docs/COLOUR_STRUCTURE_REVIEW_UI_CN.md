# 颜色结构人工标注器

这是一个与 LensAgent 评级流程隔离的标定工具。它只读取 FITS/RGB，保存人工描绘的前景和结构掩膜及备注，不执行前景减除、颜色评级或 ABC/unknown 决策。

## 启动

```powershell
$env:PYTHONPATH = "src"
python scripts\review_colour_structures.py `
  "G:\Astronomy\New data\颜色样本第二批-多独立像" `
  --output "outputs_colour_structure_annotations_batch2_20260923"
```

## 标注约定

1. `Foreground reference` 层可选：只在你能确定前景参照时绘制。
2. `Structures` 是整数标签层。S1、S2…是不同结构。
3. `角色`只有四种，彼此互斥：
   - `独立像候选`：与其他结构空间断开、可能是另一幅透镜像；无论紧致或拉长都选这一项。
   - `单弧／环整体`：一条连续弧、半环或近环。若噪声把同一条弧切断，所有碎片都画成同一个 S 标签，仍选这一项。
   - `无关源／污染`：邻星、邻近星系、PSF 残差或明显噪声。
   - `无法确定`：看得见，但无法可靠归入前三类。
4. 真正分离的像使用不同 S 标签。只有当不同 S 结构可能来自同一个背景源时，才填相同的“可能同源组”，如 `SRC1`；不确定就留空。
5. 角色是人工形态记录，不等同于已经确认物理身份，也不会影响 LensAgent 的评级。

## 显示拉伸

右侧可在 `Linear / Asinh弱 / Asinh强 / Log / Sqrt / MTF` 之间切换，并调整黑点和白点百分位。这些选项只改变屏幕显示，不修改FITS像素、标注掩膜或后续测光数据。暗弧优先试 `Asinh强`或`Log`；亮核严重饱和时可提高白点百分位。

## 输出

每个对象保存：

- `annotation.json`：样本状态、每个结构的角色、同源组、备注、像素数和哈希；
- `masks.fits`：`FOREGROUND` 二值层和 `STRUCTURES` 整数标签层；
- `review_screenshot.png`：保存时的界面快照。

输出根目录的 `annotation_index.json` 用于中断续标和后续颜色测光。原始数据目录不会被写入。

## 不确定结构颜色复核模式

先由测量表生成复核清单，再用原标注目录启动。该模式只显示至少有3个共同可靠波段的“不确定结构—同场候选结构”比较：

```powershell
python scripts\prepare_unresolved_colour_pair_review.py `
  "outputs_colour_geometry_calibration_batch2_20260924\structures.csv" `
  "outputs_colour_geometry_calibration_batch2_20260924\unresolved_pair_review_manifest.json"

$env:PYTHONPATH = "src"
python scripts\review_colour_structures.py `
  "G:\Astronomy\New data\颜色样本第二批-多独立像" `
  --output "outputs_colour_structure_annotations_batch2_20260923" `
  --pair-review-manifest "outputs_colour_geometry_calibration_batch2_20260924\unresolved_pair_review_manifest.json"
```

像对复核分别记录几何上是否可能同源、肉眼颜色是否一致、污染风险和两个掩膜是否覆盖对应物理区域。界面显示的颜色差和显著度只用于提示，不自动改变标注、阈值或等级。

当前下拉框选中的第一块结构会以青色高亮，第二块以洋红色高亮；图层名称同步显示对应的S编号。一个对象含有多个结构时，切换“待复核像对”即可查看不同组合，不需要记忆编号位置。

复核模式中的“保存并下一个像对”会先遍历当前对象的全部待复核组合，再进入下一个对象；重新启动时会自动定位到第一组尚未复核的像对。
