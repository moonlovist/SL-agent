# LensAgent v0.30.0 交付说明

## 交付定位

v0.30.0 是一个可安装、可运行、可追溯的完整代码快照，不声称已完成 Euclid Q1 等级标定。当前 ABC/unknown 判定仍是可调试的单专家模拟输出；规则准确率和与 Euclid 专家结果的一致性需继续用带标注样本标定。

交付包含：

- FITS/PNG 读取、VIS 及可用多波段测量；
- 前景中心选择、PSF 卷积前景光模型、残差与 S/N 处理；
- SEP、多阈值、几何路径搜索、候选交接与结构审计；
- 颜色、几何、伪影和前景模型敏感性的可追溯测量；
- Observation JSON、规则推理、DecisionAudit、报告和诊断图；
- 桌面批处理 UI、前景 mask/中心人工校正 UI、颜色结构复核 UI；
- 校准/回归脚本、中英文技术文档和自动测试。

## 安装与启动

需要 Python 3.11 或更高版本。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ui]"
```

启动桌面 UI：

```powershell
lens-agent ui --project lensagent_ui_outputs
```

启动独立的颜色/几何结构标注 UI：

```powershell
lens-agent-structure-review "<数据集根目录>" --output outputs_colour_structure_annotations
```

命令行分析单个 FITS 文件或一个样本目录：

```powershell
lens-agent analyze "<FITS 文件或样本目录>" --output outputs
```

开启前景中心/mask 人工校正：

```powershell
lens-agent analyze "<样本>" --interactive-foreground --output outputs
```

仅用已有 Observation 重放规则层：

```powershell
lens-agent reason "outputs/<object-id>/observation.json" --output outputs/reasoning_only
```

## 架构与规则边界

主链为：

```text
输入数据 -> 确定性视觉测量 -> Observation 1.18
         -> ExpertAssessmentAgent -> DecisionAudit -> ABC/unknown + 追溯报告
```

视觉层负责“量到了什么”，不直接宣布是否为透镜。规则层的主要科学逻辑仍应是：数据可用性 -> 中心/前景偏折体 -> 弧、环或多像候选 -> 同心性、切向性和构型联合 -> 同源颜色关系 -> 普通星系/并合/投影/伪影等竞争解释 -> 等级。

可观测性、几何边界和保护集输出当前保留为诊断/标定信息，不应被解释为已经科学标定的 Euclid 判据。不应用单个 S/N、宽度、半径或颜色阈值单独生成 A 级；等级必须来自多项联合证据和竞争解释的整体判断。

## 验收方法

```powershell
python -m pytest -q
python -m expert_agent --help
lens-agent-ui --help
```

生成本快照时，全量自动测试为 `184 passed`。该结果证明已覆盖的程序行为在当前环境中可运行，不等于证明科学分级已完成标定。
