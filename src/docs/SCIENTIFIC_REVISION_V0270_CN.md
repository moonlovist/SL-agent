# v0.27.0 Human-in-the-loop 前景选择

本版允许人在前景拟合前覆盖自动选择，也允许从已有结果重新打开标注并完整重跑。

标注窗口始终工作在原始 FITS/PNG 的 NumPy 像素网格，提供三类输入：唯一前景中心点、确定属于前景的种子掩膜、禁止进入前景拟合的 arc/环/源像掩膜。自动选择的 2–4 个候选、自动前景模型和自动残差仅供显示参考。

人工掩膜不是透镜标签，也不直接进入 A/B/C/X 规则。人工中心替换自动路由中心；前景种子用于前景几何与稳健拟合约束；禁止拟合掩膜并入前景拟合掩膜。之后必须重新进行前景建模、残差计算、结构检测、颜色测量和唯一一次专家判断。

每次接受的标注保存为 `foreground_override.json` 与 `foreground_override_masks.fits`。JSON 记录原始输入 SHA-256、零基 XY 中心、坐标约定、掩膜 HDU、时间和人工来源；FITS 保存 `FOREGROUND` 与 `FITEXCL` 两层，并在可用时保留 WCS。输入哈希或数组形状不一致时拒绝重放，避免把一个对象的人工掩膜错用于另一个对象。

运行入口：

```powershell
lens-agent analyze INPUT --interactive-foreground --output OUTPUT_ROOT
lens-agent foreground-review OLD_OBJECT_OUTPUT --output NEW_OUTPUT_ROOT
lens-agent analyze INPUT --foreground-override foreground_override.json --output OUTPUT_ROOT
```

napari/Qt 作为 `ui` 可选依赖；普通批处理和无界面测试不强制加载 GUI。
