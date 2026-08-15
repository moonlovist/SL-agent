from __future__ import annotations

from pathlib import Path

from ..schemas.observation import ObservationDocument, StructureObservation


def _value(value: float | None, digits: int = 2) -> str:
    return "未测得" if value is None else f"{value:.{digits}f}"


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "未知"
    return "是" if value else "否"


def _foreground_shape(ellipticity: float | None) -> str:
    if ellipticity is None:
        return "形态未能可靠测量"
    if ellipticity <= 0.2:
        return "较接近圆形"
    if ellipticity <= 0.45:
        return "具有中等椭率"
    return "明显拉长"


def _translate_limitation(text: str) -> str:
    translations = {
        "All residual detectors share the same galaxy-subtracted image and are correlated.": "所有残差检测器共享同一张前景星系减除图，因此这些检测结果彼此相关，并非完全独立证据。",
        "Angular measurements are in pixel geometry unless a validated pixel scale is supplied.": "在没有经过验证的像素尺度时，角度和距离测量基于像素几何。",
        "A single-band run cannot test source-color consistency.": "单波段数据不能检查不同结构的颜色一致性。",
        "No validated PSF model is supplied; PSF-specific artifact rejection is unavailable.": "没有提供经过验证的 PSF 模型，因此不能进行 PSF 专项伪影排除。",
        "Disk connectivity, multiple-core count, and residual asymmetry are screening proxies rather than calibrated morphology classifiers.": "盘连接性、多核心数量和残差不对称性只是筛查指标，不是经过标定的形态分类器。",
        "PNG noise was estimated from image pixels; no supplied RMS map was available.": "PNG 输入没有自带 RMS 噪声图，噪声由图像像素估计。",
        "No validated celestial WCS/pixel scale is available; angular measurements remain in pixel geometry.": "没有可用的天球 WCS 或像素尺度，因此角距离仍只能使用像素表示。",
        "No PSF image is present in the selected band; PSF-specific artifact rejection is unavailable.": "所选波段没有 PSF 图像，因此不能进行 PSF 专项伪影排除。",
        "A PSF image is present and preserved, but PSF-convolved foreground modeling is not yet implemented.": "FITS中存在PSF图像并已保存，但当前尚未实现结合PSF卷积的前景光建模。",
    }
    return translations.get(text, text)


def _quick_summary(document: ObservationDocument) -> str:
    foreground = document.foreground_galaxy
    shape = _foreground_shape(
        None if foreground is None else foreground.ellipticity
    )
    noise = (
        "带有输入数据提供的 RMS 噪声信息"
        if document.data_quality.supplied_rms
        else "没有输入数据提供的 RMS 噪声图"
    )
    model = (
        f"前景光模型 `{document.data_quality.model_method}` 成功完成"
        if document.data_quality.model_success
        else f"精细模型拟合失败，因此使用 `{document.data_quality.model_method}` 备用结果"
    )
    opening = (
        f"这是一张 {document.image_shape[0]}×{document.image_shape[1]} 像素的 "
        f"{document.band} 波段 {document.input_format.upper()} 图像，{noise}。"
        f"数据可用于视觉测量，程序在中心测得一个{shape}的亮度集中结构；{model}。"
    )
    if not document.structures:
        structure_text = "减除中心光后，没有结构进入最终测量列表。"
    elif len(document.structures) == 1:
        item = document.structures[0]
        structure_text = (
            f"减除中心光后，程序在距离中心约 {item.radius_px:.2f} 像素"
            f"（{_value(item.radius_arcsec)}角秒）处测得1个由 "
            f"{item.pixel_count} 个像素组成的结构：累计信噪比 {item.integrated_snr:.2f}，"
            f"环向覆盖 {item.angular_extent_deg:.2f}°，切向偏差 "
            f"{_value(item.tangential_misalignment_deg)}°，S/N 多阈值存活比例 "
            f"{_value(item.threshold_survival_fraction)}，SEP 参数存活比例 "
            f"{_value(item.sep_survival_fraction)}。"
        )
    else:
        radii = [item.radius_px for item in document.structures]
        structure_text = (
            f"减除中心光后，程序测得 {len(document.structures)} 个结构，"
            f"它们距离中心约 {min(radii):.2f}–{max(radii):.2f} 像素；"
            f"场景总环向覆盖为 "
            f"{_value(None if document.scene_geometry is None else document.scene_geometry.total_angular_coverage_deg)}°。"
        )
    return (
        opening
        + structure_text
        + "这些是视觉测量结果，不等同于透镜弧或多重像；当前摘要不判断该对象是否为强引力透镜。"
    )


def _structure_section(
    structure: StructureObservation,
    threshold_runs: int,
    sep_runs: int,
) -> list[str]:
    threshold_count = (
        None
        if structure.threshold_survival_fraction is None
        else round(structure.threshold_survival_fraction * threshold_runs)
    )
    sep_count = (
        None
        if structure.sep_survival_fraction is None
        else round(structure.sep_survival_fraction * sep_runs)
    )
    return [
        f"### {structure.id}",
        "",
        f"- 结构中心位置：({_value(structure.centroid_xy[0])}, {_value(structure.centroid_xy[1])}) 像素",
        f"- 距离图像中心：{_value(structure.radius_px)} 像素",
        f"- 天空角距离：{_value(structure.radius_arcsec)} 角秒",
        f"- 天球坐标：RA={_value(structure.centroid_ra_deg, 6)}°，Dec={_value(structure.centroid_dec_deg, 6)}°",
        f"- 天空位置角：{_value(structure.sky_position_angle_deg)}°（从北向东计）",
        f"- 参与测量的像素数：{structure.pixel_count}",
        f"- 累计信噪比：{_value(structure.integrated_snr)}（不是透镜概率）",
        f"- 环向覆盖角度：{_value(structure.angular_extent_deg)}°",
        f"- 轴比：{_value(structure.axis_ratio)}",
        f"- 与理想切向方向的偏差：{_value(structure.tangential_misalignment_deg)}°",
        (
            f"- S/N 多阈值稳定性：{threshold_count}/{threshold_runs}，"
            f"存活比例 {_value(structure.threshold_survival_fraction)}"
            if threshold_count is not None and threshold_runs
            else "- S/N 多阈值稳定性：未测得"
        ),
        (
            f"- SEP 参数稳定性：{sep_count}/{sep_runs}，"
            f"存活比例 {_value(structure.sep_survival_fraction)}"
            if sep_count is not None and sep_runs
            else "- SEP 参数稳定性：未测得"
        ),
        f"- 是否与中心区域明显连接：{_yes_no(structure.connects_to_inner_region)}",
        f"- 是否位于减除敏感的近中心区域：{_yes_no(structure.near_core)}",
        "",
    ]


def render_observation_summary_cn(document: ObservationDocument) -> str:
    """Render a deterministic Chinese explanation without adding a lens decision."""
    parameters = document.provenance.parameters
    thresholds = parameters.get("snr_thresholds", [])
    sep_nthresh = parameters.get("sep_nthresh", [])
    sep_deblend = parameters.get("sep_deblend_cont", [])
    threshold_runs = len(thresholds)
    sep_runs = len(sep_nthresh) * len(sep_deblend)
    foreground = document.foreground_galaxy
    geometry = document.scene_geometry
    alternatives = document.alternative_indicators

    lines = [
        f"# 视觉观察中文摘要：{document.object_id}",
        "",
        "> 本文件由固定模板根据 `observation.json` 自动生成，不调用 LLM，"
        "不增加原始测量中不存在的事实，也不输出透镜分类。",
        "",
        "## 快速概括",
        "",
        _quick_summary(document),
        "",
        "## 1. 输入数据",
        "",
        f"- 输入文件：`{document.input_path}`",
        f"- 输入格式：{document.input_format.upper()}",
        f"- 波段：{document.band}",
        f"- 图像尺寸：{document.image_shape[0]} × {document.image_shape[1]} 像素",
        f"- 图像中心：({_value(document.nucleus_xy[0])}, {_value(document.nucleus_xy[1])})",
        f"- 是否带有输入数据提供的 RMS 噪声图：{_yes_no(document.data_quality.supplied_rms)}",
        f"- 有限像素比例：{document.data_quality.finite_fraction:.4f}",
        f"- 数据是否可用于视觉测量：{_yes_no(document.data_quality.usable)}",
        f"- 天球 WCS 是否有效：{_yes_no(None if document.astrometry is None else document.astrometry.wcs_valid)}",
        f"- 像素尺度：{_value(None if document.astrometry is None else document.astrometry.pixel_scale_arcsec, 4)} 角秒/像素",
        f"- 中心天球坐标：RA={_value(None if document.astrometry is None else document.astrometry.center_ra_deg, 6)}°，Dec={_value(None if document.astrometry is None else document.astrometry.center_dec_deg, 6)}°",
        "",
        "## 2. 中心亮源与前景光减除",
        "",
    ]
    if foreground is None:
        lines.append("当前 Observation 没有记录中心亮源的详细测量。")
    else:
        lines.extend(
            [
                f"程序是否测得中心亮度集中结构：{_yes_no(foreground.detected)}。",
                "",
                f"其测量椭率为 {_value(foreground.ellipticity, 3)}，"
                f"因此按本摘要的描述规则记为“{_foreground_shape(foreground.ellipticity)}”。",
                "",
            ]
        )
    if document.data_quality.model_success:
        lines.append(
            f"前景光模型 `{document.data_quality.model_method}` 成功完成，"
            "其减除结果仍需结合诊断图进行人工检查。"
        )
    else:
        lines.append(
            f"精细模型未成功完成，本次使用 `{document.data_quality.model_method}`。"
            "这表示程序使用了备用处理结果，减除残差应当优先人工复核。"
        )
    lines.extend(
        [
            "",
            f"- 残差中位数：{document.data_quality.residual_median:.6g}",
            f"- 残差稳健 RMS：{document.data_quality.residual_robust_rms:.6g}",
            f"- 质量标记：{', '.join(document.data_quality.flags) or '无'}",
            "",
            "## 3. 检测到的残差结构",
            "",
            f"程序在配置的中心环带内检测到 **{len(document.structures)}** 个满足当前筛选条件的结构。",
            "这些结构是测量对象，不等同于透镜弧或多重像。",
            "",
        ]
    )
    if document.structures:
        for structure in document.structures:
            lines.extend(
                _structure_section(structure, threshold_runs, sep_runs)
            )
    else:
        lines.extend(["没有结构进入最终测量列表。", ""])

    lines.extend(["## 4. 场景整体信息", ""])
    if geometry is not None:
        lines.extend(
            [
                f"- 结构数量：{geometry.structure_count}",
                f"- 多结构共同半径离散程度：{_value(geometry.common_radius_cv, 3)}（越低表示半径越一致）",
                f"- 总环向覆盖：{_value(geometry.total_angular_coverage_deg)}°",
                f"- 平均切向偏差：{_value(geometry.mean_tangential_misalignment_deg)}°",
                f"- 满足当前切向标准的结构比例：{_value(geometry.tangential_alignment_fraction)}",
            ]
        )
    else:
        lines.append("当前 Observation 没有场景整体几何记录。")
    if alternatives is not None:
        lines.extend(
            [
                f"- 与中心区域连接的结构数：{alternatives.inner_connected_structure_count}",
                f"- 次级中心亮源数量：{alternatives.multiple_core_count if alternatives.multiple_core_count is not None else '未测得'}",
                f"- 是否具有已验证的 PSF 信息：{_yes_no(alternatives.psf_information_available)}",
            ]
        )

    lines.extend(
        [
            "",
            "## 5. 已知限制",
            "",
            *[f"- {_translate_limitation(item)}" for item in document.limitations],
            "",
            "## 6. 应当怎样理解这份结果",
            "",
            "本摘要说明视觉管线测量到了什么，以及哪些测量可能需要优先复核。"
            "它不比较强透镜、星系自身结构、并合或处理伪影等解释，"
            "也不输出‘透镜/非透镜’、候选等级或概率。",
            "",
            "请将本摘要与四张视觉过程图一起查看；最终科学判断仍需后续证据比较和人工专家复核。",
            "",
        ]
    )
    return "\n".join(lines)


def write_observation_summary_cn(
    document: ObservationDocument, output: Path
) -> Path:
    path = output / "observation_summary_cn.md"
    path.write_text(render_observation_summary_cn(document), encoding="utf-8")
    return path
