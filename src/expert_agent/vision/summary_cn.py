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
        "Raw SEP detections remain trace-only; only geometry-chain candidates are exposed, and none are promoted to scored structures before calibration.": "原始 SEP 检测只用于过程追溯；程序只对外记录经过几何链整理的候选，并且在完成标定审计前不会把任何候选送入当前评分。",
        "Geometric paths only guide masked SEP searches; unfilled path regions are not treated as rings or missing evidence against lensing.": "几何路径只负责限制 SEP 再次搜索的位置；路径上没有亮像素的部分既不算环，也不能据此反对透镜解释。",
        "A PSF image is present and used by the audit-only comparison model, but the formal detection subtraction does not yet use PSF convolution.": "输入中有 PSF 图像，且审计用的比较模型使用了它；但当前主检测通道的前景减除尚未使用 PSF 卷积。",
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
        structure_text = "减除中心光后，没有候选进入当前后端评分使用的正式结构列表。"
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
        + (
            f"视觉链另外保留了 {len(document.model_candidates)} 个待审计候选。"
            if document.model_candidates else "视觉链没有保留待审计候选。"
        )
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
    model_comparison = document.foreground_model_comparison

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
    if model_comparison is not None:
        lines.extend(
            [
                "",
                "### 前景星系两种擦除方法的比较",
                "",
                f"- 第二种PSF-MGE比较模型是否成功：{_yes_no(model_comparison.comparison_success)}",
                f"- 当前模型残差RMS：{_value(model_comparison.primary_residual_robust_rms, 6)}",
                f"- PSF-MGE残差RMS：{_value(model_comparison.comparison_residual_robust_rms, 6)}",
                f"- 模型链候选像素换一种擦法后仍超过同一S/N门槛的比例：{_value(model_comparison.measured_structure_pixel_survival_fraction, 3)}",
                f"- 两种残差在搜索环内的相关程度：{_value(model_comparison.residual_correlation, 3)}",
                "- 重要：第二种模型当前只用于暴露模型敏感性，没有参与结构检测或最终判断。",
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
            "## 3. 当前评分使用的正式结构",
            "",
            f"当前共有 **{len(document.structures)}** 个结构进入后端证据评分。",
            "v0.10 中未经标定的模型链候选不会自动进入这里。",
            "",
        ]
    )
    if document.structures:
        for structure in document.structures:
            lines.extend(
                _structure_section(structure, threshold_runs, sep_runs)
            )
    else:
        lines.extend(["没有候选进入当前评分使用的正式结构列表。", ""])

    lines.extend(["## 3.1 SEP→几何模型链候选", ""])
    lines.extend(
        [
            "SEP 先在全图提出原始线索；圆或椭圆只规定下一次应该到哪里继续找；程序随后把路线外像素遮住，在路线内再次运行 SEP。",
            "下列候选仍需要前景减除、PSF、稳定性和人工标注等审计，本版不会用它们改变 A/B/C 结果。",
            "",
        ]
    )
    if document.model_candidates:
        for item in document.model_candidates:
            origin = "全图SEP稳定线索" if item.origin == "sep_seed" else "几何路线内SEP新检出"
            status = {
                "geometry_linked": "已与几何路线关联",
                "geometry_recovered": "沿几何路线恢复",
                "unexplained_residual": "尚未得到几何支持的残差线索",
            }[item.status]
            lines.extend(
                [
                    f"### {item.id}",
                    "",
                    f"- 来源：{origin}；状态：{status}",
                    f"- 中心：({_value(item.centroid_xy[0])}, {_value(item.centroid_xy[1])}) 像素；距中心：{_value(item.radius_px)} 像素",
                    f"- 像素数：{item.pixel_count}；累计S/N：{_value(item.integrated_snr)}；环向跨度：{_value(item.angular_extent_deg)}°",
                    f"- 关联路线：{', '.join(item.path_ids) or '无'}；是否位于前景减除敏感的近中心区域：{_yes_no(item.near_core)}",
                    f"- 换用PSF-MGE比较残差后仍超过门槛的像素比例：{_value(item.foreground_comparison_survival_fraction, 3)}",
                    "- 解释限制：这是待审计的结构候选，不是弧、环、多重像或透镜标签。",
                    "",
                ]
            )
    else:
        lines.extend(["模型链没有保留待审计候选。", ""])

    lines.extend(["## 3.2 几何路径引导搜索", ""])
    if document.geometric_paths:
        lines.extend(
            [
                "这里的圆/椭圆只是一条搜索路线：优先由已检测结构提供线索，在路线附近寻找原先漏掉的真实亮像素。路线空白部分不算结构，也不等同于爱因斯坦环或强透镜。",
                "",
            ]
        )
        for item in document.geometric_paths:
            lines.extend(
                [
                    f"### {item.id}",
                    "",
                    f"- 启动方式：{item.mode}；种子候选：{', '.join(item.seed_candidate_ids) or '无（中心后备）'}",
                    f"- 路线中心：({_value(item.center_xy[0])}, {_value(item.center_xy[1])}) 像素；半长轴：{_value(item.semi_major_radius_px)} 像素；轴比：{_value(item.axis_ratio, 3)}",
                    f"- 搜索半宽：{_value(item.half_width_px)} 像素；种子拟合离散：{_value(item.fit_scatter_px)} 像素",
                    f"- 路线中位 S/N：{_value(item.median_on_path_snr, 3)}；相对邻近路线的 S/N 差：{_value(item.on_minus_off_path_snr, 3)}",
                    f"- 已关联候选：{', '.join(item.linked_candidate_ids) or '无'}；沿路线新找到的候选：{', '.join(item.recovered_candidate_ids) or '无'}",
                    "- 路线只提供搜索提示；路线内SEP真正检出的像素才会成为待审计候选。",
                    "",
                ]
            )
    else:
        lines.extend(["没有生成可用的几何搜索路线。", ""])

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
