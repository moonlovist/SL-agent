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
        "Candidate-structure audit promotion identifies measurable residual structures, not arcs, rings, multiple images, or lenses; scientific calibration is still required.": "候选结构审计只确认某个候选是否足以作为可测量残差结构交给后端；它不把该结构直接认定为弧、环、多重像或强透镜，相关判据仍需用专家标注样本进行科学标定。",
        "Geometric paths only guide masked SEP searches; unfilled path regions are not treated as rings or missing evidence against lensing.": "几何路径只负责限制 SEP 再次搜索的位置；路径上没有亮像素的部分既不算环，也不能据此反对透镜解释。",
        "A PSF image is present and used by the audit-only comparison model, but the formal detection subtraction does not yet use PSF convolution.": "输入中有 PSF 图像，且审计用的比较模型使用了它；但当前主检测通道的前景减除尚未使用 PSF 卷积。",
        "A PSF image is present and the formal detection subtraction uses a PSF-convolved MGE when that fit succeeds; the pre-existing isophotal model is retained as a same-exposure sensitivity check.": "输入中有 PSF 图像；PSF-MGE 成功时它用于主检测通道，原等亮度线模型作为同一曝光上的敏感性对照保留。",
        "Robust clipping around a PSF-convolved foreground-light model is not simultaneous lens-light, lens-mass, and source-light inference; some lensed flux can still be absorbed by the foreground model.": "PSF 卷积前景光模型周围的稳健剔除不等于同时拟合透镜光、透镜质量与背景源光；仍可能有部分被透镜像的光被前景模型吸收。",
        "A photometrically selected centre is a light-distribution hypothesis and does not establish which galaxy is foreground without redshift information.": "光度上选出的中心只是光分布假设；没有红移信息时，不能由此确定哪个星系位于前景。",
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
            f"视觉链记录了 {len(document.model_candidates)} 个候选；审计后分为 "
            f"{sum(item.handoff_class == 'primary_candidate' for item in document.candidate_structure_audits)} 个主要候选、"
            f"{sum(item.handoff_class == 'recovery_candidate' for item in document.candidate_structure_audits)} 个恢复候选、"
            f"{sum(item.handoff_class == 'foreground_contaminated' for item in document.candidate_structure_audits)} 个前景污染候选和 "
            f"{sum(item.handoff_class == 'trace_candidate' for item in document.candidate_structure_audits)} 个追溯候选。"
            if document.model_candidates else "视觉链没有记录候选。"
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
        f"- 输入 VIS 同半径背景对比 S/N：{_value(structure.input_radial_contrast_snr)}（可见性诊断，不是透镜显著性）",
        f"- 输入 VIS 正对比像素比例：{_value(structure.input_radial_positive_fraction)}；有效同半径控制覆盖：{_value(structure.input_radial_control_coverage)}",
        f"- 径向厚度：{_value(structure.radial_width_px)} 像素；相对 PSF 宽度：{_value(structure.radial_width_psf)}",
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
        f"- 来源候选：{', '.join(structure.source_candidate_ids) or '未记录'}",
        f"- 候选结构审计记录：{', '.join(structure.candidate_audit_ids) or '未记录'}",
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
                f"- 敏感性比较模型 `{model_comparison.comparison_method}` 是否可用：{_yes_no(model_comparison.comparison_success)}",
                f"- 当前模型残差RMS：{_value(model_comparison.primary_residual_robust_rms, 6)}",
                f"- 比较模型残差RMS：{_value(model_comparison.comparison_residual_robust_rms, 6)}",
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
            "## 3. 经候选结构审计后交给后端的结构",
            "",
            f"当前共有 **{len(document.structures)}** 个结构进入后端证据比较。",
            "进入这里表示该候选已具备可测量结构所需的处理支持，不表示它已经被认定为弧、环、多重像或强透镜。",
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
            "下列候选会保留完整来源；随后由候选结构审计决定它进入正式结构、暂缓复核，还是只保留追溯。",
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
                "sep_only": "SEP稳定检出，但尚无几何支持",
                "sep_geometry_consistent": "SEP稳定检出，并与合理几何路线一致",
            }[item.status]
            support_level = {
                "raw_sep_only": "原始SEP结构支持",
                "sep_plus_geometry": "SEP与几何路线共同支持",
                "route_recovered": "路线内补充结构支持",
                None: "旧版记录，未单列结构支持等级",
            }[item.structural_support_level]
            partition = {
                "route_supported": "原SEP连通块中位于合格路线内的局部像素",
                "off_route_remainder": "原SEP连通块中位于路线外的降级追溯部分",
                "route_recovery": "路线限制搜索补充恢复的像素",
                None: "旧版记录，未单列局部分区",
            }[item.pixel_partition]
            review_priority = {
                "priority_keep": "重点复核",
                "downgraded_trace": "降级留痕",
                None: "旧版记录，未单列复核优先级",
            }[item.review_priority]
            lines.extend(
                [
                    f"### {item.id}",
                    "",
                    f"- 来源：{origin}；状态：{status}；结构支持等级：{support_level}",
                    f"- 局部分区：{partition}；人工复核优先级：{review_priority}；所属原SEP连通块：{item.parent_sep_component_id or '不适用'}",
                    f"- 中心：({_value(item.centroid_xy[0])}, {_value(item.centroid_xy[1])}) 像素；距中心：{_value(item.radius_px)} 像素",
                    f"- 像素数：{item.pixel_count}；累计S/N：{_value(item.integrated_snr)}；环向跨度：{_value(item.angular_extent_deg)}°",
                    f"- 候选级S/N多阈值存活：{item.snr_threshold_survival_count if item.snr_threshold_survival_count is not None else '未测得'}/{item.snr_threshold_total_count if item.snr_threshold_total_count is not None else '未测得'}；存活比例：{_value(item.snr_threshold_survival_fraction, 3)}",
                    f"- 有像素重叠的路线：{', '.join(item.path_ids) or '无'}；通过几何一致性检查的路线：{', '.join(item.geometry_consistent_path_ids) or '无'}",
                    f"- 所属候选系统：{', '.join(item.candidate_system_ids) or '无'}；是否位于前景减除敏感的近中心区域：{_yes_no(item.near_core)}；候选像素与敏感中心区的重叠比例：{_value(item.subtraction_core_overlap_fraction, 3)}",
                    f"- 换用前景敏感性比较残差后仍超过门槛的像素比例：{_value(item.foreground_comparison_survival_fraction, 3)}",
                    f"- 结构支持来源：{', '.join(item.support_sources) or '未记录'}",
                    "- 解释限制：这是视觉链提出的结构候选，不是弧、环、多重像或透镜标签。",
                    "",
                ]
            )
    else:
        lines.extend(["模型链没有保留结构候选。", ""])

    lines.extend(["## 3.2 候选结构审计", ""])
    lines.extend(
        [
            "这一步先检查候选是否具有SEP探测事实、连通信号、S/N多阈值稳定性，以及是否受到前景减除失败影响；随后才使用椭圆路线关系决定候选如何交给后端。",
            "椭圆路线不负责证明信号真实，也不把候选直接认定为透镜像。它只帮助把稳定且路线一致的候选放入主要结构列表；路线外稳定信号进入恢复组，前景敏感信号进入污染组，弱线索只保留追溯。",
            "只有主要候选进入当前规则层的正式结构列表。其余三组不会被删除，后续编写分级细则时仍可按假设调用。",
            "",
        ]
    )
    verdict_names = {
        "promoted": "晋级为可测量结构",
        "hold_for_review": "暂缓，交人工复核",
        "trace_only": "只保留过程追溯",
    }
    discovery_names = {
        "global_sep": "全图SEP检出、当前路线外",
        "global_sep_plus_geometry": "全图SEP检出且与路线一致",
        "route_recovery": "椭圆路线内二次SEP恢复",
        None: "旧版记录，未标注发现来源",
    }
    handoff_names = {
        "primary_candidate": "主要候选（进入当前规则层）",
        "recovery_candidate": "恢复候选（可信者带完整测量进入规则层）",
        "foreground_contaminated": "前景污染候选（仅在独立模型擦除时隔离）",
        "trace_candidate": "弱线索（只留追溯）",
    }
    section_names = {
        "signal_quality": "信号质量",
        "processing_robustness": "处理稳健性",
        "discovery_provenance": "发现来源",
    }
    if document.candidate_structure_audits:
        for item in document.candidate_structure_audits:
            check_text = "；".join(
                f"{name}={'通过' if value is True else '未通过' if value is False else '无法检查'}"
                for name, value in item.checks.items()
            )
            section_text = "；".join(
                f"{section_names.get(section, section)}："
                + "、".join(
                    f"{name}={'是' if value is True else '否' if value is False else '无法检查'}"
                    for name, value in values.items()
                )
                for section, values in item.sections.items()
            )
            lines.extend(
                [
                    f"### {item.id}：{item.candidate_id}",
                    "",
                    f"- 结果：{verdict_names[item.verdict]}；四路交接：{handoff_names[item.handoff_class]}；对应正式结构：{item.structure_id or '无'}",
                    f"- 审计范围：信号真实性；发现来源：{discovery_names[item.discovery_context]}；几何路线是否参与信号真实性结论：{'是' if item.geometry_used_for_verdict else '否'}；是否参与交接分流：{'是' if item.geometry_used_for_handoff else '否'}",
                    f"- 分区检查：{section_text or '旧版记录，未分区'}",
                    f"- 检查记录：{check_text or '无'}",
                    f"- 支持事实：{', '.join(item.supporting_facts) or '无额外支持'}",
                    f"- 风险标记：{', '.join(item.caution_flags) or '无'}",
                    f"- 审计解释：{item.rationale}",
                    "",
                ]
            )
    else:
        lines.extend(["没有候选结构审计记录。", ""])

    lines.extend(["## 3.3 路线内候选系统整合", ""])
    lines.extend(
        [
            "同一路线上的绿色原始SEP结构和紫色新恢复结构会组合为一个候选系统。它只表示这些结构共享一条几何搜索路线，不表示它们已经被证明是同一背景源的多重像。",
            "",
        ]
    )
    if document.candidate_systems:
        for system in document.candidate_systems:
            lines.extend(
                [
                    f"### {system.id}",
                    "",
                    f"- 对应路线：{system.path_id}；全部成员：{', '.join(system.member_candidate_ids)}",
                    f"- 原始SEP成员：{', '.join(system.sep_candidate_ids) or '无'}；路线内新恢复成员：{', '.join(system.recovered_candidate_ids) or '无'}",
                    f"- 组成部分数量：{system.component_count}；联合像素数：{system.union_pixel_count}；联合环向跨度：{_value(system.angular_extent_deg)}°",
                    f"- 成员半径范围：{_value(system.radius_range_px[0])}–{_value(system.radius_range_px[1])} 像素",
                    f"- 需要重点审计的标记：{', '.join(system.audit_flags) or '无'}",
                    "- 解释限制：这是候选结构组合，不是已确认的透镜系统，也不进入当前A/B/C评分。",
                    "",
                ]
            )
    else:
        lines.extend(["没有形成路线关联的候选系统。", ""])

    lines.extend(["## 3.4 几何路径引导搜索", ""])
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
    if document.candidate_handoff_review is not None:
        from ..agents.candidate_review import render_candidate_review_cn

        lines.extend(render_candidate_review_cn(document))
    diagnostic = document.supplementary_diagnostics
    if diagnostic:
        colour = diagnostic.get("multiband", {})
        lines.extend(["", "## 多波段颜色与补充测量", "",
            f"- 多波段测光状态：{colour.get('status', 'unavailable')}；VIS形态仍保持原分辨率。",
            f"- 颜色PSF匹配：{colour.get('psf_matching', {}).get('method', '不可用')}；目标FWHM={_value(colour.get('psf_matching', {}).get('target_fwhm_px'))}像素。",
            "- 原图污染孔径颜色只供显示；评级使用候选保护、固定椭圆径向前景扣除和旋转控制误差后的颜色关系。",
            "- 只有后续被明确透镜构型关联为同源像的候选对才允许颜色参与评级；可靠异色可削弱该构型，一致性只能支持已有形态，不能单独升级。候选与前景颜色相似只记歧义，不单独降级。",
            "- 核心混光、孔径重叠、波段不足或低信噪比记为未知；未知不是颜色一致，也不是颜色冲突。",
            f"- 联合弧链：{len(document.arc_chains)} 组；成员只合并真实像素，不填断口，不把碎片数量当成多像数量。",
            "- 全路径有符号测量保留负值和低于阈值的像素；名义信噪比未标定搜索次数、原生相关噪声及模型误差。",
            "- PSF伪影、本征结构来源尚未完成诊断；残差代理不能证明它们，也不能排除它们。", ""])
        for chain in document.arc_chains:
            lines.append(f"- {chain.id}：成员 {', '.join(chain.source_candidate_ids)}；真实像素 {chain.pixel_count}；实际角覆盖 {chain.angular_occupied_deg}°。")
        lines.extend(["", "### 孔径颜色（未去除前景污染）", "",
                      "| 孔径 | VIS−Y | VIS−J | VIS−H |", "|---|---|---|---|"])
        for key, values in colour.get("colours", {}).items():
            entries = []
            for band in ("NIR_Y", "NIR_J", "NIR_H"):
                pair = values.get(f"VIS-{band}", {})
                entries.append(f"{pair['value_mag']:.3f} ± {pair['error_mag_nominal']:.3f}" if pair.get("value_mag") is not None else "未知/低信噪比")
            lines.append("| " + " | ".join([key, *entries]) + " |")
        relation = colour.get("relationship_audit", {})
        pair_status = {
            "strongly_inconsistent": "明显不一致（可限制A）",
            "consistent_with_achromatic_scaling": "允许亮度倍率后的颜色一致",
            "inconclusive": "结论不确定", "unknown": "不可测/混光",
        }
        foreground_status = {
            "distinct_from_foreground": "与前景颜色显著不同",
            "indistinguishable_from_foreground": "无法与前景颜色区分（仅歧义，不单独降级）",
            "inconclusive": "结论不确定", "unknown": "不可测/核心混光",
        }
        lines.extend(["", "### 候选颜色关系（全量测量）", "",
                      "这里只列出全部候选对；专家层随后只允许被支持透镜构型连接的同源候选对投票。",
                      "",
                      "| 候选关系 | 判定 | 最大非灰度颜色差 | 显著度/原因 |", "|---|---|---:|---|"])
        for item in relation.get("component_pairs", []):
            difference = item.get("maximum_color_departure_mag")
            significance = item.get("maximum_departure_sigma")
            detail = (f"{significance:.2f}σ" if significance is not None
                      else item.get("reason", "未给出"))
            lines.append(f"| {item['first']} ↔ {item['second']} | {pair_status.get(item.get('status'), item.get('status'))} | "
                         f"{difference:.3f} mag | {detail} |" if difference is not None else
                         f"| {item['first']} ↔ {item['second']} | {pair_status.get(item.get('status'), item.get('status'))} | 未测得 | {detail} |")
        for item in relation.get("candidate_vs_foreground", []):
            difference = item.get("maximum_color_departure_mag")
            significance = item.get("maximum_departure_sigma")
            detail = (f"{significance:.2f}σ" if significance is not None
                      else item.get("reason", "未给出"))
            lines.append(f"| {item['candidate']} ↔ 前景 | {foreground_status.get(item.get('status'), item.get('status'))} | "
                         f"{difference:.3f} mag | {detail} |" if difference is not None else
                         f"| {item['candidate']} ↔ 前景 | {foreground_status.get(item.get('status'), item.get('status'))} | 未测得 | {detail} |")
    return "\n".join(lines)


def write_observation_summary_cn(
    document: ObservationDocument, output: Path
) -> Path:
    path = output / "observation_summary_cn.md"
    path.write_text(render_observation_summary_cn(document), encoding="utf-8")
    return path
