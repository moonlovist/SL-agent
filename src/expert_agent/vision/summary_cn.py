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
            f"视觉链记录了 {len(document.model_candidates)} 个候选，其中 "
            f"{sum(item.verdict == 'promoted' for item in document.candidate_structure_audits)} 个经审计进入正式结构列表。"
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
                    f"- 有像素重叠的路线：{', '.join(item.path_ids) or '无'}；通过几何一致性检查的路线：{', '.join(item.geometry_consistent_path_ids) or '无'}",
                    f"- 所属候选系统：{', '.join(item.candidate_system_ids) or '无'}；是否位于前景减除敏感的近中心区域：{_yes_no(item.near_core)}",
                    f"- 换用PSF-MGE比较残差后仍超过门槛的像素比例：{_value(item.foreground_comparison_survival_fraction, 3)}",
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
            "这一步审计图4候选池中的全部候选，只回答“这个候选是否具有可用的探测信号，可以作为可测量残差结构交给后端解释”。它检查数据可用性、既有SEP探测事实、连通信号以及更换前景模型后是否被完全抹掉。",
            "候选位于椭圆路线内、路线外或由路线二次恢复，只作为发现来源和几何关系记录，不作为审计晋级或淘汰条件。审计不判断这个结构究竟是透镜像、星系自身结构还是处理伪影。",
            "审计结果分为：晋级为结构、暂缓并交人工复核、只保留过程追溯。这里没有再计算一个新的总分。",
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
                    f"- 结果：{verdict_names[item.verdict]}；对应正式结构：{item.structure_id or '无'}",
                    f"- 审计范围：信号真实性；发现来源：{discovery_names[item.discovery_context]}；几何路线是否参与审计结论：{'是' if item.geometry_used_for_verdict else '否'}",
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
    return "\n".join(lines)


def write_observation_summary_cn(
    document: ObservationDocument, output: Path
) -> Path:
    path = output / "observation_summary_cn.md"
    path.write_text(render_observation_summary_cn(document), encoding="utf-8")
    return path
