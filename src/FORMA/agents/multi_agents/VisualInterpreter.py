import json
import os
import numpy as np
import logging
from pathlib import Path

from FORMA.agents.common.state import SpectroState
from FORMA.agents.common.base_agent import BaseAgent
from FORMA.agents.common.result_writer import ResultWriter

from FORMA.core.runtime.runtime_container import RuntimeContainer

from FORMA.agents.multi_agents.utils.usage import safe_to_bool, find_overlap_regions
from FORMA.agents.multi_agents.utils.VI import (
    # ═══════════════════════════════════════════════════════════════════
    # PNG 通道已注释（2026-07-06）：以下函数依赖 cv2，仅 PNG 输入使用。
    # _detect_chart_border, _crop_img,
    # _remap_to_cropped_canvas, _pixel_tickvalue_fitting,
    # _convert_to_spectrum,
    # _detect_axis_ticks_tesseract,
    # _detect_axis_ticks_paddle,
    # ═══════════════════════════════════════════════════════════════════
    run_continuum_fitting_masked,
    # brute_force_line_matching,  # 已注释（2026-07-06）：仅 REDROCK=false 时使用
    _load_spectrum_from_fits,
    run_redshift_scoring_v2,
    run_redshift_scoring_v3,
    run_redrock_pipeline,
)
from FORMA.agents.multi_agents.utils.cwt_feature_finder import (
    run_cwt_feature_detection,
)
from FORMA.agents.multi_agents.utils.plot import (
    plot_spec_extract,
    plot_spectrum_snr,
    plot_continuum,
    plot_residual_spectrum,
    plot_features,
)


# ---------------------------------------------------------
# 1. Visual Assistant — 负责图像理解与坐标阅读
# ---------------------------------------------------------
class VisualInterpreter(BaseAgent):
    """
    从科学光谱图中自动提取坐标轴刻度、边框、像素映射、峰/谷等信息
    """

    agent_name = "VisualInterpreter"
    _SKILL_DIR = Path(__file__).resolve().parent / "harness" / "skills" / "VisualInterpreter"

    def __init__(self, runtime: RuntimeContainer):
        super().__init__(runtime)

    # ======================================================
    # LLM 交互方法
    # ======================================================

    def load_spectrum_from_fits(self, state: SpectroState) -> SpectroState:
        """从 FITS 文件加载光谱数据，波段信息来自 .env 中的 ARM_NAME / ARM_WAVELENGTH_RANGE"""
        params = self.runtime.configs.params
        state['spectrum'] = _load_spectrum_from_fits(
            state['file_path'],
            arm_names=params.arm_name,
            arm_wavelength_ranges=params.arm_wavelength_range,
        )
        return state

    # ═════════════════════════════════════════════════════════════════════
    # PNG 通道已注释（2026-06-30）：以下方法仅用于 PNG 输入处理流程。
    # 如需恢复 PNG 支持，取消注释即可。
    # ═════════════════════════════════════════════════════════════════════

    # async def detect_axis_ticks(self, state: SpectroState) -> SpectroState:
    #     """调用 VLM 检测坐标轴刻度"""
    #     if not state['file_path'] or not os.path.exists(state['file_path']):
    #         logging.error("No image provided or image path does not exist")
    #         raise
    #
    #     system_prompt = self._load_skill("detect_axis_ticks")
    #     user_prompt = "Please analyze this spectrum plot and extract the axis information."
    #
    #     axis_info = await self.call_llm_with_context(
    #         system_prompt=system_prompt,
    #         user_prompt=user_prompt,
    #         image_path=state['file_path'],
    #         parse_json=True,
    #         description="Axis information",
    #         want_tools=False
    #     )
    #     if axis_info == "Non-spectrum":
    #         logging.error("The input image is not a spectral plot. LLM output: %s", axis_info)
    #         raise ValueError("Non-spectrum image detected")
    #     state["axis_info"] = axis_info
    #     return state

    # async def detect_axis_ticks_OCR(self, state: SpectroState) -> SpectroState:
    #     """调用 OCR 检测坐标轴刻度"""
    #     OCR = self.runtime.configs.params.ocr
    #     print(f"OCR: {OCR}")
    #     if OCR == 'paddle':
    #         state['OCR_detected_ticks'] = _detect_axis_ticks_paddle(state)
    #     else:
    #         state['OCR_detected_ticks'] = _detect_axis_ticks_tesseract(state)
    #     print(state["OCR_detected_ticks"])
    #     return state

    # async def combine_axis_mapping(self, state: SpectroState) -> SpectroState:
    #     """结合视觉结果与 OCR 结果生成像素-数值映射"""
    #     system_prompt = self._load_skill("combine_axis_mapping")
    #     user_prompt = (
    #         "Please perform an integrated correction on the following two sets of "
    #         "structured scale results from the same spectral plot:\n\n"
    #         "【Visual Model Result】\n"
    #         f"{json.dumps(state['axis_info'], indent=2, ensure_ascii=False)}\n\n"
    #         "【OCR / OpenCV Result】\n"
    #         f"{json.dumps(state['OCR_detected_ticks'], indent=2, ensure_ascii=False)}\n\n"
    #         "Task:\n"
    #         "Consistency-correct and complete the OCR result, "
    #         "ensuring the final output satisfies system monotonicity and "
    #         "conflict-resolution rules.\n\n"
    #         "Output strictly the corrected JSON array.\n"
    #         "Do not include any explanations or additional text."
    #     )
    #
    #     tick_pixel_raw = await self.call_llm_with_context(
    #         system_prompt=system_prompt,
    #         user_prompt=user_prompt,
    #         image_path=None,
    #         parse_json=True,
    #         description="Tick pixel mapping",
    #         want_tools=False
    #     )
    #     state["tick_pixel_raw"] = tick_pixel_raw
    #     return state

    # async def revise_axis_mapping(self, state: SpectroState) -> SpectroState:
    #     """检查并修正刻度值与像素位置匹配关系"""
    #     system_prompt = self._load_skill("revise_axis_mapping")
    #     user_prompt = (
    #         "Please check the following scale value to pixel mapping:\n"
    #         f"{json.dumps(state['tick_pixel_raw'], indent=2, ensure_ascii=False)}\n\n"
    #         "Tasks:\n"
    #         "- Revise any entries that violate the monotonicity rule\n"
    #         "- Keep null values unchanged\n"
    #         "- Output the revised JSON array; if the original input is correct, "
    #         "return it as-is\n"
    #         "- Do not output any explanations or additional text"
    #     )
    #
    #     tick_pixel_revised = await self.call_llm_with_context(
    #         system_prompt=system_prompt,
    #         user_prompt=user_prompt,
    #         image_path=None,
    #         parse_json=True,
    #         description="Revised tick mapping",
    #         want_tools=False
    #     )
    #     state["tick_pixel_raw"] = tick_pixel_revised
    #     return state

    # async def check_border(self, state: SpectroState):
    #     """调用 LLM 判断裁剪边界是否干净"""
    #     system_prompt = self._load_skill("check_border")
    #     user_prompt = (
    #         "You will receive two images:\n\n"
    #         "1. The original spectral image, which may include plot borders.\n"
    #         "2. A matplotlib astronomical spectrum image preprocessed with OCR and "
    #         "OpenCV, where an attempt has been made to crop out the borders and "
    #         "surrounding areas.\n\n"
    #         "Task:\n"
    #         "Determine whether obvious straight-line border remnants remain along "
    #         "the four edges (top, right, bottom, left) of the image "
    #         "(e.g., long, straight black or dark line segments, typically part of "
    #         "the outer frame of the coordinate axes).\n\n"
    #         "Judgment criteria:\n"
    #         "- If **no such straight-line segment is visible** along a given edge "
    #         '→ "cropped cleanly"\n'
    #         "- If **obvious straight-line segments are still visible** along a "
    #         'given edge → "not cropped cleanly"\n\n'
    #         "Please output your result strictly in the following JSON format, "
    #         "containing only the four specified keys, with values as the strings "
    #         "'true' or 'false':\n\n"
    #         "{\n"
    #         '    "top": "true" or "false",\n'
    #         '    "right": "true" or "false",\n'
    #         '    "bottom": "true" or "false",\n'
    #         '    "left": "true" or "false"\n'
    #         "}\n\n"
    #         "Do not output any other content."
    #     )
    #
    #     response = await self.call_llm_with_context(
    #         system_prompt,
    #         user_prompt,
    #         image_path=[state['file_path'], state['crop_path']],
    #         parse_json=True,
    #         description='Check cropping',
    #         want_tools=False
    #     )
    #     result = {}
    #     for key in ['top', 'right', 'bottom', 'left']:
    #         try:
    #             val = safe_to_bool(response.get(key, 'false'))
    #             result[key] = val == 'true'
    #         except Exception:
    #             logging.warning(f"Error parsing {key} border, defaulting to False: {response}")
    #             result[key] = False
    #     return result

    # async def border_detection_and_cropping(self, state: SpectroState) -> SpectroState:
    #     """迭代调整边距，直到 LLM 确认四边干净"""
    #     state['margin'] = {
    #         'top': 20, 'right': 10, 'bottom': 15, 'left': 10,
    #     }
    #     MAX_MARGIN = 30
    #     INCREMENT = 2
    #
    #     stop = False
    #     while not stop:
    #         state["chart_border"] = _detect_chart_border(state['file_path'], state['margin'])
    #         _crop_img(state['file_path'], state["chart_border"], state['crop_path'])
    #
    #         box_new = await self.check_border(state)
    #
    #         if all(box_new.values()):
    #             stop = True
    #             break
    #         elif any(state['margin'][k] >= MAX_MARGIN for k in state['margin']):
    #             stop = True
    #             logging.info(f"Reached maximum margin, stopping cropping: {state['margin']}")
    #             break
    #         else:
    #             for k, clean in box_new.items():
    #                 if not clean:
    #                     state['margin'][k] += INCREMENT
    #
    #     return state

    # ======================================================
    # 主流程
    # ======================================================

    async def run(self, state: SpectroState, plot: bool = True):
        """执行完整视觉分析流程"""
        params = self.runtime.configs.params
        try:
            if self.runtime.configs.io.input_format == 'fits':
                self.load_spectrum_from_fits(state)
            # ═════════════════════════════════════════════════════════════
            # PNG 通道已注释（2026-06-30）
            # PaddleOCR / Tesseract 安装繁琐，测试阶段仅支持 FITS 输入。
            # 如需恢复 PNG 输入通道，请：
            #   1. 取消 VI.py 顶部 pytesseract / PaddleOCR 导入的注释
            #   2. 取消 VI.py 中 _detect_axis_ticks_* 函数的注释
            #   3. 取消本文件 OCR 导入及下方 else 分支的注释
            #   4. 取消本类中 PNG-only 方法（detect_axis_ticks 等）的注释
            # ═════════════════════════════════════════════════════════════
            else:
                raise NotImplementedError(
                    "PNG 输入通道已暂时禁用（2026-06-30）。"
                    "当前仅支持 INPUT_FORMAT=fits。"
                    "如需恢复 PNG 支持，请参见 VisualInterpreter.py 中相关注释。"
                )
            # else:
            #     # === Phase A: 坐标轴检测与校准 ===
            #     await self.detect_axis_ticks(state)
            #     await self.detect_axis_ticks_OCR(state)
            #     await self.combine_axis_mapping(state)
            #     await self.revise_axis_mapping(state)
            #
            #     # === Phase B: 图像裁剪与像素映射 ===
            #     await self.border_detection_and_cropping(state)
            #     state["tick_pixel_remap"] = _remap_to_cropped_canvas(
            #         state['tick_pixel_raw'], state["chart_border"]
            #     )
            #     state["pixel_to_value"] = _pixel_tickvalue_fitting(state['tick_pixel_remap'])
            #
            #     # === Phase C: 光谱重建 ===
            #     arm_name = self.runtime.configs.params.arm_name
            #     arm_wavelength_range = self.runtime.configs.params.arm_wavelength_range
            #     state["spectrum"] = _convert_to_spectrum(
            #         state['crop_path'], state['pixel_to_value'], arm_name, arm_wavelength_range
            #     )

            # ── Compute and store overlap regions ──────────────────────
            overlap = find_overlap_regions(params.arm_name, params.arm_wavelength_range)
            if overlap:
                state['spectrum']['overlap_regions'] = list(overlap.values())

            plot_spec_extract(state)
            plot_spectrum_snr(state)

            # 保存光谱数组
            spec = state["spectrum"]
            vi_dir = os.path.join(state['output_dir'], 'visual_interpreter')
            os.makedirs(vi_dir, exist_ok=True)
            save_data = {k: spec[k] for k in ("wavelength", "flux", "snr") if k in spec}
            if spec.get("ivar") is not None:
                save_data["ivar"] = spec["ivar"]
            np.savez_compressed(
                os.path.join(vi_dir, f"{state['file_name']}_spectrum.npz"),
                **save_data,
            )
            state['spectrum_npz_path'] = os.path.join(
                state['output_dir'], 'visual_interpreter', f"{state['file_name']}_spectrum.npz"
            )

            # === Phase D: 迭代特征检测（在 continuum fitting 之前）===
            spec = state["spectrum"]
                        
            # 优先使用 ivar，其次使用 snr
            ivar_data = spec.get("ivar", None)
            effective_snr_data = spec.get("snr", 7.0) if ivar_data is None else None
                        
            # ── CWT 发射线检测参数 ──
            emission_detection_params = {
                'snr_thresh': params.cwt_snr_thresh,
                'min_ridge_length': params.cwt_min_ridge_length,
                'n_scales': params.cwt_n_scales,
                'min_scale': params.cwt_min_width,
                'max_scale': params.cwt_max_width,
            }

            feature_result = run_cwt_feature_detection(
                output_dir=state['output_dir'],
                file_name=state['file_name'],
                wavelength=spec["wavelength"],
                flux=spec["flux"],
                ivar=ivar_data,
                effective_snr=effective_snr_data,
                n_iterations=params.n_iterations if hasattr(params, 'n_iterations') else 3,
                emission_detection_params=emission_detection_params,
                verbose=True,
            )
                        
            # 保存结果到 state（发射线 → peaks，吸收线 → troughs）
            # 注意：返回値已按波长从小到大排序，包含 amplitude_rank 字段
            df_em = feature_result['df_emission']
            df_ab = feature_result['df_absorption']
                        
            state['peaks'] = df_em.to_dict('records') if len(df_em) > 0 else []
            state['troughs'] = df_ab.to_dict('records') if len(df_ab) > 0 else []
                        
            # records 同样已按波长排序
            state['absorption_records'] = feature_result['records_absorption']
            state['emission_records'] = feature_result['records_emission']

            # === Phase E: Continuum 拟合（将已检测峰/谷区域 mask掉后再拟合）===
            state['continuum'], state['residual_spectrum'] = run_continuum_fitting_masked(
                spec["wavelength"],
                spec["flux"],
                peaks=state['emission_records'],
                troughs=state['absorption_records'],
                chebyshev_degree=params.chebyshev_degree if hasattr(params, 'chebyshev_degree') else None,
                chebyshev_min_degree=params.chebyshev_min_degree if hasattr(params, 'chebyshev_min_degree') else 1,
                chebyshev_max_degree=params.chebyshev_max_degree if hasattr(params, 'chebyshev_max_degree') else 10,
            )
            plot_continuum(state)
            plot_residual_spectrum(state)

            plot_features(state)

            # tol_wavelength = self.runtime.configs.params.tol_wavelength  # 已注释（2026-07-06）

            print(params.redrock)
            # ═══════════════════════════════════════════════════════════════
            # REDROCK 路径：运行 rrdesi 生成红移假设。
            # 原 brute_force_line_matching 分支已注释（2026-07-06），
            # 如需恢复 REDROCK=false 路径，取消下面和 import 处的注释。
            # ═══════════════════════════════════════════════════════════════
            if params.rr_template_dir is None:
                raise ValueError(
                    "REDROCK=true but RR_TEMPLATE_DIR is not set in .env"
                )
            if params.use_archetypes and params.archetype_dir is None:
                raise ValueError(
                    "REDROCK=true, USE_ARCHETYPES=true but ARCHETYPE_DIR is not set in .env"
                )
            state['redshift_hypotheses'] = run_redrock_pipeline(
                input_fits=state['file_path'],
                output_dir=state['output_dir'],
                file_name=state['file_name'],
                rr_template_dir=params.rr_template_dir,
                archetype_dir=params.archetype_dir,
                use_archetypes=params.use_archetypes,
                nminima=params.rr_nminima,
                nnearest=params.rr_nnearest,
                omp_num_threads=params.rr_omp_num_threads,
            )
            # ── 原 REDROCK=false 分支（已注释）────────────────────
            # else:
            #     state['redshift_hypotheses'] = brute_force_line_matching(
            #         state, tol_wavelength,
            #     )

            ResultWriter().write_redshift_hypotheses(state)

            return state
        except Exception as e:
            print(f"run pipeline terminated with error: {e}")
            raise
