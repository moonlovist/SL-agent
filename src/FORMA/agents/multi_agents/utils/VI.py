import csv
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
# import cv2  # 已注释（2026-07-06）：PNG 通道已禁用，仅 _detect_chart_border 等 PNG 函数使用
import numpy as np
from astropy.io import fits
# from scipy.ndimage import gaussian_filter1d
import pandas as pd
# ═══════════════════════════════════════════════════════════════════════════
# PNG 通道已注释（2026-06-30）：PaddleOCR / Tesseract 安装繁琐，测试阶段不启用。
# 如需恢复 PNG 输入通道，请取消以下导入及对应 OCR 函数的注释。
# ═══════════════════════════════════════════════════════════════════════════
# import pytesseract
from collections import defaultdict
# from paddleocr import PaddleOCR
from typing import Any, Dict, List, Tuple
from scipy.optimize import curve_fit

from FORMA.agents.common.state import SpectroState
from FORMA.agents.multi_agents.utils.usage import find_overlap_regions, average_flux_by_wavelength

# ── 物理常数 ────────────────────────────────────────────────────────────
SIGMA_TO_FWHM = 2.355  # FWHM = 2.355 × σ (Gaussian)

# ═══════════════════════════════════════════════════════════════════════════
# Step 1.2: OCR / Axis Tick Detection
# ═══════════════════════════════════════════════════════════════════════════
# PNG 通道已注释（2026-06-30）：PaddleOCR / Tesseract 安装繁琐，测试阶段不启用。
# 如需恢复，取消下面两个函数及文件顶部对应 import 的注释即可。

# def _detect_axis_ticks_tesseract(state: SpectroState, config=None):
#     # Tesseract is not good. I prefer paddle.
#     if config is None:
#         # config = r'--oem 3 --psm 5 -c tessedit_char_whitelist=0123456789.-eE+ '
#         config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.-eE+ '
#         # config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.-eE+ '
#     image_path = state['file_path']
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#     sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
#     # # _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
#     thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#
#     data = pytesseract.image_to_data(
#         thresh, config=config, output_type=pytesseract.Output.DICT
#     )
#
#     tick_values = []
#     for i in range(len(data['text'])):
#         text = data['text'][i].strip()
#         if text != "":
#             try:
#                 value = float(text)
#                 (x, y, w, h) = (data['left'][i], data['top'][i],
#                                 data['width'][i], data['height'][i])
#                 cx, cy = x + w//2, y + h//2
#                 tick_values.append({
#                     "value": value,
#                     "position": [cx, cy],
#                     "bounding-box-scale": [w, h]
#                 })
#             except ValueError:
#                 pass
#
#     return tick_values
#
#
# def _detect_axis_ticks_paddle(state: SpectroState):
#     ocr = PaddleOCR(
#         use_doc_orientation_classify=False,
#         use_doc_unwarping=False,
#         use_textline_orientation=False
#     )
#     result = ocr.predict(state['file_path'])
#     for res in result:
#         # res.print()
#         res.save_to_img(state['output_dir'])
#         res.save_to_json(state['output_dir'])
#     data = []
#     for i in range(len(result[-1]['rec_texts'])):
#         pos = result[-1]['rec_polys'][i]
#         center = [
#             int((pos[0][0] + pos[2][0]) / 2),
#             int((pos[0][1] + pos[2][1]) / 2),
#         ]
#         width = int((pos[1][0] - pos[0][0] + pos[2][0] - pos[3][0]) / 2)
#         height = int((pos[3][1] - pos[0][1] + pos[2][1] - pos[1][1]) / 2)
#         info = {
#             'value': result[-1]['rec_texts'][i],
#             'position': center,
#             'bounding-box-scale': [width, height]
#         }
#         data.append(info)
#     return data


# ===========================================================
# Step 1.5: Border Detection & Image Cropping
# ===========================================================

def _detect_chart_border(
        image_path: str,
        margin: Dict = {
            'top': 10,
            'right': 10,
            'bottom': 10,
            'left': 10
        }
) -> dict:
    """
    检测图像中图表的外围边框，并微调尺寸。

    参数:
        image_path: 图像文件路径
        margin: 调整边框的像素量（正数表示收缩边框）

    返回:
        dict 包含边框位置: {"x": int, "y": int, "w": int, "h": int}
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到任何轮廓")

    # 找到最大轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    # 微调边框
    x += margin['left']
    y += margin['top']
    w -= (margin['left'] + margin['right'])
    h -= (margin['top'] + margin['bottom'])

    return {"x": x, "y": y, "w": w, "h": h}


def _crop_img(image_path: str, border_info: dict, save_path: str) -> str:
    """
    裁剪图像指定区域并保存。

    参数:
        image_path: 输入图像路径
        save_path: 保存裁剪后图像路径

    返回:
        save_path
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    x = border_info['x']
    y = border_info['y']
    w = border_info['w']
    h = border_info['h']

    img_cropped = img[y:y+h, x:x+w]
    cv2.imwrite(save_path, img_cropped)

    print(f'cropped image is saved to {save_path}')

    return save_path


# ===========================================================
# Step 1.6~1.7: Pixel Coordinate Remapping & Fitting
# ===========================================================

def _remap_to_cropped_canvas(old_info, chart_border):
    """
    将原图坐标映射到裁剪后的画布上，自动处理 None 坐标。
    """
    x0, y0, w, h = chart_border["x"], chart_border["y"], chart_border["w"], chart_border["h"]

    new_info = []
    for d in old_info:
        ox = d.get("position_x") if d.get("position_x") is not None else None
        oy = d.get("position_y") if d.get("position_y") is not None else None

        new_d = d.copy()

        # 重映射到裁剪画布
        nx = ox - x0 if ox is not None else None
        ny = oy - y0 if oy is not None else None
        # 越界裁剪
        nx = max(0, min(nx, w - 1)) if ox is not None else None
        ny = max(0, min(ny, h - 1)) if oy is not None else None
        new_d["position_x"] = nx
        new_d["position_y"] = ny

        new_info.append(new_d)

    return new_info


def _linear_func(x, a, b):
    return a * x + b


def _pixel_tickvalue_fitting(arr: list) -> dict:
    """
    对刻度数据做加权线性拟合（支持 x/y 轴分开）。
    输入: Python list，每个元素为 dict
    输出: dict 包含各轴拟合结果
    """
    results = {}
    for axis in ["x", "y"]:
        # 提取有效数据
        values, pixels, sigmas, confs = [], [], [], []
        for d in arr:
            if d["axis"] == axis and d[f'position_{axis}'] is not None:
                values.append(float(d["value"]))
                pixels.append(float(d["position_x"] if axis == 'x' else d["position_y"]))
                sigmas.append(float(d["sigma_pixel"]) if d["sigma_pixel"] is not None else np.inf)
                confs.append(float(d["conf_llm"]) if d["conf_llm"] is not None else 1.0)

        if len(values) < 2:
            continue

        values = np.array(values, dtype=float)
        pixels = np.array(pixels, dtype=float)
        sigmas = np.array(sigmas, dtype=float)
        confs = np.array(confs, dtype=float)

        # 有效 sigma
        sigma_eff = sigmas / np.sqrt(confs)

        # 拟合
        popt, _ = curve_fit(
            _linear_func,
            pixels,
            values,
            sigma=sigma_eff,
            absolute_sigma=True
        )
        a_fit, b_fit = popt
        value_fit = _linear_func(pixels, a_fit, b_fit)
        residual = values - value_fit
        rms = np.sqrt(np.mean(residual**2))

        results[axis] = {
            "a": float(a_fit),
            "b": float(b_fit),
            "rms": float(rms),
            "residuals": residual.tolist()
        }
        print(f"{axis}: {results[axis]}")

    return results


# ===========================================================
# Step 1.8: Curve Extraction & Spectrum Reconstruction
# ===========================================================

def _process_and_extract_curve_points(input_path: str):
    """
    读取图像，去除背景并转换为二值图像，提取曲线的像素点云。

    参数：
    - input_path：原始图像文件路径

    返回：
    - curve_points: 曲线像素点云（列表形式，包含每个点的(x, y)坐标）
    - curve_gray_values: 曲线像素灰度值（列表形式）
    """
    # 1. 读取原始图像
    img = cv2.imread(input_path)

    # 2. 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 二值化处理，背景为白色，曲线为黑色
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 4. 提取白色曲线像素点（黑色曲线会变为白色，背景变为黑色）
    curve_points = []
    curve_gray_values = []

    # 遍历所有像素点，提取白色区域（即曲线部分）
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 255:  # 白色区域（曲线）
                curve_points.append([x, y])
                curve_gray_values.append(gray[y, x])

    curve_gray_values = np.array(curve_gray_values, dtype=np.float64)

    return curve_points, curve_gray_values


def _convert_to_spectrum(crop_path, axis_fitting_info, band_names=None, band_wavelengths=None):
    """
    从裁剪后的图像中提取曲线并转换为波长（wavelength）和光谱强度（flux）。

    输入：
    - crop_path: 裁剪后的图像路径
    - axis_fitting_info: 包含 x/y 轴拟合系数的字典
    - band_names: 波段名称列表（可选），用于计算并去除波段重叠区域
    - band_wavelengths: 波段波长范围列表（可选），每个元素为 [start, end]

    输出：
    - spectrum_dict: 包含转换后的波长、flux 和平均后的波长与flux的字典
    """
    # Step 1: 提取曲线像素点
    points, gray = _process_and_extract_curve_points(crop_path)

    # Step 2: 转换坐标到物理量
    points = np.array(points)
    xs = points[:, 0]
    ys = points[:, 1]

    # 提取 x 轴和 y 轴的物理量转换系数
    a_y = axis_fitting_info['y']['a']
    b_y = axis_fitting_info['y']['b']
    flux = a_y * ys + b_y

    a_x = axis_fitting_info['x']['a']
    b_x = axis_fitting_info['x']['b']
    wavelength = a_x * xs + b_x

    # Step 2.5: 去除波段重叠区域（若提供了波段信息）
    if band_names is not None and band_wavelengths is not None:
        overlap_regions = find_overlap_regions(band_names, band_wavelengths)
        if overlap_regions:
            print(f"检测到 {len(overlap_regions)} 个重叠区域，将从光谱中移除: {overlap_regions}")
            # 构建原始点的 mask：True 表示保留
            keep_mask = np.ones(len(wavelength), dtype=bool)
            for region_name, (ov_start, ov_end) in overlap_regions.items():
                in_overlap = (wavelength >= ov_start) & (wavelength <= ov_end)
                keep_mask &= ~in_overlap
                print(f"  移除重叠区域 '{region_name}': [{ov_start}, {ov_end}]，共 {in_overlap.sum()} 个点")
            wavelength = wavelength[keep_mask]
            flux = flux[keep_mask]

    # 计算每个波长的平均flux
    unique_wavelength, mean_flux = average_flux_by_wavelength(wavelength, flux)

    max_unresolved_flux = []
    min_unresolved_flux = []
    std_flux = []  # 存储每个波长的标准差

    # 计算每个波长的统计量
    for w in unique_wavelength:
        unresolved_flux = flux[wavelength == w]
        max_unresolved_flux.append(np.max(unresolved_flux))
        min_unresolved_flux.append(np.min(unresolved_flux))

        # 计算标准差
        if len(unresolved_flux) > 1:
            # 使用无偏估计（除以 n-1）计算样本标准差
            std_val = np.std(unresolved_flux, ddof=1)
        else:
            # 如果只有一个数据点，标准差设为 0
            std_val = 0.0
        std_flux.append(std_val)

    # 计算 delta_flux
    delta_flux = np.array(max_unresolved_flux) - np.array(min_unresolved_flux)

    # 计算信噪比 SNR = mean / std
    snr = np.where(
        np.array(std_flux) != 0,
        mean_flux / np.array(std_flux),
        np.inf
    )

    snr_medium = np.median(snr)
    # result = mode(snr)
    # print("SNR中位数:", snr_medium)
    # print("SNR众数:", result.mode)

    spectrum_dict = {
        'wavelength': unique_wavelength.tolist(),
        'flux': mean_flux.tolist(),
        'max_unresolved_flux': max_unresolved_flux,
        'min_unresolved_flux': min_unresolved_flux,
        'snr': snr.tolist(),
    }

    return spectrum_dict


# ===========================================================
# Step 1.10: Load Spectrum from FITS
# ===========================================================

def _load_spectrum_from_fits(fits_path: str,
                           arm_names=None,
                           arm_wavelength_ranges=None) -> dict:
    """
    从 FITS 文件中读取光谱数据并返回 spectrum_dict。

    支持两种格式：
    1. 多波段分波段格式（通过 arm_names 指定波段名，如 ['B','R','Z'] 或 ['U','V','I']）
    2. 单表格格式（包含 WAVELENGTH/FLUX 列的 BinTableHDU）

    Parameters
    ----------
    fits_path : str
        FITS 文件路径
    arm_names : list[str], optional
        波段名称列表，如 ['B','R','Z']（DESI）或 ['U','V','I']（CSST）。
        若为 None 则报错终止，要求用户在 .env 中配置。
    arm_wavelength_ranges : list[list[float]], optional
        各波段波长范围，如 [[3600,5800],[5760,7620],[7520,9824]]。
        若为 None 则从数据中自动计算。用于波段重叠区域检测。

    Returns
    -------
    dict with keys: wavelength, flux, snr, ivar
    """

    with fits.open(fits_path) as hdul:
        # 获取所有 HDU 名称
        hdu_names = [hdu.name.upper() for hdu in hdul]
        logging.info(f"FITS HDU names: {hdu_names}")

        # ── 从 FIBERMAP / METADATA HDU 读取 VI_Z / VI_SPECTYPE ──────
        vi_z = None
        vi_spectype = None
        # 优先从 FIBERMAP 读取（新格式），回退到 METADATA（旧格式）
        meta_hdu_name = None
        if 'FIBERMAP' in hdu_names:
            meta_hdu_name = 'FIBERMAP'
        elif 'METADATA' in hdu_names:
            meta_hdu_name = 'METADATA'
        if meta_hdu_name is not None:
            try:
                meta = hdul[meta_hdu_name].data
                if 'VI_Z' in meta.dtype.names:
                    vi_z = float(meta['VI_Z'][0])
                if 'VI_SPECTYPE' in meta.dtype.names:
                    vi_spectype = str(meta['VI_SPECTYPE'][0]).strip()
                logging.info(
                    f"FITS {meta_hdu_name}: VI_Z={vi_z}, VI_SPECTYPE={vi_spectype}"
                )
            except Exception as exc:
                logging.warning(f"Failed to read FITS {meta_hdu_name}: {exc}")

        # 检测是否为多波段分波段格式
        # 若 arm_names 由外部配置提供，直接使用；否则自动从 HDU 名推断波段
        if arm_names is not None:
            band_names = arm_names
            is_multi_arm = all(
                f'{band}_WAVELENGTH' in hdu_names and f'{band}_FLUX' in hdu_names
                for band in band_names
            )
        else:
            # 未配置 arm_names，报错提示用户
            raise ValueError(
                "未配置多波段信息。请在 .env 中设置 ARM_NAME 和 ARM_WAVELENGTH_RANGE。\n"
                "例如：\n"
                "  ARM_NAME = B,R,Z\n"
                "  ARM_WAVELENGTH_RANGE = 3600-5800,5760-7620,7520-9824\n"
                "如果你使用 CSST 数据，请替换为 U,V,I 及对应波长范围。"
            )

        if is_multi_arm:
            # === 多波段分波段格式 ===
            logging.info(f"Detected multi-arm FITS format: {band_names}")

            # 读取各波段的数据并记录波长范围
            wavelength_list = []
            flux_list = []
            ivar_list = []
            snr_list = []
            mask_list = []
            has_ivar = False
            has_snr = False
            band_wavelengths = []
            if arm_wavelength_ranges is not None:
                band_wavelengths = arm_wavelength_ranges

            for band in band_names:
                wave_hdu = hdul[f'{band}_WAVELENGTH']
                flux_hdu = hdul[f'{band}_FLUX']

                wave_data = np.ravel(wave_hdu.data)
                wavelength_list.append(wave_data)
                flux_list.append(np.ravel(flux_hdu.data))

                # 读取 IVAR / SNR（有则读，二者都有则都读）
                ivar_hdu_name = f'{band}_IVAR'
                snr_hdu_name = f'{band}_SNR'
                band_has_ivar = ivar_hdu_name in hdu_names
                band_has_snr = snr_hdu_name in hdu_names

                if band_has_ivar:
                    ivar_list.append(np.ravel(hdul[ivar_hdu_name].data))
                    has_ivar = True
                else:
                    ivar_list.append(None)

                if band_has_snr:
                    snr_list.append(np.ravel(hdul[snr_hdu_name].data))
                    has_snr = True
                else:
                    snr_list.append(None)

                if not band_has_ivar and not band_has_snr:
                    logging.warning(f"波段 {band} 缺少 IVAR 和 SNR，将使用默认 effective_snr=5.0")

                # 读取 SPECMASK（如果存在）
                mask_hdu_name = f'{band}_MASK'
                if mask_hdu_name in hdu_names:
                    mask_data = np.ravel(hdul[mask_hdu_name].data)
                    mask_list.append(mask_data)
                    unique_masks = np.unique(mask_data)
                    if len(unique_masks) > 1 or unique_masks[0] != 0:
                        logging.info(f"  {band}_MASK has non-zero values: {unique_masks}")
                else:
                    mask_list.append(np.zeros(len(wave_data), dtype=np.uint32))
                    logging.info(f"  {band}_MASK not found, assuming all pixels are good")

                # 记录该波段的波长范围（未提供配置值时从数据计算）
                if arm_wavelength_ranges is None:
                    band_wavelengths.append([float(wave_data.min()), float(wave_data.max())])

            # 校验各波段噪声数据一致性（不允许混合 IVAR/SNR）
            all_have_ivar = all(x is not None for x in ivar_list)
            all_have_snr = all(x is not None for x in snr_list)
            if not all_have_ivar and any(x is not None for x in ivar_list):
                raise ValueError(
                    "部分波段有 IVAR 而部分没有，请确保所有波段的 IVAR 数据一致。"
                    f"IVAR 状态: {['有' if x is not None else '无' for x in ivar_list]}"
                )
            if not all_have_snr and any(x is not None for x in snr_list):
                raise ValueError(
                    "部分波段有 SNR 而部分没有，请确保所有波段的 SNR 数据一致。"
                    f"SNR 状态: {['有' if x is not None else '无' for x in snr_list]}"
                )

            # 合并数组
            wavelength = np.concatenate(wavelength_list)
            flux = np.concatenate(flux_list)

            if has_ivar:
                ivar = np.concatenate(ivar_list)  # 全部非 None，直接拼接
            else:
                ivar = None

            if has_snr:
                snr = np.concatenate(snr_list)  # 全部非 None，直接拼接
            else:
                snr = None

            specmask = np.concatenate(mask_list)

            logging.info(f"Combined {len(wavelength)} wavelength points from {len(band_names)} arms")
            logging.info(f"Wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} Å")

            # 统计整体 mask 情况
            n_masked = np.sum(specmask != 0)
            if n_masked > 0:
                logging.info(f"SPECMASK: {n_masked} pixels ({100*n_masked/len(specmask):.2f}%) have quality issues")
            else:
                logging.info("SPECMASK: All pixels are clean (mask=0)")

            # === 构建 quality mask ===
            if has_ivar and ivar is not None:
                quality_mask = (specmask == 0) & (ivar > 0)
            else:
                # 仅有 SNR 的情况（如 CSST）：mask 为 0 即表示好像素
                quality_mask = (specmask == 0)

            # === 引入 arm overlap mask ===
            overlap_regions = find_overlap_regions(band_names, band_wavelengths)

            keep_mask = quality_mask.copy()

            if overlap_regions:
                logging.info(f"Detected {len(overlap_regions)} overlap regions: {overlap_regions}")
                for region_name, (ov_start, ov_end) in overlap_regions.items():
                    in_overlap = (wavelength >= ov_start) & (wavelength <= ov_end)
                    keep_mask &= ~in_overlap
                    logging.info(f"  Masking overlap '{region_name}': [{ov_start:.2f}, {ov_end:.2f}], "
                                f"{in_overlap.sum()} points removed")

            # 应用 mask
            wavelength_masked = wavelength[keep_mask]
            flux_masked = flux[keep_mask]
            ivar_masked = ivar[keep_mask] if ivar is not None else None
            snr_masked = snr[keep_mask] if snr is not None else None

            # 排序 + 同波长去重（取均值）
            sort_idx = np.argsort(wavelength_masked)
            wavelength_masked = wavelength_masked[sort_idx]
            flux_masked = flux_masked[sort_idx]
            ivar_masked = ivar_masked[sort_idx] if ivar_masked is not None else None
            snr_masked = snr_masked[sort_idx] if snr_masked is not None else None

            # 合并相同波长：取 flux / ivar / snr 的均值
            _, unique_idx, dup_counts = np.unique(
                wavelength_masked, return_index=True, return_counts=True
            )
            if len(unique_idx) < len(wavelength_masked):
                n_dup = len(wavelength_masked) - len(unique_idx)
                logging.warning(f"发现 {n_dup} 个重复波长点，将对 flux/ivar/snr 取均值")
                new_wl = wavelength_masked[unique_idx]
                new_flux = np.zeros(len(unique_idx))
                new_ivar = np.zeros(len(unique_idx)) if ivar_masked is not None else None
                new_snr = np.zeros(len(unique_idx)) if snr_masked is not None else None
                for j, (start, count) in enumerate(
                    zip(unique_idx, dup_counts)
                ):
                    sl = slice(start, start + count)
                    new_flux[j] = np.mean(flux_masked[sl])
                    if new_ivar is not None:
                        new_ivar[j] = np.mean(ivar_masked[sl])
                    if new_snr is not None:
                        new_snr[j] = np.mean(snr_masked[sl])
                wavelength_masked = new_wl
                flux_masked = new_flux
                ivar_masked = new_ivar
                snr_masked = new_snr

            n_removed = len(wavelength) - len(wavelength_masked)
            n_quality_removed = np.sum(~quality_mask)
            logging.info(f"After masking: {len(wavelength_masked)} points remain "
                        f"({n_removed} removed, including {n_quality_removed} from SPECMASK)")

            if has_snr and snr_masked is not None:
                effective_snr = snr_masked.astype(np.float64)
            else:
                effective_snr = None

        else:
            # === 单表格格式 ===
            logging.info("Detected single-table FITS format")
            is_multi_arm = False
            overlap_regions = None

            data = None
            for i, hdu in enumerate(hdul):
                if hdu.data is not None:
                    data = hdu.data
                    logging.info(f"Found data in HDU {i}")
                    break

            if data is None:
                raise ValueError(f"No data found in FITS file: {fits_path}")

            if hasattr(data, 'dtype') and hasattr(data.dtype, 'names') and data.dtype.names is not None:
                col_names = data.dtype.names
                logging.info(f"FITS columns: {col_names}")

                wavelength_col = None
                flux_col = None
                snr_col = None
                ivar_col = None

                for name in col_names:
                    name_upper = name.upper()
                    if name_upper in ['WAVELENGTH', 'WAVE', 'LAMBDA']:
                        wavelength_col = name
                    elif name_upper in ['FLUX', 'F']:
                        flux_col = name
                    elif name_upper in ['SNR', 'SIGNAL_TO_NOISE']:
                        snr_col = name
                    elif name_upper in ['IVAR', 'INVERSE_VARIANCE']:
                        ivar_col = name

                if wavelength_col is None or flux_col is None:
                    raise ValueError(f"Required columns not found. Available: {col_names}")

                wavelength = np.array(data[wavelength_col])
                flux = np.array(data[flux_col])

                if snr_col is not None:
                    effective_snr = np.array(data[snr_col])
                    ivar = None
                    has_ivar = False
                elif ivar_col is not None:
                    ivar = np.array(data[ivar_col])
                    effective_snr = None
                    has_ivar = True
                else:
                    effective_snr = np.full_like(flux, 5.0, dtype=float)
                    ivar = None
                    has_ivar = False
            else:
                # 普通数组
                wavelength = data[:, 0]
                flux = data[:, 1]
                effective_snr = np.full_like(flux, 5.0, dtype=float)
                ivar = None
                has_ivar = False

        # 确保是 numpy 数组
        wavelength = np.array(wavelength, dtype=np.float64)
        flux = np.array(flux, dtype=np.float64)

        if has_ivar and ivar is not None:
            ivar = np.array(ivar, dtype=np.float64)

        # 多波段格式：使用 mask 后的数据
        if is_multi_arm:
            new_wavelength = np.array(wavelength_masked, dtype=np.float64)
            weighted_flux = np.array(flux_masked, dtype=np.float64)
            ivar_final = np.array(ivar_masked, dtype=np.float64) if ivar_masked is not None else None
        else:
            new_wavelength = wavelength
            weighted_flux = flux
            ivar_final = ivar if has_ivar and ivar is not None else None

        # 处理 effective_snr
        if effective_snr is not None:
            effective_snr = np.array(effective_snr, dtype=np.float64)
        elif ivar_final is not None:
            ivar_safe = np.maximum(ivar_final, 0)
            effective_snr = weighted_flux * np.sqrt(ivar_safe)
        else:
            effective_snr = np.full_like(weighted_flux, 5.0, dtype=np.float64)

        # smooth the spectrum
        # weighted_flux = gaussian_filter1d(weighted_flux, sigma=2)

        # 构建 spectrum_dict
        spectrum_dict = {
            'wavelength': new_wavelength.tolist(),
            'flux': weighted_flux.tolist(),
            'snr': effective_snr.tolist(),
            'ivar': ivar_final.tolist() if ivar_final is not None else None,
            'VI_Z': vi_z,
            'VI_SPECTYPE': vi_spectype,
        }

        return spectrum_dict

def generate_continuum_description(
    wavelengths: np.ndarray,
    continuum_flux: np.ndarray,
    monotonic_regions: np.ndarray
) -> str:
    """
    Generate a natural language description of the continuum based on monotonic intervals.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array
    continuum_flux : np.ndarray
        Continuum flux array
    monotonic_regions : np.ndarray
        Indices where monotonicity changes (from np.where(np.diff(np.sign(...))))
    
    Returns
    -------
    description : str
        Natural language description of the continuum, e.g.:
        "The continuum has a value of 2.027 at 4000.0 Å, monotonically increases
         from 4000.0 Å to 5010.1 Å, reaching 3.499 at 5010.1 Å; ..."
    """
    # monotonic_regions already computed by the caller (run_continuum_fitting);
    # derive per-segment sign directly from flux endpoints to avoid recomputing the derivative.
    # sign > 0: increasing, sign < 0: decreasing, == 0: flat
    boundaries = np.concatenate([[0], monotonic_regions + 1, [len(wavelengths) - 1]])
    segment_signs = np.sign(
        continuum_flux[boundaries[1:]] - continuum_flux[boundaries[:-1]]
    )
    
    # Generate descriptions
    descriptions = []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        # Get wavelength and flux at interval endpoints
        lambda_start = wavelengths[start_idx]
        lambda_end = wavelengths[end_idx]
        flux_start = continuum_flux[start_idx]
        flux_end = continuum_flux[end_idx]
        
        # Determine monotonicity from segment sign
        seg_sign = segment_signs[i]
        if seg_sign > 0:
            monotonic_str = 'monotonically increases'
        elif seg_sign < 0:
            monotonic_str = 'monotonically decreases'
        else:
            monotonic_str = 'remains approximately flat'
        
        # Build description sentence
        if start_idx == 0:
            # First interval
            desc = f'The continuum has a value of {flux_start:.3f} at {lambda_start:.1f} Å, '
            desc += f'{monotonic_str} from {lambda_start:.1f} Å to {lambda_end:.1f} Å, '
            desc += f'reaching {flux_end:.3f} at {lambda_end:.1f} Å'
        else:
            # Subsequent intervals
            desc = f'{monotonic_str} from {lambda_start:.1f} Å to {lambda_end:.1f} Å, '
            desc += f'reaching {flux_end:.3f} at {lambda_end:.1f} Å'
        
        descriptions.append(desc)
    
    # Join into full description
    full_description = '; '.join(descriptions) + '.'
    
    return full_description


# ===========================================================
# Chebyshev 阶数自适应选择（原 feature_finder_precise.py）
# ===========================================================

def select_chebyshev_degree(sp, min_degree=1, max_degree=10, verbose=False):
    """
    自动选择最佳切比雪夫多项式阶数。
    准则：残差 MAD 小 + 偏度接近理想值（0.5）+ 复杂度惩罚。
    """
    from scipy.stats import skew
    from specutils.fitting import fit_generic_continuum
    from astropy.modeling import models

    flux = sp.flux.value
    best_degree = min_degree
    best_score = np.inf

    for deg in range(min_degree, max_degree + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cf = fit_generic_continuum(sp, model=models.Chebyshev1D(degree=deg))
            res = flux - cf(sp.spectral_axis).value
            mask = np.abs(res) < np.percentile(np.abs(res), 95)
            rm = res[mask]
            res_skew = abs(skew(rm))
            res_mad = np.median(np.abs(rm - np.median(rm)))
            score = res_mad * 10.0 + (res_skew - 0.5) ** 2 * 5.0 + deg * 0.2
            if verbose:
                print(f"  degree={deg}: skew={res_skew:.4f}, MAD={res_mad:.4e}, score={score:.4f}")
            if score < best_score:
                best_score = score
                best_degree = deg
        except Exception as e:
            if verbose:
                print(f"  degree={deg}: 拟合失败 - {e}")
    if verbose:
        print(f"最佳阶数: {best_degree}")
    return best_degree


# ===========================================================
# File I/O Helpers
# ===========================================================

def run_continuum_fitting_masked(
    wavelengths,
    flux,
    peaks=None,
    troughs=None,
    chebyshev_degree=None,
    chebyshev_min_degree=1,
    chebyshev_max_degree=10,
    verbose=False,
):
    """
    在将外检测到的峰/谷区域（中心 ±3σ）mask 掉后，对剩余光谱做切比雪夫拟合。
    拟合完成后用完整波长数组插値，返回 continuum_dict 和 residual_spectrum_dict，
    格式与原 run_continuum_fitting 完全相同。

    Parameters
    ----------
    wavelengths : array-like
        波长数组（Å）
    flux : array-like
        流量数组
    peaks : list of dict, optional
        发射线检测结果（records 格式），包含 'wavelength' 和 'FWHM_A' 字段
    troughs : list of dict, optional
        吸收线检测结果（records 格式），包含 'wavelength' 和 'FWHM_A' 字段
    chebyshev_degree : int or None
        指定切比雪夫多项式阶数；若为 None 则自动选择
    chebyshev_min_degree : int
    chebyshev_max_degree : int
    verbose : bool

    Returns
    -------
    (continuum_dict, residual_spectrum_dict) : tuple[dict, dict]
    """
    from specutils.fitting import fit_generic_continuum
    from specutils import Spectrum
    from astropy.modeling import models
    import astropy.units as u
    import warnings

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    # ── 构建特征区域 mask：中心 ±3σ 的点均被排除 ───────────────
    # 与迭代检测内部的 mask 逻辑保持一致：中心 ±3σ，其中 σ = FWHM_A / SIGMA_TO_FWHM
    fit_mask = np.ones(len(wavelengths), dtype=bool)  # True = 参与拟合

    all_features = []
    if peaks:
        all_features.extend(peaks)
    if troughs:
        all_features.extend(troughs)

    n_masked_pts = 0
    for feat in all_features:
        wl_center = feat.get('wavelength')
        fwhm_a = feat.get('FWHM_A')
        if wl_center is None or fwhm_a is None:
            continue
        sigma = fwhm_a / SIGMA_TO_FWHM
        lo, hi = wl_center - 3 * sigma, wl_center + 3 * sigma
        in_region = (wavelengths >= lo) & (wavelengths <= hi)
        n_masked_pts += int(in_region.sum())
        fit_mask &= ~in_region

    n_fit_pts = int(fit_mask.sum())
    print(f'[Continuum masked] 共 {len(all_features)} 个特征，'
          f'mask掉 {n_masked_pts} 个数据点，'
          f'参与拟合点数: {n_fit_pts}/{len(wavelengths)}')

    # 少于 20 个点时回退到未掉点的拟合
    if n_fit_pts < 20:
        print('[Continuum masked] 可用点数不足 20，回退为全谱拟合')
        fit_mask = np.ones(len(wavelengths), dtype=bool)

    wave_fit = wavelengths[fit_mask]
    flux_fit = flux[fit_mask]

    # ── 在 mask 后的子光谱上自动选阶和拟合 ────────────────────────
    sp_fit = Spectrum(flux=flux_fit * u.Jy, spectral_axis=wave_fit * u.AA)

    if chebyshev_degree is None:
        chebyshev_degree = select_chebyshev_degree(
            sp_fit, min_degree=chebyshev_min_degree, max_degree=chebyshev_max_degree,
            verbose=verbose
        )

    print(f'[Continuum masked] Chebyshev degree={chebyshev_degree}')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cf = fit_generic_continuum(sp_fit, model=models.Chebyshev1D(degree=chebyshev_degree))

    # 在完整波长数组上求値（包括被 mask 的点）
    continuum_flux = cf(wavelengths * u.AA).value

    # ── 计算单调区间和自然语言描述 ───────────────────────────────
    continuum_flux_safe = np.maximum(np.abs(continuum_flux), 1e-10)
    continuum_log_derivative = np.gradient(continuum_flux, wavelengths) / continuum_flux_safe
    monotonic_regions = np.where(np.diff(np.sign(continuum_log_derivative)))[0]
    continuum_description = generate_continuum_description(wavelengths, continuum_flux, monotonic_regions)
    print(continuum_description)

    continuum_dict = {
        'wavelength': wavelengths.tolist(),
        'flux': continuum_flux.tolist(),
        'chebyshev_degree': chebyshev_degree,
        'description': continuum_description,
        'n_masked_features': len(all_features),
        'n_masked_points': n_masked_pts,
    }

    residual_flux = flux - continuum_flux
    residual_spectrum_dict = {
        'wavelength': wavelengths.tolist(),
        'flux': residual_flux.tolist()
    }

    return continuum_dict, residual_spectrum_dict




# ===========================================================
# Brute-Force Line Matching
# ============================================================

# 静止系谱线表
EMISSION_LINES = {
    # 高电离 / AGN 特征线
    "Lyα":       1216.0,
    "C IV":      1549.0,
    "He II":     1640.4,
    "C III]":    1909.0,
    "Mg II":     2800.0,
    "[Ne V]":    3426.0,
    "[O II]":    3727.0,
    # Balmer 系列
    "Hε":        3970.1,
    "Hδ":        4102.9,
    "Hγ":        4341.7,
    "Hβ":        4862.7,
    # 窄线区
    "[O III]a":  4960.3,
    "[O III]b":  5008.2,
    "[N II]a":   6549.8,
    "Hα":        6564.6,
    "[N II]b":   6585.3,
    "[S II]a":   6718.3,
    "[S II]b":   6732.7,
}

# 发射线宽窄分类：broad = 宽线区 (BLR) 允许的宽线，narrow = 窄线区 (NLR) 典型窄线
# both = BLR+NLR 均可产生，宽窄皆合理（Balmer 系在 QSO 中有 broad+narrow 叠加，
#        在 galaxy 中仅 narrow），宽度校验对 both 类跳过
# 用于匹配时检查寻峰宽度与物理期望是否一致
EMISSION_LINE_WIDTHS = {
    # BLR 宽线
    "Lyα":       "broad",
    "C IV":      "broad",
    "C III]":    "broad",
    "He II":     "both", # 在 QSO 中可以表现为 BLR 宽线，在低电离 AGN 或 Galaxy 中也可以是较窄的线
    "Mg II":     "broad",
    # Balmer 系列：QSO 中 broad+narrow 叠加，galaxy 中仅 narrow
    "Hε":        "both",
    "Hδ":        "both",
    "Hγ":        "both",
    "Hβ":        "both",
    "Hα":        "both",
    # NLR 窄线
    "[Ne V]":    "narrow",
    "[O II]":    "narrow",
    "[O III]a":  "narrow",
    "[O III]b":  "narrow",
    "[N II]a":   "narrow",
    "[N II]b":   "narrow",
    "[S II]a":   "narrow",
    "[S II]b":   "narrow",
}

ABSORPTION_LINES = {
    "Ca K_abs":      3934.8,
    "Ca H_abs":      3969.6,
    "G-band_abs":    4305.6,
    "Mg I_abs":        5176.7,
    "Mg II_abs":     2800.0,
    "Na D_abs":      5895.6,
    "CaT1_abs":      8498.0,
    "CaT2_abs":      8542.0,
    "CaT3_abs":      8662.0,
    # Balmer 吸收
    "Hε_abs":    3970.1,
    "Hδ_abs":    4102.9,
    "Hγ_abs":    4341.7,
    "Hβ_abs":    4862.7,
    "Hα_abs":    6564.6,
}

# Mg II_abs ≈ BLR 宽吸收线；其余为恒星/ISM 窄吸收
ABSORPTION_LINE_WIDTHS = {
    "Mg II_abs": "broad",
    # 其余默认 "absorption"（窄），不在表中列出
}

# 锚定时只用发射线假设
# Mg II 2800 / H 系列既可发射也可吸收，但锚定时统一按发射线处理
ANCHOR_EMISSION_LINES = EMISSION_LINES

# LRG/BGS 专用锚定谱线表：排除 BLR 宽线（broad），保留窄线和 Balmer both 类
# 派生自 EMISSION_LINE_WIDTHS，EMISSION_LINE_WIDTHS 是唯一宽窄分类来源，改一处全部生效
LRG_ANCHOR_EMISSION_LINES = {
    k: v for k, v in EMISSION_LINES.items()
    if EMISSION_LINE_WIDTHS.get(k) != 'broad'
}



# ═══════════════════════════════════════════════════════════════════════════════
# brute_force_line_matching — 已弃用（2026-07-06）
#   仅用于 REDROCK=false 路径。当前默认 REDROCK=true，使用 run_redrock_pipeline()。
#   如需恢复，取消 VisualInterpreter.py 中 import 和调用处的注释即可。
# ═══════════════════════════════════════════════════════════════════════════════
def brute_force_line_matching(state, tol_wavelength=None):
    """
    暴力破解：对每个峰/谷假设其为某条发射/吸收线，计算红移，
    再用该红移预测所有其他谱线位置，匹配峰列表和谷列表。
    最后按 (波长, 线名) 对做 Union-Find 去重，合并属于同一物理场景的假设。

    参数
    ----
    state : dict
        必须包含 state['peaks'] 和 state['troughs']，
        每个元素需有 'wavelength' 键。
    tol_wavelength : float, optional
        匹配容差 (Å)。若为 None 则默认为 10.0。

    返回
    ----
    dict with keys:
        z : list[float]           — 所有代表红移（去重排序）
        zmedian : float           — 代表红移中位数
        hypotheses : list[dict]   — 匹配结果，每个元素参考一个物理场景
    """

    if tol_wavelength is None:
        tol_wavelength = 10.0

    # ── 提取光谱数据，供组内 scoring 使用 ─────────────────────
    spec = state.get('spectrum', {})
    scoring_wave = np.asarray(spec.get('wavelength', []), dtype=np.float64)
    scoring_flux = np.asarray(spec.get('flux', []), dtype=np.float64)
    scoring_snr = np.asarray(spec.get('snr', []), dtype=np.float64)
    continuum = state.get('continuum', {})
    continuum_flux = np.asarray(continuum.get('flux', []), dtype=np.float64)
    _has_scoring_data = (
        len(scoring_wave) > 0 and len(scoring_flux) > 0
        and len(continuum_flux) > 0 and len(scoring_snr) > 0
    )

    # [deprecated] 3-mode branch removed — now uses all peaks + all troughs in single pass.
    # Kept for rollback reference:
    #
    # if mode == 'lrg_bgs':
    #     active_peaks = [p for p in state['peaks'] if p.get('width_class') == 'narrow']
    #     active_anchor_lines = LRG_ANCHOR_EMISSION_LINES
    #     active_emission_lines = LRG_ANCHOR_EMISSION_LINES
    # elif mode == 'elg':
    #     active_peaks = [p for p in state['peaks'] if p.get('width_class') == 'narrow']
    #     active_anchor_lines = ANCHOR_EMISSION_LINES
    #     active_emission_lines = EMISSION_LINES
    # else:  # mode == 'qso'
    #     active_peaks = state['peaks']
    #     active_anchor_lines = ANCHOR_EMISSION_LINES
    #     active_emission_lines = EMISSION_LINES

    active_peaks = state['peaks']
    active_anchor_lines = ANCHOR_EMISSION_LINES
    active_emission_lines = EMISSION_LINES

    peak_wavelengths = [p['wavelength'] for p in active_peaks]
    trough_wavelengths = [t['wavelength'] for t in state['troughs']]
    trough_wl_set = set(trough_wavelengths)  # 用于区分 emission/absorption

    # 观测波长范围（用于 Missing Emission Lines 判断）
    try:
        _wl_arr = state['spectrum']['wavelength']
        obs_wl_min = float(_wl_arr[0])
        obs_wl_max = float(_wl_arr[-1])
    except Exception:
        obs_wl_min = None
        obs_wl_max = None

    # ── 构建波长 → 峰/谷特征信息的查找表 ────────────────────
    peak_info = {}   # wavelength -> {amplitude, FWHM_A, FWHM_km_s, width_class, snr, ridge_length}
    for p in active_peaks:
        peak_info[p['wavelength']] = {
            'amplitude': p.get('amplitude'),
            'FWHM_A': p.get('FWHM_A'),
            'FWHM_km_s': p.get('FWHM_km_s'),
            'width_class': p.get('width_class'),
            'snr': p.get('snr'),
            'ridge_length': p.get('ridge_length'),
        }
    trough_info = {}
    for t in state['troughs']:
        trough_info[t['wavelength']] = {
            'amplitude': t.get('amplitude'),
            'FWHM_A': t.get('FWHM_A'),
            'FWHM_km_s': t.get('FWHM_km_s'),
            'width_class': t.get('width_class'),
            'snr': t.get('snr'),
            'ridge_length': t.get('ridge_length'),
        }

    def _fmt_feature(wl, info):
        """格式化单条峰/谷的宽高信息片段，数值保留 3 位小数"""
        if info is None:
            return ""
        amp = info.get('amplitude')
        fwhm_a = info.get('FWHM_A')
        fwhm_kms = info.get('FWHM_km_s')
        wc = info.get('width_class')
        parts = []
        if amp is not None:
            parts.append(f"Amp={amp:.3f}")
        if fwhm_a is not None or fwhm_kms is not None:
            w_parts = []
            if fwhm_a is not None:
                w_parts.append(f"{fwhm_a:.3f}Å")
            if fwhm_kms is not None:
                w_parts.append(f"{fwhm_kms:.3f} km/s")
            parts.append(f"W={'/'.join(w_parts)}")
        if wc is not None:
            parts.append(wc)
        if not parts:
            return ""
        return f" ({', '.join(parts)})"

    # ── 第一阶段：逐峰逐线暴力匹配，收集原始结果 ──────────────
    # 每个原始假设产出一组 (wavelength, line_name, z) 三元组
    # raw_results: [{anchor_wl, anchor_line, anchor_z, pairs: [(wl, line, z), ...]}]

    raw_results = []

    for peak_wl in peak_wavelengths:
        for line_name, line_rest in active_anchor_lines.items():
            z = peak_wl / line_rest - 1.0
            if z < 0 or z > 10:
                continue
            z_rounded = round(z, 3)

            pairs = []  # (obs_wavelength, line_name, z)

            # 锚定线自身
            pairs.append((peak_wl, line_name, z_rounded))

            # 发射线匹配
            for ename, erest in active_emission_lines.items():
                if ename == line_name:
                    continue
                lambda_theory = erest * (1.0 + z)
                for pwl in peak_wavelengths:
                    delta = abs(pwl - lambda_theory)
                    if delta <= tol_wavelength:
                        pair_z = round(pwl / erest - 1.0, 3)
                        pairs.append((pwl, ename, pair_z))

            # 吸收线匹配
            for aname, arest in ABSORPTION_LINES.items():
                lambda_theory = arest * (1.0 + z)
                for twl in trough_wavelengths:
                    delta = abs(twl - lambda_theory)
                    if delta <= tol_wavelength:
                        pair_z = round(twl / arest - 1.0, 3)
                        pairs.append((twl, aname, pair_z))

            raw_results.append({
                'anchor_wl': peak_wl,
                'anchor_line': line_name,
                'anchor_z': z_rounded,
                'pairs': pairs,
            })

    # ── 第一阶段补充：逐谷锚定 ──────────────────────
    # 对每个谷，假设它是某条吸收线，计算红移，再验证其余谷和峰是否吻合
    # [deprecated] was guarded by `if mode == 'lrg_bgs':` — now always active
    for trough_wl in trough_wavelengths:
        for aname, arest in ABSORPTION_LINES.items():
            z = trough_wl / arest - 1.0
            if z < 0 or z > 10:
                continue
            z_rounded = round(z, 3)

            pairs = []

            # 锚定线自身
            pairs.append((trough_wl, aname, z_rounded))

            # 验证其余吸收谷
            for aname2, arest2 in ABSORPTION_LINES.items():
                if aname2 == aname:
                    continue
                lambda_theory = arest2 * (1.0 + z)
                for twl in trough_wavelengths:
                    delta = abs(twl - lambda_theory)
                    if delta <= tol_wavelength:
                        pair_z = round(twl / arest2 - 1.0, 3)
                        pairs.append((twl, aname2, pair_z))

            # 验证峰（全部发射线）
            for ename, erest in active_emission_lines.items():
                lambda_theory = erest * (1.0 + z)
                for pwl in peak_wavelengths:
                    delta = abs(pwl - lambda_theory)
                    if delta <= tol_wavelength:
                        pair_z = round(pwl / erest - 1.0, 3)
                        pairs.append((pwl, ename, pair_z))

            raw_results.append({
                'anchor_wl': trough_wl,
                'anchor_line': aname,
                'anchor_z': z_rounded,
                'pairs': pairs,
            })


    # ── 第二阶段：Union-Find 去重 ─────────────────────────────
    # 节点 = (wavelength, line_name) 对
    # 同一假设中的所有节点互连

    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 收集所有节点并初始化
    for hyp in raw_results:
        for wl, lname, _ in hyp['pairs']:
            node = (wl, lname)
            if node not in parent:
                parent[node] = node

    # 同一假设内所有节点 union
    for hyp in raw_results:
        pair_nodes = [(wl, lname) for wl, lname, _ in hyp['pairs']]
        if len(pair_nodes) < 2:
            continue
        first = pair_nodes[0]
        for node in pair_nodes[1:]:
            union(first, node)

    # 按根节点分组
    groups = {}  # root -> set of (wl, line_name)
    for node in parent:
        root = find(node)
        groups.setdefault(root, set()).add(node)

    # ── 第三阶段：每个连通分量格式化为 LLM 可读输出 ───────────

    results = []

    for root, node_set in groups.items():
        # 收集组内所有 (wl, line_name, z) 信息
        # 需要从 raw_results 中查找每个 (wl, line_name) 对应的 z
        pair_z_map = {}  # (wl, line_name) -> z
        for hyp in raw_results:
            for wl, lname, z in hyp['pairs']:
                node = (wl, lname)
                if node in node_set:
                    if node not in pair_z_map:
                        pair_z_map[node] = z

        # 区分发射线和吸收线：以波长来源（peak/trough）为准，
        # 而非仅凭谱线名——Mg II 等线同时存在于发射线表与吸收线表
        emission_nodes = []
        absorption_nodes = []
        for (wl, lname), z in pair_z_map.items():
            if wl in trough_wl_set:
                absorption_nodes.append((wl, lname, z))
            else:
                emission_nodes.append((wl, lname, z))

        # 按波长升序排列
        emission_nodes.sort(key=lambda x: x[0])
        absorption_nodes.sort(key=lambda x: x[0])

        # z 统计
        all_z = list(pair_z_map.values())
        if not all_z:
            continue
        z_max = max(all_z)
        z_min = min(all_z)
        z_spread = round(z_max - z_min, 3)

        # Hypothesis：列出组内所有锚定对（从 raw_results 中找 anchor）
        anchor_pairs = []
        for hyp in raw_results:
            anchor_node = (hyp['anchor_wl'], hyp['anchor_line'])
            if anchor_node in node_set:
                anchor_pairs.append((hyp['anchor_wl'], hyp['anchor_line']))
        # 去重，按波长排序，同一波长的多条谱线合并为 wl-line1/line2 格式
        anchor_pairs = sorted(set(anchor_pairs), key=lambda x: x[0])
        # 按波长分组，将同一波长的谱线名用 / 连接；谱线名按静止系波长升序排列
        # 静止系波长查找表：合并发射线表与吸收线表
        _all_line_rest = {**EMISSION_LINES, **ABSORPTION_LINES}
        _anchor_wl_lines = defaultdict(list)
        for wl, lname in anchor_pairs:
            if lname not in _anchor_wl_lines[wl]:
                _anchor_wl_lines[wl].append(lname)
        hypothesis_str = ", ".join(
            f"{wl:.3f}-{'/'.join(sorted(lnames, key=lambda n: _all_line_rest.get(n, 0)))}"
            for wl, lnames in sorted(_anchor_wl_lines.items())
        )

        # 格式化 Emission / Absorption matches（嵌入寻峰/寻谷的宽高信息）
        emission_formatted = []
        for wl, lname, z in emission_nodes:
            feat = _fmt_feature(wl, peak_info.get(wl))
            # 宽度不匹配检查：narrow 峰匹配 broad 线 或 broad 峰匹配 narrow 线
            # "both" 类（Balmer 系）宽窄皆合理，跳过校验
            width_warn = ""
            pinfo = peak_info.get(wl)
            if pinfo is not None:
                pwc = pinfo.get('width_class')
                lwc = EMISSION_LINE_WIDTHS.get(lname)
                if (pwc is not None and lwc is not None
                        and lwc != 'both'
                        and pwc != 'intermediate'
                        and pwc != lwc):
                    width_warn = " ⚠ width mismatch"
            emission_formatted.append(f"{wl:.3f} Å{feat} → {lname} (z={z}){width_warn}")

        absorption_formatted = []
        for wl, lname, z in absorption_nodes:
            feat = _fmt_feature(wl, trough_info.get(wl))
            absorption_formatted.append(f"{wl:.3f} Å{feat} → {lname} (z={z})")

        # N_emission / N_absorption: min(独立峰/谷波长数, 独立线名数)
        # 同时处理「一峰配多线」(峰少线多→取峰数) 和「一线配多峰」(线少峰多→取线数)
        # 两种情况都会虚高朴素计数，min 保证有效独立约束数不被高估。
        n_em = min(
            len({wl    for wl, _, _    in emission_nodes}),   # 独立峰值波长数
            len({lname for _, lname, _ in emission_nodes}),   # 独立发射线名数
        )
        n_ab = min(
            len({wl    for wl, _, _    in absorption_nodes}),  # 独立谷值波长数
            len({lname for _, lname, _ in absorption_nodes}),  # 独立吸收线名数
        )

        # [deprecated] Redshift warning / Missing lines / Observable lines / Dn4000
        # removed — kept below as comment for rollback reference:
        #
        # # Redshift warning（用 z_max 比较）
        # low_z_parts = []
        # if z_max < min_qso_redshift:
        #     low_z_parts.append("QSO")
        # if z_max < min_galaxy_redshift:
        #     low_z_parts.append("Galaxy")
        # redshift_warning = f"z too low for {' and '.join(low_z_parts)}" if low_z_parts else None
        #
        # ... (Missing/Observable lines / Dn4000 omitted for brevity, see git history)

        z_center = round(float(np.median(all_z)), 4)
        z_list = sorted(set(round(float(z), 4) for z in all_z))

        # ── Build matched_lines: {line_name -> feature_info} ──────
        matched_lines = {}
        for wl, lname, z in emission_nodes:
            feat = dict(peak_info.get(wl, {}))
            feat['wavelength'] = wl
            matched_lines[lname] = feat
        for wl, lname, z in absorption_nodes:
            feat = dict(trough_info.get(wl, {}))
            feat['wavelength'] = wl
            matched_lines[lname] = feat

        results.append({
            "Hypothesis": hypothesis_str,
            "z_center": z_center,
            "z_list": z_list,
            "z_max": z_max,
            "z_min": z_min,
            "z_spread": z_spread,
            "Emission matches": emission_formatted,
            "Absorption matches": absorption_formatted,
            "N_emission": n_em,
            "N_absorption": n_ab,
            "matched_lines": matched_lines,
        })

    # 按 N_emission + N_absorption 降序排列
    results.sort(key=lambda r: r['N_emission'] + r['N_absorption'], reverse=True)

    # 过滤掉有效独立约束数 < 2 的候选
    results = [r for r in results if r['N_emission'] + r['N_absorption'] >= 1]

    # ── 组内 scoring v2：对每个假设的 z_list 逐 z 打分，选最高分作为代表红移 ──
    flux_norm = None
    if _has_scoring_data and results:
        flux_norm = scoring_flux / np.maximum(continuum_flux, 1e-6)

    for r in results:
        if flux_norm is None or not r['z_list']:
            r['z_representative'] = r['z_center']
            r['score'] = float(r['N_emission'] + r['N_absorption'])
            r['scoring_details'] = []
            continue

        best_z = r['z_center']
        best_score = -1.0
        best_details = []
        for z in r['z_list']:
            s, n_ib, det = _score_one_redshift(
                z, scoring_wave, flux_norm, scoring_snr, half=20.0,
            )
            if s > best_score:
                best_score = s
                best_z = z
                best_details = det

        r['z_representative'] = best_z
        r['score'] = round(float(best_score), 4)
        r['scoring_details'] = [
            (name, morph, round(float(sc), 4), round(float(pred), 1),
             round(float(obs), 1))
            for name, morph, sc, pred, obs, *_ in best_details
        ]

    # ── 构建顶层 z / zmedian（用代表红移）──────────────────────
    all_z_rep = sorted(set(
        r['z_representative'] for r in results
        if r.get('z_representative') is not None
    ))
    zmedian = round(float(np.median(all_z_rep)), 4) if all_z_rep else None

    return {
        "z": all_z_rep,
        "zmedian": zmedian,
        "hypotheses": results,
    }


# =============================================================================
# Redshift scoring — rank brute-force hypotheses for LLM triage
# =============================================================================

# 完整静止系谱线表 (rest_wavelength, name, abs_weight, em_weight, universality)
_SCORER_LINES = [
    (1216.0,  'Lyα',        0.0, 3.0, 1.5),
    (1549.0,  'C IV',       0.0, 3.0, 1.0),
    (1640.4,  'He II',      0.0, 1.5, 0.5),
    (1909.0,  'C III]',     0.0, 2.5, 1.0),
    (2800.0,  'Mg II',      0.5, 3.0, 2.0),
    (3426.0,  'Ne V',       0.0, 1.0, 0.5),
    (3727.0,  '[O II]',     0.0, 3.0, 2.0),
    (3970.1,  'Hε',         1.0, 0.5, 2.0),
    (4102.9,  'Hδ',         1.0, 0.5, 2.0),
    (4341.7,  'Hγ',         1.0, 0.5, 2.0),
    (4862.7,  'Hβ',         2.0, 2.5, 3.0),
    (6564.6,  'Hα',         1.5, 2.5, 3.0),
    (3934.8,  'Ca K',       3.0, 0.0, 2.5),
    (3969.6,  'Ca H',       3.0, 0.0, 2.5),
    (4305.6,  'G-band',     2.0, 0.0, 2.0),
    (5176.7,  'Mg I',       2.0, 0.0, 2.0),
    (5895.6,  'Na D',       3.0, 0.0, 2.5),
    (4960.3,  '[O III]a',   0.0, 2.5, 1.5),
    (5008.2,  '[O III]b',   0.0, 3.0, 2.0),
    (6549.8,  '[N II]a',    0.0, 2.0, 1.5),
    (6585.3,  '[N II]b',    0.0, 2.5, 1.5),
    (6718.3,  '[S II]a',    0.0, 1.5, 1.0),
    (6732.7,  '[S II]b',    0.0, 1.5, 1.0),
    (8498.0,  'CaT1',       2.5, 0.0, 2.0),
    (8542.0,  'CaT2',       2.5, 0.0, 2.0),
    (8662.0,  'CaT3',       2.5, 0.0, 2.0),
]


def _score_one_redshift(z, wave, flux_norm, snr, half=20.0, min_wavelength=0.0):
    """对单个 z 打分（norm 方法 + 归一化），返回 (score, n_in_band, details).

    min_wavelength: 跳过预测波长低于此值的谱线 (Å)。用于砍掉低 z 光谱的噪蓝端。
    """
    total = 0.0
    n_in_band = 0
    details = []

    for rest, name, aw, ew, u in _SCORER_LINES:
        pred = float(rest) * (1.0 + float(z))
        if pred < float(wave[0]) + half or pred > float(wave[-1]) - half:
            continue
        if pred < min_wavelength:
            continue
        mask = (wave > pred - half) & (wave < pred + half)
        w = wave[mask]
        f = flux_norm[mask]
        if len(w) < 5:
            continue

        n_in_band += 1
        snr_local = snr[mask]
        snr_w = float(min(float(np.median(snr_local)) / 20.0, 1.5))

        ai = int(np.argmin(f))
        abs_depth = float(max(0.0, 1.0 - float(f[ai])))
        abs_p = float(np.exp(-0.5 * (float(w[ai] - pred) / 5.0)**2))
        abs_s = u * aw * abs_depth * 10.0 * abs_p * snr_w

        ei = int(np.argmax(f))
        em_height = float(max(0.0, float(f[ei]) - 1.0))
        em_p = float(np.exp(-0.5 * (float(w[ei] - pred) / 5.0)**2))
        em_s = u * ew * em_height * 5.0 * em_p * snr_w

        best = float(max(abs_s, em_s))
        total += best
        if best > 0.01:
            obs_pos = w[ai] if abs_s >= em_s else w[ei]
            morph = 'ABS' if abs_s >= em_s else 'EM'
            details.append((name, morph, best, pred, obs_pos, abs_s, em_s))

    if n_in_band > 0:
        total = total / np.sqrt(n_in_band)
    return total, n_in_band, details


def _score_one_redshift_cwt(z, wave, peaks, troughs, half=20.0, min_wavelength=0.0):
    """Score a single z using CWT-detected features instead of raw flux.

    For each rest-frame line, finds the nearest CWT feature of the correct type
    (peak for emission, trough for absorption) within a ±half Å window around
    the predicted observed wavelength.  A line with no nearby CWT feature gets
    zero contribution — raw flux noise cannot score.

    Score components per matched line:
      - *position_penalty*: Gaussian fall-off with sigma=5 Å
      - *quality*: min(ridge_length/5, 1) × min(snr/10, 1), each capped at 1
      - *weight*: line-specific u × ew (or u × aw) from ``_SCORER_LINES``
      - *amplitude*: |CWT amplitude| of the matched feature

    Parameters
    ----------
    z : float
        Candidate redshift.
    wave : ndarray
        Full wavelength array (used only for range checks).
    peaks : list[dict]
        CWT emission features. Each dict must have keys: wavelength, amplitude,
        ridge_length, snr.
    troughs : list[dict]
        CWT absorption features.  Same key requirements as *peaks*.
    half : float
        Half-width of the search window around predicted wavelength (Å).
    min_wavelength : float
        Skip lines whose predicted wavelength is below this value.

    Returns
    -------
    (score, n_in_band, details)
    """
    total = 0.0
    n_in_band = 0
    details = []

    for rest, name, aw, ew, u in _SCORER_LINES:
        pred = float(rest) * (1.0 + float(z))
        if pred < float(wave[0]) + half or pred > float(wave[-1]) - half:
            continue
        if pred < min_wavelength:
            continue

        n_in_band += 1

        # ── Emission: find nearest peak ──
        best_em = None
        if ew > 0 and peaks:
            best_dist = half + 1.0
            for p in peaks:
                pw = float(p.get('wavelength', 0))
                dist = abs(pw - pred)
                if dist < best_dist and dist <= half:
                    best_dist = dist
                    best_em = p

        # ── Absorption: find nearest trough ──
        best_ab = None
        if aw > 0 and troughs:
            best_dist = half + 1.0
            for t in troughs:
                tw = float(t.get('wavelength', 0))
                dist = abs(tw - pred)
                if dist < best_dist and dist <= half:
                    best_dist = dist
                    best_ab = t

        def _feature_score(feat, weight):
            amp = abs(float(feat.get('amplitude', 0)))
            delta = abs(float(feat.get('wavelength', 0)) - pred)
            pos_p = float(np.exp(-0.5 * (delta / 5.0) ** 2))
            ridge = float(feat.get('ridge_length', 1))
            feat_snr = float(feat.get('snr', 0))
            quality = min(ridge / 5.0, 1.0) * min(feat_snr / 10.0, 1.0)
            return weight * amp * pos_p * quality, delta, amp, ridge, feat_snr

        em_s = 0.0
        ab_s = 0.0
        if best_em is not None:
            em_s, _, _, _, _ = _feature_score(best_em, u * ew)
        if best_ab is not None:
            ab_s, _, _, _, _ = _feature_score(best_ab, u * aw)

        best = max(em_s, ab_s)
        total += best

        if best > 0.01:
            if em_s >= ab_s and best_em is not None:
                _, delta, amp, ridge, feat_snr = _feature_score(best_em, u * ew)
                details.append((name, 'EM', best, pred,
                                float(best_em.get('wavelength', 0)),
                                delta, amp, ridge, feat_snr))
            elif best_ab is not None:
                _, delta, amp, ridge, feat_snr = _feature_score(best_ab, u * aw)
                details.append((name, 'ABS', best, pred,
                                float(best_ab.get('wavelength', 0)),
                                delta, amp, ridge, feat_snr))

    if n_in_band > 0:
        total = total / np.sqrt(n_in_band)
    return total, n_in_band, details


def run_redshift_scoring_v2(wavelength, flux, continuum_flux, snr,
                            brute_force_matches, split_z=1.0, top=5,
                            min_lines=3, half=20.0, blue_cut=4000.0,
                            peak_tol=30.0, scoring_workers=1):
    """v2: 对每组假设的 z_list 逐 z 打分取 max，并按 primary peak 去重。

    与 v1 的关键区别：
    - 不再只对 z_center 打分，而是遍历 z_list 中的每个 z，取最高分作为该组得分。
    - 最优 z 作为该组的代表红移（后续 LLM 的 initial guess）。
    - 打完后按 primary observed wavelength 去重，同峰只留最高分。
    - 并行打分：scoring_workers 控制线程数（0=自动）。

    Parameters
    ----------
    peak_tol : float
        primary observed wavelength 去重容差 (Å)，默认 30。
    scoring_workers : int
        并行线程数，0=自动（CPU 核数）。
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    continuum_flux = np.asarray(continuum_flux, dtype=np.float64)
    snr = np.asarray(snr, dtype=np.float64)

    flux_norm = flux / np.maximum(continuum_flux, 1e-6)

    # 构建 (hypothesis_index, z) 任务列表
    tasks = []
    for i, m in enumerate(brute_force_matches):
        z_list = m.get('z_list', [m.get('z_center', 0)])
        for z in z_list:
            if z > 0:
                tasks.append((i, z))

    if not tasks:
        return {'split_z': split_z, 'top': top, 'low_z': [], 'high_z': [],
                'all_low_z': [], 'all_high_z': []}

    # 并行打分
    workers = scoring_workers if scoring_workers > 0 else os.cpu_count() or 4

    def _score_one_task(args):
        i, z = args
        m = brute_force_matches[i]
        min_wl = blue_cut if z < split_z else 0.0
        s, n_ib, det = _score_one_redshift(z, wavelength, flux_norm, snr,
                                            half=half, min_wavelength=min_wl)
        return (i, z, s, n_ib, det)

    # 按 hypothesis 聚合取 max
    hyp_best = {}  # i -> {best_z, best_score, ...}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, z, s, n_ib, det in ex.map(_score_one_task, tasks):
            if s <= 0:
                continue
            if i not in hyp_best or s > hyp_best[i]['score']:
                hyp_best[i] = {'z': z, 'score': s, 'n_lines': n_ib, 'details': det}

    # 收集结果
    scored = []
    for i, best in hyp_best.items():
        if best['n_lines'] < min_lines:
            continue
        m = brute_force_matches[i]
        scored.append({'z': best['z'], 'score': best['score'],
                       'n_lines': best['n_lines'], 'details': best['details'],
                       'hypothesis': m.get('Hypothesis', ''),
                       'n_em': m.get('N_emission', 0),
                       'n_ab': m.get('N_absorption', 0)})

    # 分组
    low = [r for r in scored if r['z'] < split_z]
    high = [r for r in scored if r['z'] >= split_z]
    low.sort(key=lambda x: -x['score'])
    high.sort(key=lambda x: -x['score'])

    # 按 primary observed wavelength 去重
    low = _dedup_by_primary_peak(low, peak_tol=peak_tol)
    high = _dedup_by_primary_peak(high, peak_tol=peak_tol)

    return {
        'split_z': split_z,
        'top': top,
        'low_z': low[:top],
        'high_z': high[:top],
        'all_low_z': low,
        'all_high_z': high,
    }


def run_redshift_scoring_v3(wavelength, flux, continuum_flux, snr,
                             brute_force_matches, peaks, troughs,
                             split_z=1.0, top=5, min_lines=3, half=20.0,
                             blue_cut=4000.0, peak_tol=30.0, scoring_workers=1):
    """v3: 同 v2 的架构，但使用 ``_score_one_redshift_cwt`` 基于 CWT 特征打分。

    与 v2 的区别：
    - 从 CWT peaks/troughs 表里显式匹配每条线，而非在 raw flux 窗口里取 argmin/argmax。
    - 未被 CWT 检测到的位置不计分（噪声无法得分）。
    - 分数综合了 CWT amplitude、位置匹配、ridge_length 和 snr。

    .. note::
        v3 不再需要 *flux* 和 *continuum_flux* —— 打分完全基于 CWT 特征表。
       保留这两个参数仅为保持调用签名兼容。
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)

    tasks = []
    for i, m in enumerate(brute_force_matches):
        z_list = m.get('z_list', [m.get('z_center', 0)])
        for z in z_list:
            if z > 0:
                tasks.append((i, z))

    if not tasks:
        return {'split_z': split_z, 'top': top, 'low_z': [], 'high_z': [],
                'all_low_z': [], 'all_high_z': []}

    workers = scoring_workers if scoring_workers > 0 else os.cpu_count() or 4

    def _score_one_task(args):
        i, z = args
        m = brute_force_matches[i]
        min_wl = blue_cut if z < split_z else 0.0
        s, n_ib, det = _score_one_redshift_cwt(
            z, wavelength, peaks, troughs, half=half, min_wavelength=min_wl,
        )
        return (i, z, s, n_ib, det)

    hyp_best = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, z, s, n_ib, det in ex.map(_score_one_task, tasks):
            if s <= 0:
                continue
            if i not in hyp_best or s > hyp_best[i]['score']:
                hyp_best[i] = {'z': z, 'score': s, 'n_lines': n_ib, 'details': det}

    scored = []
    for i, best in hyp_best.items():
        if best['n_lines'] < min_lines:
            continue
        m = brute_force_matches[i]
        scored.append({'z': best['z'], 'score': best['score'],
                       'n_lines': best['n_lines'], 'details': best['details'],
                       'hypothesis': m.get('Hypothesis', ''),
                       'n_em': m.get('N_emission', 0),
                       'n_ab': m.get('N_absorption', 0)})

    low = [r for r in scored if r['z'] < split_z]
    high = [r for r in scored if r['z'] >= split_z]
    low.sort(key=lambda x: -x['score'])
    high.sort(key=lambda x: -x['score'])

    low = _dedup_by_primary_peak(low, peak_tol=peak_tol)
    high = _dedup_by_primary_peak(high, peak_tol=peak_tol)

    return {
        'split_z': split_z,
        'top': top,
        'low_z': low[:top],
        'high_z': high[:top],
        'all_low_z': low,
        'all_high_z': high,
    }


def _extract_primary_wavelength(hypothesis):
    """从 hypothesis 字符串中提取第一个观测波长 (Å)。"""
    try:
        return float(hypothesis.split('-')[0].split(',')[0].strip())
    except (ValueError, AttributeError):
        return None


def _dedup_by_primary_peak(candidates, peak_tol=30.0):
    """按 primary observed wavelength 贪婪去重，保留高分候选。"""
    kept = []
    for c in candidates:
        wl = _extract_primary_wavelength(c.get('hypothesis', ''))
        if wl is None:
            kept.append(c)
            continue
        if any(abs(wl - _extract_primary_wavelength(k.get('hypothesis', ''))) < peak_tol
               for k in kept if _extract_primary_wavelength(k.get('hypothesis', '')) is not None):
            continue
        kept.append(c)
    return kept


# ===========================================================
# Redrock Integration — run rrdesi, parse rrdetails.h5,
# and convert to redshift_hypotheses-compatible format
# ===========================================================

def _run_redrock_subprocess(
    input_fits: str,
    output_fits: str,
    details_h5: str,
    *,
    rr_template_dir: str,
    archetype_dir: str = None,
    use_archetypes: bool = True,
    nminima: int = 9,
    nnearest: int = 2,
    omp_num_threads: int = 1,
    per_camera: bool = True,
    no_legendre: bool = False,
    legendre_degree: int = 2,
    legendre_prior: float = 0.1,
) -> None:
    """
    运行 redrock (rrdesi) 拟合红移。

    Parameters
    ----------
    input_fits : str
        输入 FITS 文件路径
    output_fits : str
        redrock 输出 FITS 路径
    details_h5 : str
        rrdetails.h5 输出路径
    """
    import os as _os
    import shlex as _shlex
    import subprocess as _subprocess
    from pathlib import Path as _Path

    _Path(details_h5).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rrdesi",
        "-i", input_fits,
        "-o", output_fits,
        "-d", details_h5,
        "--nminima", str(nminima),
    ]

    if use_archetypes:
        if archetype_dir is None:
            raise ValueError("ARCHETYPE_DIR is not set but USE_ARCHETYPES=true")
        cmd.extend(["--archetypes", archetype_dir])
        cmd.extend(["--archetype-nnearest", str(nnearest)])
        cmd.extend(["--archetype-legendre-percamera", str(per_camera)])  # rrdesi expects "True"/"False"
        if no_legendre:
            cmd.append("--archetypes-no-legendre")
        else:
            cmd.extend(["--archetype-legendre-degree", str(legendre_degree)])
            cmd.extend(["--archetype-legendre-prior", str(legendre_prior)])

    env = _os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_num_threads)
    env["RR_TEMPLATE_DIR"] = rr_template_dir

    logging.info("Running redrock: %s", " ".join(_shlex.quote(x) for x in cmd))

    result = _subprocess.run(cmd, env=env, text=True, capture_output=True)

    if result.returncode != 0:
        logging.error("Redrock failed (rc=%d): %s", result.returncode, result.stderr)
        raise RuntimeError(f"Redrock exited with code {result.returncode}: {result.stderr}")

    logging.info("Redrock finished successfully → %s", output_fits)


def _parse_rrdetails_h5(details_h5: str) -> list:
    """
    解析 rrdetails.h5 文件，返回所有拟合结果。

    Returns
    -------
    list[dict]
        按 chi2 升序排列的拟合结果列表，每个 dict 包含:
        z, zerr, zwarn, chi2, deltachi2, spectype, subtype, npixels, fitmethod, znum
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required to parse rrdetails.h5. Install with: pip install h5py")

    results = []

    with h5py.File(details_h5, 'r') as f:
        targetids = f['targetids'][()]
        for targetid in targetids:
            tid = int(targetid)
            zfit_path = f'zfit/{tid}/zfit'
            if zfit_path not in f:
                continue
            zfit_data = f[zfit_path][()]
            for i in range(len(zfit_data)):
                row = zfit_data[i]
                results.append({
                    'targetid': tid,
                    'z': float(row['z']),
                    'zerr': float(row['zerr']),
                    'zwarn': int(row['zwarn']),
                    'chi2': float(row['chi2']),
                    'deltachi2': float(row['deltachi2']),
                    'spectype': row['spectype'].decode('utf-8') if isinstance(row['spectype'], bytes) else str(row['spectype']),
                    'subtype': row['subtype'].decode('utf-8') if isinstance(row['subtype'], bytes) else str(row['subtype']),
                    'npixels': int(row['npixels']),
                    'fitmethod': row['fitmethod'].decode('utf-8') if isinstance(row['fitmethod'], bytes) else str(row['fitmethod']),
                    'znum': int(row['znum']),
                })

    # 按 chi2 升序排列（最佳拟合在最前）
    results.sort(key=lambda x: x['chi2'])
    return results


def _redrock_to_hypotheses(redrock_results: list) -> dict:
    """
    将 redrock 拟合结果转换为 redshift_hypotheses 兼容的格式。

    适配 collect_redshift_hypotheses() 和下游 harness 的输入约定。

    Parameters
    ----------
    redrock_results : list[dict]
        _parse_rrdetails_h5 的输出

    Returns
    -------
    dict
        与 brute_force_line_matching 相同格式的字典:
        {z, zmedian, hypotheses}
    """
    hypotheses = []

    for i, rr in enumerate(redrock_results):
        z = rr['z']

        # 用 deltachi2 估算得分：deltachi2 越小表示拟合越好
        # score 映射到 [0, 100]，最佳拟合（deltachi2=0）得满分
        score = float(100.0 / (1.0 + rr['deltachi2']))

        hypothesis_str = (
            f"{z:.4f}-{rr['spectype']}/{rr['subtype']}"
            f" (redrock, χ²={rr['chi2']:.2f})"
        )

        hypotheses.append({
            "Hypothesis": hypothesis_str,
            "z_center": z,
            "z_list": [z],
            "z_max": z,
            "z_min": z,
            "z_spread": 0.0,
            "Emission matches": [],
            "Absorption matches": [],
            "N_emission": rr['npixels'],
            "N_absorption": 0,
            "matched_lines": {},
            "z_representative": z,
            "score": score,
            "source": "redrock",
            "scoring_details": [
                ["redrock", rr['spectype'], score,
                 z, z,
                 rr['chi2'], rr['deltachi2'],
                 rr['zwarn']]
            ],
            # 额外 redrock 元数据
            "_redrock": {
                "spectype": rr['spectype'],
                "subtype": rr['subtype'],
                "chi2": rr['chi2'],
                "deltachi2": rr['deltachi2'],
                "zerr": rr['zerr'],
                "zwarn": rr['zwarn'],
                "npixels": rr['npixels'],
                "znum": rr['znum'],
            },
        })

    # 按 score 降序
    hypotheses.sort(key=lambda h: -h['score'])

    all_z = sorted(set(h['z_representative'] for h in hypotheses))
    zmedian = round(float(np.median(all_z)), 4) if all_z else None

    return {
        "z": all_z,
        "zmedian": zmedian,
        "hypotheses": hypotheses,
    }


def run_redrock_pipeline(
    input_fits: str,
    output_dir: str,
    file_name: str,
    *,
    rr_template_dir: str,
    archetype_dir: str = None,
    use_archetypes: bool = True,
    nminima: int = 9,
    nnearest: int = 2,
    omp_num_threads: int = 1,
) -> dict:
    """
    完整的 Redrock 流水线：运行 rrdesi → 解析 rrdetails.h5 → 转换为 BFM 格式。

    Parameters
    ----------
    input_fits : str
        输入 FITS 文件路径
    output_dir : str
        输出目录
    file_name : str
        文件名（不含扩展名），用于生成输出文件

    Returns
    -------
    dict
        redshift_hypotheses 兼容格式的假设列表
    """
    import os as _os

    redrock_dir = _os.path.join(output_dir, f"{file_name}_redrock")
    _os.makedirs(redrock_dir, exist_ok=True)

    output_fits = _os.path.join(redrock_dir, f"{file_name}_redrock.fits")
    details_h5 = _os.path.join(redrock_dir, f"{file_name}_rrdetails.h5")

    _run_redrock_subprocess(
        input_fits=input_fits,
        output_fits=output_fits,
        details_h5=details_h5,
        rr_template_dir=rr_template_dir,
        archetype_dir=archetype_dir,
        use_archetypes=use_archetypes,
        nminima=nminima,
        nnearest=nnearest,
        omp_num_threads=omp_num_threads,
    )

    redrock_results = _parse_rrdetails_h5(details_h5)
    logging.info("Parsed %d redrock fit results from %s", len(redrock_results), details_h5)

    if not redrock_results:
        logging.warning("Redrock returned no valid fits — falling back to empty hypotheses")
        return {"z": [], "zmedian": None, "hypotheses": []}

    return _redrock_to_hypotheses(redrock_results)
