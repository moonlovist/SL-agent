import os
# import cv2  # 已注释（2026-07-06）：PNG 通道已禁用，仅 _get_figsize 中非 FITS 路径使用
import numpy as np
import matplotlib.pyplot as plt
from typing import List
# from scipy.ndimage import gaussian_filter1d
from FORMA.agents.common.state import SpectroState
from FORMA.agents.multi_agents.utils.usage import normalize_line_name


# ===========================================================
# Plotting
# ===========================================================

def _get_figsize(state: SpectroState):
    """
    获取绘图尺寸。
    对于图像输入，根据图像尺寸计算；
    对于 FITS 输入，使用默认尺寸。
    """
    # 检查是否为 FITS 格式输入
    file_path = state.get('file_path', '')
    if file_path.lower().endswith('.fits'):
        # FITS 格式使用默认尺寸
        return (12, 4)
    
    # 图像格式：根据图像尺寸计算（PNG 通道已禁用）
    # if img is None:
    #     return (12, 4)
    # h, w = img.shape[:2]
    # while h > 10 and w > 10:
    #     h //= 10
    #     w //= 10
    # return (w, h)
    return (12, 4)


def plot_spectrum_snr(state: SpectroState):
    """
    绘制双子图：上方为光谱 flux（含不确定性填充），下方为 Effective SNR。
    保存为 {file_name}_spectrum.png，并返回 fig。
    """
    wavelength = state['spectrum']['wavelength']
    flux = state['spectrum']['flux']
    flux_top = state['spectrum'].get('max_unresolved_flux')
    flux_bottom = state['spectrum'].get('min_unresolved_flux')
    effective_snr = state['spectrum'].get('snr')
    ivar = state['spectrum'].get('ivar')

    h, w = _get_figsize(state)

    fig, axs = plt.subplots(2, 1, figsize=(h, 2*w))

    # 上方：光谱 flux
    if flux_top is not None and flux_bottom is not None:
        axs[0].fill_between(wavelength, flux_bottom, flux_top, alpha=0.4, color='gray', 
                           label='information lossed in Opencv processing')
    axs[0].plot(wavelength, flux, color='b', label=r'$\bar F$: signal extracted from picture')
    axs[0].set_ylabel('flux')
    axs[0].set_xlabel('wavelength')
    axs[0].legend()

    # 下方：SNR
    if effective_snr is not None:
        # 确保 effective_snr 是 numpy 数组
        effective_snr = np.asarray(effective_snr, dtype=np.float64)
        axs[1].plot(wavelength, effective_snr, c='orange', label=r'$SNR=\frac{{\bar F}_i}{\sigma_{i,j}}$')
    elif ivar is not None:
        # 从 ivar 计算 SNR
        ivar = np.asarray(ivar, dtype=np.float64)
        flux_arr = np.asarray(flux, dtype=np.float64)
        ivar_safe = np.maximum(ivar, 1e-10)
        snr_from_ivar = flux_arr * np.sqrt(ivar_safe)
        axs[1].plot(wavelength, snr_from_ivar, c='orange', label='SNR from IVAR')
    else:
        axs[1].text(0.5, 0.5, 'SNR data not available', ha='center', va='center', 
                    transform=axs[1].transAxes, fontsize=14)
    axs[1].set_ylabel('Effective SNR')
    axs[1].set_xlabel('wavelength')
    axs[1].legend(fontsize=15)

    fig.savefig(os.path.join(state['output_dir'], f"{state['file_name']}_spectrum.png"), bbox_inches='tight')
    plt.close(fig)


def plot_spec_extract(state: SpectroState):
    """
    绘制单子图：光谱 flux 曲线 + 信息损失区域填充（来自 OpenCV 处理）。
    保存为 {file_name}_spec_extract.png。
    """
    wavelength = state['spectrum']['wavelength']
    flux = state['spectrum']['flux']
    flux_top = state['spectrum'].get('max_unresolved_flux', [])
    flux_bottom = state['spectrum'].get('min_unresolved_flux', [])

    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    if flux_top and flux_bottom:
        plt.fill_between(wavelength, flux_bottom, flux_top,
                         color='#FFB6A6', alpha=0.5, linewidth=0,
                         label='information lost in OpenCV processing (pink #FFB6A6)')
    plt.plot(wavelength, flux, color='b', lw=1.5,
             label=r'$\bar{F}_i$: signal extracted from picture (blue)')
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.legend(fontsize=12)
    plt.savefig(
        os.path.join(state['output_dir'], f"{state['file_name']}_spec_extract.png"),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()


def plot_continuum(state: SpectroState):
    """
    绘制 continuum 曲线，保存到 state['continuum_path']。
    """
    continuum_wavelength = np.array(state['continuum']['wavelength'])
    continuum_flux = np.array(state['continuum']['flux'])

    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    plt.plot(continuum_wavelength, continuum_flux, color='orange', label='Continuum')
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.legend(fontsize=12)
    plt.savefig(
        state['continuum_path'],
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

def plot_residual_spectrum(state: SpectroState):
    """
    绘制 residual_spectrum 曲线。
    """
    residual_spectrum_wavelength = np.array(state['residual_spectrum']['wavelength'])
    residual_spectrum_flux = np.array(state['residual_spectrum']['flux'])

    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    plt.plot(residual_spectrum_wavelength, residual_spectrum_flux, color='orange', label='residual_spectrum')
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.legend(fontsize=12)
    path = os.path.join(state['output_dir'], "visual_interpreter", f"{state['file_name']}_residual_spectrum.png")
    plt.savefig(
        path,
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()


def plot_masked_spectrum(state: SpectroState):
    """
    绘制原始光谱与 masked 后的光谱对比图。
    保存为 {file_name}_masked_spectrum.png。
    """
    spec = state.get("spectrum", {})
    wavelengths = spec.get("wavelength", [])
    flux = spec.get("flux", [])

    h, w = _get_figsize(state)

    fig = plt.figure(figsize=(h, w))
    plt.plot(wavelengths, flux, label="Original Spectrum", alpha=0.5)
    plt.plot(
        state["cleaned_spectrum"]["wavelength"],
        state["cleaned_spectrum"]["flux"],
        label="Masked Spectrum",
        alpha=0.9
    )
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.legend(fontsize=12)
    path = os.path.join(
        state["output_dir"],
        f"{state['file_name']}_masked_spectrum.png"
    )
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_features(state: SpectroState, wavelength_label: bool = True):
    """
    绘制三行子图：光谱与连续谱、残差与吸收线、残差与发射线。
    保存为 {file_name}_features.png。
    
    Parameters
    ----------
    state : SpectroState
        状态字典，需包含 'spectrum', 'continuum', 'peaks', 'troughs'
    wavelength_label : bool
        是否标注特征的中心波长
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from FORMA.agents.multi_agents.utils.VI import SIGMA_TO_FWHM
    
    # 获取数据
    spec = state["spectrum"]
    wavelengths = np.array(spec["wavelength"])
    flux = np.array(spec["flux"])
    
    # 获取连续谱
    continuum = state.get('continuum', {})
    continuum_wavelength = np.array(continuum.get('wavelength', []))
    continuum_flux = np.array(continuum.get('flux', []))
    
    # 计算残差（光谱 - 连续谱）
    if len(continuum_wavelength) > 0:
        # 确保 continuum 和 spectrum 的波长对齐
        residual = flux - continuum_flux
    else:
        residual = flux.copy()
    
    # 创建三行子图，共享x轴
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # ===========================================================================
    # 子图1：原始光谱 + 连续谱
    # ===========================================================================
    ax1 = axes[0]
    ax1.plot(wavelengths, flux, 'k-', lw=0.8, alpha=0.75, label='Spectrum')
    if len(continuum_wavelength) > 0:
        ax1.plot(continuum_wavelength, continuum_flux, 'r--', lw=1.5, alpha=0.9, label='Continuum')
    ax1.set_ylabel('flux')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.25)
    
    # ===========================================================================
    # 子图2：残差 + 吸收线（troughs）
    # ===========================================================================
    ax2 = axes[1]
    ax2.plot(wavelengths, residual, color='steelblue', lw=0.7, alpha=0.75, label='Residual')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 获取 y 轴范围用于文本定位
    residual_y_min, residual_y_max = np.min(residual), np.max(residual)
    text_y_position = residual_y_max * 0.92
    
    # 绘制吸收线（troughs，用番茄红色）
    troughs = state.get('troughs', [])
    n_troughs = 0
    for trough in troughs:
        center = trough.get('wavelength')
        if center is None or center <= 0:
            continue
        n_troughs += 1

        fwhm = trough.get('FWHM_A', trough.get('FWHM(Å)', None))
        if fwhm is None:
            sigma_val = trough.get('sigma')
            fwhm = sigma_val * SIGMA_TO_FWHM if sigma_val else 10.0
        amplitude = trough.get('amplitude', trough.get('深度', 0))

        # 绘制倒高斯拟合曲线（吸收线 amplitude 为负值，取绝对值判断）
        if abs(amplitude) > 0 and fwhm > 0:
            sigma = fwhm / SIGMA_TO_FWHM
            wave_local = np.linspace(center - 3.5*sigma, center + 3.5*sigma, 200)
            gaussian_profile = -abs(amplitude) * np.exp(-0.5 * ((wave_local - center) / sigma) ** 2)
            ax2.plot(wave_local, gaussian_profile, color='tomato', linewidth=1.8, alpha=0.85)
        
        # 绘制特征位置垂直线
        ax2.axvline(center, color='tomato', linestyle='--', linewidth=0.8, alpha=0.4)
        
        # 标注中心波长
        if wavelength_label:
            ax2.text(center, text_y_position, f'{center:.1f}',
                     rotation=90, verticalalignment='top', horizontalalignment='center',
                     fontsize=7, color='tomato', alpha=0.8)
    
    ax2.set_ylabel('flux')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.25)
    
    # ===========================================================================
    # 子图3：残差 + 发射线（peaks）
    # ===========================================================================
    ax3 = axes[2]
    ax3.plot(wavelengths, residual, color='steelblue', lw=0.7, alpha=0.75, label='Residual')
    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 绘制发射线（peaks，用深橙色）
    peaks = state.get('peaks', [])
    n_peaks = 0
    for peak in peaks:
        center = peak.get('wavelength')
        if center is None:
            continue
        n_peaks += 1
        
        fwhm = peak.get('FWHM_A', peak.get('FWHM(Å)', None))
        if fwhm is None:
            sigma_val = peak.get('sigma')
            fwhm = sigma_val * SIGMA_TO_FWHM if sigma_val else 10.0
        amplitude = peak.get('amplitude', peak.get('幅度', 0))

        # 绘制高斯拟合曲线
        if abs(amplitude) > 0 and fwhm > 0:
            sigma = fwhm / SIGMA_TO_FWHM
            wave_local = np.linspace(center - 3.5*sigma, center + 3.5*sigma, 200)
            gaussian_profile = amplitude * np.exp(-0.5 * ((wave_local - center) / sigma) ** 2)
            ax3.plot(wave_local, gaussian_profile, color='darkorange', linewidth=1.8, alpha=0.85)
        
        # 绘制特征位置垂直线
        ax3.axvline(center, color='darkorange', linestyle='--', linewidth=0.8, alpha=0.4)
        
        # 标注中心波长
        if wavelength_label:
            ax3.text(center, text_y_position, f'{center:.1f}',
                     rotation=90, verticalalignment='top', horizontalalignment='center',
                     fontsize=7, color='darkorange', alpha=0.8)
    
    ax3.set_xlabel('wavelength')
    ax3.set_ylabel('flux')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.25)
    
    plt.tight_layout()
    
    n_peaks_valid = len([p for p in peaks if p.get('wavelength') is not None])
    n_troughs_valid = len([t for t in troughs if t.get('wavelength') is not None and t.get('wavelength') > 0])
    print(f"Plot {n_peaks_valid} peaks, {n_troughs_valid} troughs.")

    fig.savefig(os.path.join(state['output_dir'], "visual_interpreter", f"{state['file_name']}_features.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_harness_candidate(
    wavelength,
    flux,
    continuum_flux,
    lines_csv_path: str,
    output_path: str,
    redshift: float = None,
    title: str = None,
):
    """
    为单个 harness candidate 绘制采纳的特征线（三行子图）。
    与 plot_features 画法一致。

    Parameters
    ----------
    wavelength, flux, continuum_flux : array-like
    lines_csv_path : str
         harness 输出的 _lines.csv 路径
    output_path : str
         输出 PNG 路径
    redshift : float, optional
    title : str, optional
    """
    import csv
    import os as _os
    from FORMA.agents.multi_agents.utils.VI import SIGMA_TO_FWHM

    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    continuum_flux = np.asarray(continuum_flux, dtype=np.float64)
    residual = flux - continuum_flux

    # 读取 lines CSV，筛选被采纳的行
    adopted_lines = []
    if _os.path.exists(lines_csv_path):
        with open(lines_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = (row.get("status") or "").strip().upper()
                if status in ("LIKELY", "MARGINAL", "ESTIMATED"):
                    adopted_lines.append(row)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # ── 子图1：光谱 + 连续谱 ──
    ax1 = axes[0]
    ax1.plot(wavelength, flux, 'k-', lw=0.8, alpha=0.75, label='Spectrum')
    ax1.plot(wavelength, continuum_flux, 'r--', lw=1.5, alpha=0.9, label='Continuum')
    ax1.set_ylabel('flux')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # ── 子图2：残差 + 吸收线 ──
    ax2 = axes[1]
    ax2.plot(wavelength, residual, color='steelblue', lw=0.7, alpha=0.75, label='Residual')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    residual_y_min, residual_y_max = np.min(residual), np.max(residual)
    text_y_position = residual_y_max * 0.92

    # ── 子图3：残差 + 发射线 ──
    ax3 = axes[2]
    ax3.plot(wavelength, residual, color='steelblue', lw=0.7, alpha=0.75, label='Residual')
    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    n_abs = 0
    n_em = 0

    from FORMA.agents.multi_agents.harness.tools import EMISSION_LINES, ABSORPTION_LINES

    for row in adopted_lines:
        name = row.get("name", "")
        # Normalize line name to canonical form (handles LLM-generated variants)
        canonical_name = normalize_line_name(name)
        is_emission = canonical_name in EMISSION_LINES
        is_absorption = canonical_name in ABSORPTION_LINES

        center = None
        for key in ("fitted_center", "predicted_obs"):
            val = row.get(key)
            if val and val.strip():
                try:
                    center = float(val)
                    if center > 0:
                        break
                except (ValueError, TypeError):
                    continue

        if center is None:
            continue

        # 尝试获取振幅和宽度
        amp = None
        for key in ("amplitude",):
            val = row.get(key)
            if val and val.strip():
                try:
                    amp = float(val)
                    break
                except (ValueError, TypeError):
                    continue

        fwhm = None
        for key in ("fitted_sigma",):
            val = row.get(key)
            if val and val.strip():
                try:
                    sigma_val = float(val)
                    fwhm = sigma_val * SIGMA_TO_FWHM
                    break
                except (ValueError, TypeError):
                    continue

        if fwhm is None:
            fwhm = 10.0

        if is_emission:
            n_em += 1
            color = 'darkorange'
            ax = ax3
            # 发射线振幅应为正
            if amp is not None and amp > 0 and fwhm > 0:
                sigma = fwhm / SIGMA_TO_FWHM
                wave_local = np.linspace(center - 3.5 * sigma, center + 3.5 * sigma, 200)
                gaussian_profile = amp * np.exp(-0.5 * ((wave_local - center) / sigma) ** 2)
                ax.plot(wave_local, gaussian_profile, color=color, linewidth=1.8, alpha=0.85)
        elif is_absorption:
            n_abs += 1
            color = 'tomato'
            ax = ax2
            # 吸收线振幅为负（CWT 已保证），绘图用 -|amp|
            if fwhm > 0:
                sigma = fwhm / SIGMA_TO_FWHM
                wave_local = np.linspace(center - 3.5 * sigma, center + 3.5 * sigma, 200)
                amp_val = abs(amp) if amp else 0.0
                gaussian_profile = -amp_val * np.exp(-0.5 * ((wave_local - center) / sigma) ** 2)
                ax.plot(wave_local, gaussian_profile, color='tomato', linewidth=1.8, alpha=0.85)
        else:
            # 无法判断类型，跳过
            continue

        ax.axvline(center, color=color, linestyle='--', linewidth=0.8, alpha=0.4)
        ax.text(center, text_y_position, f'{name}\n{center:.1f}',
                rotation=90, verticalalignment='top', horizontalalignment='center',
                fontsize=6, color=color, alpha=0.8)

    ax2.set_ylabel('flux')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.25)

    ax3.set_xlabel('wavelength')
    ax3.set_ylabel('flux')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.25)

    # 标题
    title_str = title or ""
    if redshift is not None:
        title_str = f"z = {redshift:.4f}  {title_str}"
    if title_str:
        fig.suptitle(title_str.strip(), fontsize=13, y=0.98)

    plt.tight_layout()
    _os.makedirs(_os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot harness candidate: {n_em} emission, {n_abs} absorption → {output_path}")