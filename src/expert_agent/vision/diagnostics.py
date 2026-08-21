from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import numpy as np

from ..core.config import VisionConfig
from .astrometry import orientation_vectors, pixel_to_sky
from .detectors import GeometricPathCandidate, ResidualStructure, StabilityDetection, threshold_masks
from .foreground_models import ForegroundModelComparison
from .preprocess import PreprocessedImage


def _matplotlib(output: Path):
    cache_dir = output / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _view(prep: PreprocessedImage) -> tuple[slice, slice, int]:
    y, x = prep.nucleus_yx
    half = min(60, y, x, prep.image.data.shape[0] - y, prep.image.data.shape[1] - x)
    return slice(y - half, y + half), slice(x - half, x + half), half


def _mark(axis, half: int) -> None:
    axis.plot(half, half, "r+", ms=10, mew=1.6)
    axis.set_xticks([])
    axis.set_yticks([])


def _display_limits(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [3.0, 99.5])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _origin(prep: PreprocessedImage) -> str:
    return "lower" if prep.image.input_format == "fits" else "upper"


def normalize_euclid_display(array: np.ndarray) -> np.ndarray:
    """Normalize flux for human display without changing scientific arrays."""
    values = np.asarray(array, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = np.percentile(finite, [1.0, 99.85])
    if high <= low:
        return np.zeros_like(values)
    normalized = (np.nan_to_num(values, nan=low) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def arcsinh_stretch(normalized: np.ndarray, q: float = 500.0) -> np.ndarray:
    values = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
    return np.arcsinh(q * values) / np.arcsinh(q)


def midtone_transfer(normalized: np.ndarray, m: float) -> np.ndarray:
    values = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
    m = float(np.clip(m, 1e-4, 1.0 - 1e-4))
    denominator = (2.0 * m - 1.0) * values - m
    return np.divide(
        (m - 1.0) * values,
        denominator,
        out=np.zeros_like(values),
        where=np.abs(denominator) > 1e-12,
    )


def solve_midtone_parameter(
    normalized: np.ndarray, target_mean: float = 0.2
) -> float:
    height, width = normalized.shape
    half_y, half_x = min(50, height // 2), min(50, width // 2)
    cy, cx = height // 2, width // 2
    central = normalized[cy - half_y : cy + half_y, cx - half_x : cx + half_x]
    low, high = 1e-4, 1.0 - 1e-4
    for _ in range(48):
        middle = (low + high) / 2.0
        mean = float(np.mean(midtone_transfer(central, middle)))
        if mean > target_mean:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def write_expert_views(prep: PreprocessedImage, output: Path) -> Path:
    """Write clean Euclid-style VIS views intended for human/agent inspection."""
    plt = _matplotlib(output)
    normalized = normalize_euclid_display(prep.image.data)
    arcsinh = arcsinh_stretch(normalized, q=500.0)
    mtf_parameter = solve_midtone_parameter(normalized, target_mean=0.2)
    mtf = midtone_transfer(normalized, mtf_parameter)
    figure, axes = plt.subplots(2, 2, figsize=(9, 9))
    origin = _origin(prep)
    cy, cx = prep.nucleus_yx
    half = min(50, cy, cx, arcsinh.shape[0] - cy, arcsinh.shape[1] - cx)
    crop = np.s_[cy - half : cy + half, cx - half : cx + half]
    panels = (
        (arcsinh, "VIS arcsinh全视场（Q=500）"),
        (mtf, f"VIS MTF全视场（m={mtf_parameter:.4f}）"),
        (arcsinh[crop], "VIS arcsinh中央100×100像素"),
        (mtf[crop], "VIS MTF中央100×100像素"),
    )
    for axis, (values, title) in zip(axes.ravel(), panels):
        axis.imshow(values, origin=origin, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Euclid风格专家观察图（仅改变显示，不改变测量数据）\n"
        "99.85%高值裁剪；无搜索圆、裁剪框或结构标签"
    )
    figure.tight_layout()
    path = output / "vision_expert_views.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def write_foreground_model_comparison(
    prep: PreprocessedImage,
    comparison: ForegroundModelComparison,
    output: Path,
) -> Path | None:
    if not comparison.success or comparison.model is None or comparison.residual is None:
        return None
    plt = _matplotlib(output)
    ys, xs, _ = _view(prep)
    origin = _origin(prep)
    data_view = prep.image.data[ys, xs]
    primary_model = prep.model[ys, xs]
    primary_residual = prep.residual[ys, xs]
    comparison_model = comparison.model[ys, xs]
    comparison_residual = comparison.residual[ys, xs]
    model_difference = primary_model - comparison_model
    data_low, data_high = np.percentile(data_view[np.isfinite(data_view)], [1.0, 99.5])
    residual_values = np.concatenate(
        [primary_residual.ravel(), comparison_residual.ravel(), model_difference.ravel()]
    )
    residual_values = residual_values[np.isfinite(residual_values)]
    residual_limit = float(np.percentile(np.abs(residual_values), 99.0))
    residual_limit = max(residual_limit, 1e-12)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = (
        (data_view, "1. 原始VIS中央切图", "gray", data_low, data_high),
        (primary_model, f"2. 当前模型\n{prep.model_method}", "gray", data_low, data_high),
        (primary_residual, "3. 当前模型擦除后", "coolwarm", -residual_limit, residual_limit),
        (comparison_model, "4. PSF-MGE比较模型", "gray", data_low, data_high),
        (comparison_residual, "5. PSF-MGE擦除后", "coolwarm", -residual_limit, residual_limit),
        (model_difference, "6. 两块‘橡皮擦’的差异", "coolwarm", -residual_limit, residual_limit),
    )
    for axis, (values, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        axis.imshow(values, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "前景星系擦除方法对比（PSF-MGE当前仅用于审计，不参与结构检测）\n"
        "红/蓝表示正/负残差；两种结果都保留的结构更不依赖单一模型"
    )
    figure.tight_layout()
    path = output / "vision_foreground_model_comparison.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def write_threshold_process(
    prep: PreprocessedImage,
    config: VisionConfig,
    detections: list[StabilityDetection],
    level_map: np.ndarray,
    output: Path,
) -> Path:
    """Render all real thresholds, masked residuals, shells, and survival map."""
    plt = _matplotlib(output)
    masks, _ = threshold_masks(prep, config)
    ys, xs, half = _view(prep)
    thresholds = list(config.snr_thresholds)
    columns = len(thresholds)
    figure, axes = plt.subplots(3, columns, figsize=(3.0 * columns, 9.0))
    if columns == 1:
        axes = np.asarray(axes).reshape(3, 1)
    residual_view = prep.residual[ys, xs]
    origin = _origin(prep)
    vmin, vmax = _display_limits(residual_view)

    for index, (threshold, mask) in enumerate(zip(thresholds, masks)):
        axes[0, index].imshow(mask[ys, xs], origin=origin, cmap="gray")
        axes[0, index].set_title(f"mask ≥ {threshold:.2f}σ")
        _mark(axes[0, index], half)

        axes[1, index].imshow(
            np.where(mask, prep.residual, np.nan)[ys, xs],
            origin=origin,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, index].set_title(f"残差 × mask ≥ {threshold:.2f}σ")
        _mark(axes[1, index], half)

        if index < columns - 1:
            shell = mask & ~masks[index + 1]
            axes[2, index].imshow(shell[ys, xs], origin=origin, cmap="gray")
            axes[2, index].set_title(
                f"层壳 {threshold:.2f}–{thresholds[index + 1]:.2f}σ"
            )
        else:
            axes[2, index].imshow(
                level_map[ys, xs],
                origin=origin,
                cmap="turbo",
                vmin=0,
                vmax=max(columns, 1),
            )
            axes[2, index].set_title("最高存活层级图")
        _mark(axes[2, index], half)

    for detection in detections:
        local_x = detection.x - (prep.nucleus_yx[1] - half)
        local_y = detection.y - (prep.nucleus_yx[0] - half)
        if 0 <= local_x < 2 * half and 0 <= local_y < 2 * half:
            axes[2, -1].plot(local_x, local_y, "wx", ms=8, mew=1.3)
            axes[2, -1].annotate(
                f"{detection.survival}/{detection.total_runs}",
                (local_x + 2, local_y - 2),
                color="white",
                fontsize=7,
            )

    axes[0, 0].set_ylabel("1. 阈值掩膜")
    axes[1, 0].set_ylabel("2. 掩膜后的残差")
    axes[2, 0].set_ylabel("3. 相邻层壳 / 层级图")
    figure.suptitle(
        f"S/N 多阈值完整过程：{prep.image.object_id} [{prep.image.band}]\n"
        "输入为前景星系减除后的原始残差 S/N；红十字为中心",
        fontsize=13,
    )
    figure.tight_layout()
    path = output / "vision_snr_threshold_process.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_sep_consensus(
    prep: PreprocessedImage,
    config: VisionConfig,
    detections: list[StabilityDetection],
    output: Path,
) -> Path:
    plt = _matplotlib(output)

    ys, xs, half = _view(prep)
    figure, axis = plt.subplots(figsize=(8, 8))
    residual_view = prep.residual[ys, xs]
    vmin, vmax = _display_limits(residual_view)
    axis.imshow(residual_view, origin=_origin(prep), cmap="gray", vmin=vmin, vmax=vmax)
    for detection in detections:
        if detection.pixels_y is None or detection.pixels_x is None:
            continue
        local_x = detection.x - (prep.nucleus_yx[1] - half)
        local_y = detection.y - (prep.nucleus_yx[0] - half)
        fraction = detection.survival / max(detection.total_runs, 1)
        color = "lime" if detection.survival >= config.sep_min_survival else "orange"
        pixel_x = detection.pixels_x - (prep.nucleus_yx[1] - half)
        pixel_y = detection.pixels_y - (prep.nucleus_yx[0] - half)
        visible = (
            (pixel_x >= 0) & (pixel_x < 2 * half)
            & (pixel_y >= 0) & (pixel_y < 2 * half)
        )
        axis.scatter(
            pixel_x[visible], pixel_y[visible], s=5 + 7 * fraction,
            facecolors="none", edgecolors=color, linewidths=0.5,
        )
        axis.annotate(
            f"{detection.survival}/{detection.total_runs}\n"
            f"切向偏差={detection.tangential_misalignment_deg:.1f}°",
            (local_x + 4, local_y - 4), color="yellow", fontsize=8,
        )
    _mark(axis, half)
    axis.set_title(
        f"SEP 多参数共识：{len(config.sep_nthresh)}×{len(config.sep_deblend_cont)} "
        f"= {len(config.sep_nthresh) * len(config.sep_deblend_cont)} 次运行\n"
        "逐像素显示多次运行的并集；绿色=达到存活门槛，橙色=低稳定原始残差"
    )
    figure.tight_layout()
    path = output / "vision_sep_consensus.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_model_chain(
    prep: PreprocessedImage,
    config: VisionConfig,
    footprint: np.ndarray,
    residual_structure_mask: np.ndarray,
    sep_only_candidate_mask: np.ndarray,
    sep_geometry_mask: np.ndarray,
    geometry_recovered_mask: np.ndarray,
    output: Path,
) -> Path:
    plt = _matplotlib(output)
    ys, xs, half = _view(prep)
    residual_view = prep.residual[ys, xs]
    vmin, vmax = _display_limits(residual_view)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    origin = _origin(prep)
    axes[0].imshow(residual_view, origin=origin, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("1. 前景减除残差")
    axes[1].imshow(residual_view, origin=origin, cmap="gray", vmin=vmin, vmax=vmax)
    raw_y, raw_x = np.where(footprint[ys, xs])
    axes[1].scatter(raw_x, raw_y, s=5, facecolors="none", edgecolors="orange", linewidths=0.5)
    axes[1].set_title("2. SEP 原始多参数像素并集（仅留痕）")
    axes[2].imshow(residual_view, origin=origin, cmap="gray", vmin=vmin, vmax=vmax)
    supported_y, supported_x = np.where(sep_geometry_mask[ys, xs])
    recovered_y, recovered_x = np.where(geometry_recovered_mask[ys, xs])
    axes[2].scatter(supported_x, supported_y, s=7, c="lime", marker="s")
    axes[2].scatter(recovered_x, recovered_y, s=7, c="magenta", marker="s")
    axes[2].set_title("3. 绿=SEP+路线一致；紫=路线内新恢复")
    axes[3].imshow(residual_view, origin=origin, cmap="gray", vmin=vmin, vmax=vmax)
    sep_only_y, sep_only_x = np.where(sep_only_candidate_mask[ys, xs])
    axes[3].scatter(
        sep_only_x, sep_only_y, s=5, facecolors="none",
        edgecolors="orange", linewidths=0.5,
    )
    axes[3].scatter(supported_x, supported_y, s=7, c="lime", marker="s")
    axes[3].scatter(recovered_x, recovered_y, s=7, c="magenta", marker="s")
    axes[3].set_title("4. 路线内结构整合（颜色不是最终置信度）")
    for axis in axes:
        _mark(axis, half)
    figure.suptitle(
        "SEP局部分割与路线内结构整合\n"
        "橙=路线外SEP降级追溯；绿=路线内原SEP；紫=路线内补充恢复；绿色与紫色都需重点复核"
    )
    figure.tight_layout()
    path = output / "vision_model_chain.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_geometric_path_search(
    prep: PreprocessedImage,
    geometric_paths: list[GeometricPathCandidate],
    output: Path,
) -> Path:
    """Show proposed search paths and only the pixels newly recovered on them."""
    plt = _matplotlib(output)
    from matplotlib.patches import Ellipse as PlotEllipse

    ys, xs, half = _view(prep)
    residual_view = prep.residual[ys, xs]
    snr_view = prep.residual_snr[ys, xs]
    vmin, vmax = _display_limits(residual_view)
    snr_limit = max(2.0, float(np.percentile(np.abs(snr_view[np.isfinite(snr_view)]), 99.0)))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    origin = _origin(prep)
    axes[0].imshow(residual_view, origin=origin, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].imshow(snr_view, origin=origin, cmap="coolwarm", vmin=-snr_limit, vmax=snr_limit)
    for index, candidate in enumerate(geometric_paths, start=1):
        local_x = candidate.center_x - (prep.nucleus_yx[1] - half)
        local_y = candidate.center_y - (prep.nucleus_yx[0] - half)
        color = "deepskyblue"
        for axis in axes:
            axis.add_patch(
                PlotEllipse(
                    (local_x, local_y),
                    2.0 * candidate.semi_major_radius_px,
                    2.0 * candidate.semi_major_radius_px * candidate.axis_ratio,
                    angle=candidate.position_angle_deg,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5,
                )
            )
            axis.annotate(
                f"G{index}: {candidate.mode}, candidates={len(candidate.recovered_segments)}",
                (local_x + 2, local_y + 2), color=color, fontsize=7,
            )
            if np.any(candidate.recovered_mask):
                recovered_y, recovered_x = np.where(candidate.recovered_mask[ys, xs])
                axis.scatter(recovered_x, recovered_y, s=7, c="magenta", marker="s")
    for axis in axes:
        _mark(axis, half)
    axes[0].set_title("前景减除残差")
    axes[1].set_title("残差信噪比（S/N）")
    figure.suptitle(
        "SEP之后的几何模型引导搜索\n"
        "蓝线=允许再次寻找的路线；紫色=路线内SEP检出的待审计候选；路线本身不是环"
    )
    figure.tight_layout()
    path = output / "vision_geometric_path_search.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def write_field_context(
    prep: PreprocessedImage,
    config: VisionConfig,
    segments: list[ResidualStructure],
    output: Path,
    candidate_points: list[tuple[str, float, float, str]] | None = None,
) -> Path:
    """Map the analyzed crop and every measured structure onto the full field."""
    plt = _matplotlib(output)
    from matplotlib.patches import Circle, Rectangle

    data = prep.image.data
    finite = data[np.isfinite(data)]
    low, high = np.percentile(finite, [1.0, 99.5]) if finite.size else (0.0, 1.0)
    figure, axis = plt.subplots(figsize=(8.5, 8.5))
    axis.imshow(
        data,
        origin=_origin(prep),
        cmap="gray",
        vmin=float(low),
        vmax=float(high if high > low else low + 1.0),
    )
    center_x, center_y = prep.nucleus_yx[1], prep.nucleus_yx[0]
    half = min(
        60, center_y, center_x,
        data.shape[0] - center_y, data.shape[1] - center_x,
    )
    axis.add_patch(
        Rectangle(
            (center_x - half, center_y - half),
            2 * half, 2 * half,
            fill=False, edgecolor="cyan", linewidth=1.2,
            label="diagnostic crop",
        )
    )
    for radius, linestyle, label in (
        (config.ring_inner_px, "--", "search inner radius"),
        (config.ring_outer_px, "-", "search outer radius"),
    ):
        axis.add_patch(
            Circle(
                (center_x, center_y), radius, fill=False,
                edgecolor="yellow", linestyle=linestyle, linewidth=1.0,
                label=label,
            )
        )
    axis.plot(center_x, center_y, "r+", ms=11, mew=1.8)
    for index, segment in enumerate(segments, start=1):
        axis.plot(segment.x, segment.y, "mx", ms=8, mew=1.4)
        axis.annotate(
            f"S{index}", (segment.x + 3, segment.y + 3),
            color="magenta", fontsize=9, weight="bold",
        )
    candidate_colors = {
        "sep_only": "orange",
        "sep_geometry_consistent": "lime",
        "geometry_recovered": "magenta",
    }
    for candidate_id, candidate_x, candidate_y, status in candidate_points or []:
        color = candidate_colors.get(status, "orange")
        axis.plot(candidate_x, candidate_y, "s", color=color, ms=5, mew=1.0)
        axis.annotate(
            candidate_id, (candidate_x + 3, candidate_y + 3),
            color=color, fontsize=9, weight="bold",
        )
    north, east = orientation_vectors(
        prep.image, (float(center_x), float(center_y))
    )
    if north is not None and east is not None:
        base_x, base_y = 35.0, 35.0
        for vector, label, color in (
            (north, "N", "lime"), (east, "E", "cyan")
        ):
            axis.annotate(
                "", xy=(base_x + 24 * vector[0], base_y + 24 * vector[1]),
                xytext=(base_x, base_y),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
            )
            axis.text(
                base_x + 29 * vector[0], base_y + 29 * vector[1], label,
                color=color, fontsize=10, weight="bold",
            )
    center_ra, center_dec = pixel_to_sky(
        prep.image, float(center_x), float(center_y)
    )
    sky_text = (
        f"center RA={center_ra:.6f}°, Dec={center_dec:.6f}°"
        if center_ra is not None and center_dec is not None
        else "center sky coordinate unavailable"
    )
    scale_text = (
        f"scale={prep.image.pixel_scale_arcsec:.4f} arcsec/pixel"
        if prep.image.pixel_scale_arcsec is not None
        else "pixel scale unavailable"
    )
    axis.set_title(
        f"Full-field location audit: {prep.image.object_id} [{prep.image.band}]\n"
        f"center pixel=({center_x}, {center_y}); {sky_text}; {scale_text}"
    )
    axis.set_xlabel("FITS pixel x")
    axis.set_ylabel("FITS pixel y")
    axis.legend(loc="upper left", fontsize=7)
    figure.tight_layout()
    path = output / "vision_field_context.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def write_rgb_context(
    prep: PreprocessedImage,
    config: VisionConfig,
    segments: list[ResidualStructure],
    output: Path,
    candidate_points: list[tuple[str, float, float, str]] | None = None,
) -> Path | None:
    """Overlay the analysis region on a sibling RGB preview after orientation matching."""
    rgb_path = prep.image.path.parent / "rgb_0.png"
    if not rgb_path.exists():
        return None
    try:
        from PIL import Image
        from scipy import ndimage
    except ImportError:
        return None
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    if rgb.shape[:2] != prep.image.data.shape:
        return None
    gray = np.asarray(rgb, dtype=float).mean(axis=2)
    science = ndimage.gaussian_filter(prep.image.data, 1.5)

    def normalized(values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return np.zeros_like(values, dtype=float)
        low, high = np.percentile(finite, [1.0, 99.0])
        clipped = np.clip(values, low, high)
        std = float(np.nanstd(clipped))
        return (clipped - float(np.nanmean(clipped))) / max(std, 1e-9)

    target = normalized(science).ravel()
    candidates = {
        "identity": gray,
        "flip_y": gray[::-1, :],
        "flip_x": gray[:, ::-1],
        "rotate_180": gray[::-1, ::-1],
    }
    scored = []
    for name, candidate in candidates.items():
        score = float(
            np.corrcoef(
                target,
                normalized(ndimage.gaussian_filter(candidate, 1.5)).ravel(),
            )[0, 1]
        )
        scored.append((score, name))
    score, mapping = max(scored)
    if not np.isfinite(score) or score < 0.3:
        return None

    height, width = gray.shape

    def to_png(x: float, y: float) -> tuple[float, float]:
        if mapping == "identity":
            return x, y
        if mapping == "flip_y":
            return x, height - 1 - y
        if mapping == "flip_x":
            return width - 1 - x, y
        return width - 1 - x, height - 1 - y

    plt = _matplotlib(output)
    from matplotlib.patches import Circle, Rectangle

    figure, axis = plt.subplots(figsize=(8.5, 8.5))
    axis.imshow(rgb, origin="upper")
    center_x, center_y = prep.nucleus_yx[1], prep.nucleus_yx[0]
    png_center_x, png_center_y = to_png(center_x, center_y)
    half = min(
        60, center_y, center_x,
        prep.image.data.shape[0] - center_y,
        prep.image.data.shape[1] - center_x,
    )
    crop_corners = [
        to_png(center_x + dx, center_y + dy)
        for dx in (-half, half)
        for dy in (-half, half)
    ]
    crop_x = min(item[0] for item in crop_corners)
    crop_y = min(item[1] for item in crop_corners)
    axis.add_patch(
        Rectangle(
            (crop_x, crop_y), 2 * half, 2 * half,
            fill=False, edgecolor="cyan", linewidth=1.3,
            label="diagnostic crop",
        )
    )
    for radius, linestyle, label in (
        (config.ring_inner_px, "--", "search inner radius"),
        (config.ring_outer_px, "-", "search outer radius"),
    ):
        axis.add_patch(
            Circle(
                (png_center_x, png_center_y), radius, fill=False,
                edgecolor="yellow", linestyle=linestyle, linewidth=1.1,
                label=label,
            )
        )
    axis.plot(png_center_x, png_center_y, "r+", ms=11, mew=1.8)
    for index, segment in enumerate(segments, start=1):
        x, y = to_png(segment.x, segment.y)
        axis.plot(x, y, "mx", ms=8, mew=1.4)
        axis.annotate(
            f"S{index}", (x + 3, y - 3), color="magenta",
            fontsize=9, weight="bold",
        )
    candidate_colors = {
        "sep_only": "orange",
        "sep_geometry_consistent": "lime",
        "geometry_recovered": "magenta",
    }
    for candidate_id, candidate_x, candidate_y, status in candidate_points or []:
        x, y = to_png(candidate_x, candidate_y)
        color = candidate_colors.get(status, "orange")
        axis.plot(x, y, "s", color=color, ms=5, mew=1.0)
        axis.annotate(
            candidate_id, (x + 3, y - 3), color=color,
            fontsize=9, weight="bold",
        )
    north, east = orientation_vectors(
        prep.image, (float(center_x), float(center_y))
    )
    if north is not None and east is not None:
        base_x, base_y = 35.0, 50.0
        mapped_center = np.asarray(to_png(center_x, center_y))
        for vector, label, color in (
            (north, "N", "lime"), (east, "E", "cyan")
        ):
            mapped_tip = np.asarray(
                to_png(center_x + vector[0], center_y + vector[1])
            )
            mapped_vector = mapped_tip - mapped_center
            length = float(np.hypot(*mapped_vector))
            mapped_vector = mapped_vector / max(length, 1e-9)
            axis.annotate(
                "",
                xy=(base_x + 25 * mapped_vector[0], base_y + 25 * mapped_vector[1]),
                xytext=(base_x, base_y),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
            )
            axis.text(
                base_x + 30 * mapped_vector[0],
                base_y + 30 * mapped_vector[1],
                label, color=color, weight="bold",
            )
    axis.set_title(
        f"RGB-to-FITS location audit: {prep.image.object_id}\n"
        f"orientation mapping={mapping}; image correlation={score:.3f}"
    )
    axis.set_xlabel("RGB pixel x")
    axis.set_ylabel("RGB pixel y")
    axis.legend(loc="upper left", fontsize=7)
    figure.tight_layout()
    path = output / "vision_rgb_context.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def write_manifest(
    prep: PreprocessedImage,
    config: VisionConfig,
    files: dict[str, str],
    output: Path,
) -> Path:
    input_path = Path(prep.image.path)
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "object_id": prep.image.object_id,
        "input_path": str(prep.image.path),
        "input_size_bytes": input_path.stat().st_size,
        "input_sha256": digest.hexdigest(),
        "band": prep.image.band,
        "astrometry": {
            "wcs_available": prep.image.wcs is not None,
            "pixel_scale_arcsec": prep.image.pixel_scale_arcsec,
            "display_origin": _origin(prep),
            "reprojected": False,
            "orientation_note": "Native pixel grid with WCS north/east arrows when available.",
        },
        "scientific_channel": "raw_foreground_subtracted_residual_snr",
        "wavelet_denoising_applied": False,
        "parameters": config.to_dict(),
        "process": [
            "load calibrated flux and RMS (or conservative PNG fallback)",
            "locate central light concentration and create masks",
            "fit foreground-galaxy light model",
            "compute residual and residual S/N",
            "run all configured S/N thresholds",
            "run SEP deblending parameter grid",
            "retain exact SEP pixel unions as trace-only geometry seeds",
            "propose circular/elliptical search routes from seeds and centre-based fallback",
            "run SEP again with pixels outside each route masked",
            "emit unified model-chain candidates pending calibration audit",
            "write structured Observation and trace artifacts",
        ],
        "files": files,
        "numerical_arrays": {
            "data": "loaded science image values",
            "rms": "supplied or conservatively estimated per-pixel noise",
            "model": "fitted foreground-galaxy light model",
            "residual": "flux minus foreground model",
            "residual_snr": "residual divided by RMS",
            "source_mask": "contaminating-source exclusion mask",
            "threshold_level_map": "number of accepted S/N threshold levels survived",
            "sep_raw_consensus_segmentation": "exact labels of the raw multi-parameter SEP pixel unions",
            "sep_raw_consensus_footprint": "all raw SEP consensus pixels; trace-only",
            "sep_geometry_seed_mask": "raw SEP pixels accepted into the configured annular geometry chain before local partitioning",
            "model_candidate_mask": "union of recorded local SEP partitions and route-recovered candidates pending audit; sub-minimum trace crumbs excluded",
            "sep_only_candidate_mask": "all raw SEP pixels outside admissible route-supported partitions; downgraded trace",
            "sep_geometry_consistent_mask": "local raw SEP pixel partitions inside admissible geometric routes",
            "geometry_recovered_candidate_mask": "supplementary candidates found by route-masked SEP",
            "residual_structure_mask": "empty compatibility array; no candidate is promoted to a scored structure in v0.11.0",
            "arc_mask": "empty deprecated compatibility array",
            "comparison_psf_mge_model": "audit-only PSF-convolved fixed-geometry MGE foreground model when available",
            "comparison_psf_mge_residual": "flux minus the audit-only PSF-MGE comparison model when available",
        },
        "notes": [
            "PNG files are human-readable diagnostics; NPZ arrays remain the numerical evidence.",
            "Threshold and SEP results share the same residual and are correlated measurements.",
            "Raw SEP detections are retained for traceability but are not exposed as scored structures.",
            "A geometric path restricts where SEP searches again; it is not evidence that the full path is luminous.",
            "A merged raw SEP component is locally split into route-supported pixels and downgraded off-route trace pixels.",
            "Route-consistent raw SEP and route-recovered components share review priority but retain different provenance.",
            "No wavelet channel is used in this version.",
        ],
    }
    path = output / "vision_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_visual_audit_bundle(
    prep: PreprocessedImage,
    config: VisionConfig,
    threshold_detections: list[StabilityDetection],
    consensus_detections: list[StabilityDetection],
    level_map: np.ndarray,
    footprint: np.ndarray,
    residual_structure_mask: np.ndarray,
    sep_only_candidate_mask: np.ndarray,
    sep_geometry_mask: np.ndarray,
    geometry_recovered_mask: np.ndarray,
    segments: list[ResidualStructure],
    geometric_paths: list[GeometricPathCandidate],
    foreground_comparison: ForegroundModelComparison,
    output: Path,
    candidate_points: list[tuple[str, float, float, str]] | None = None,
) -> dict[str, str]:
    try:
        threshold_path = write_threshold_process(prep, config, threshold_detections, level_map, output)
        sep_path = write_sep_consensus(prep, config, consensus_detections, output)
        structure_path = write_model_chain(
            prep,
            config,
            footprint,
            residual_structure_mask,
            sep_only_candidate_mask,
            sep_geometry_mask,
            geometry_recovered_mask,
            output,
        )
        path_search_path = write_geometric_path_search(prep, geometric_paths, output)
        expert_views_path = write_expert_views(prep, output)
        model_comparison_path = write_foreground_model_comparison(
            prep, foreground_comparison, output
        )
        # Raw SEP seeds are already shown exactly in the SEP diagnostic. Do not
        # mark them as formal measured structures in full-field/RGB context.
        field_path = write_field_context(
            prep, config, [], output, candidate_points=candidate_points
        )
        rgb_path = write_rgb_context(
            prep, config, [], output, candidate_points=candidate_points
        )
    except ImportError:
        return {}
    files = {
        "vision_arrays": str(output / "vision_artifacts.npz"),
        "overview_diagnostic": str(output / "vision_diagnostic.png"),
        "snr_threshold_process": str(threshold_path),
        "sep_consensus_diagnostic": str(sep_path),
        "model_chain_diagnostic": str(structure_path),
        "geometric_path_search_diagnostic": str(path_search_path),
        "expert_views_diagnostic": str(expert_views_path),
        "field_context_diagnostic": str(field_path),
        "vision_manifest": str(output / "vision_manifest.json"),
    }
    if model_comparison_path is not None:
        files["foreground_model_comparison_diagnostic"] = str(model_comparison_path)
    if rgb_path is not None:
        files["rgb_context_diagnostic"] = str(rgb_path)
    manifest_path = write_manifest(prep, config, files, output)
    files["vision_manifest"] = str(manifest_path)
    return files
