from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import numpy as np

from ..core.config import VisionConfig
from .astrometry import orientation_vectors, pixel_to_sky
from .detectors import ArcSegment, StabilityDetection, threshold_masks
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
    from matplotlib.patches import Ellipse as PlotEllipse

    ys, xs, half = _view(prep)
    figure, axis = plt.subplots(figsize=(8, 8))
    residual_view = prep.residual[ys, xs]
    vmin, vmax = _display_limits(residual_view)
    axis.imshow(residual_view, origin=_origin(prep), cmap="gray", vmin=vmin, vmax=vmax)
    for detection in detections:
        local_x = detection.x - (prep.nucleus_yx[1] - half)
        local_y = detection.y - (prep.nucleus_yx[0] - half)
        fraction = detection.survival / max(detection.total_runs, 1)
        ratio = detection.axis_ratio or 1.0
        width = 12.0
        height = max(3.0, width / max(ratio, 1.0))
        patch = PlotEllipse(
            (local_x, local_y), width=width, height=height,
            fill=False, edgecolor="lime" if detection.survival >= config.sep_min_survival else "orange",
            linewidth=0.8 + 2.2 * fraction,
        )
        axis.add_patch(patch)
        axis.annotate(
            f"{detection.survival}/{detection.total_runs}\n"
            f"切向偏差={detection.tangential_misalignment_deg:.1f}°",
            (local_x + 4, local_y - 4), color="yellow", fontsize=8,
        )
    _mark(axis, half)
    axis.set_title(
        f"SEP 多参数共识：{len(config.sep_nthresh)}×{len(config.sep_deblend_cont)} "
        f"= {len(config.sep_nthresh) * len(config.sep_deblend_cont)} 次运行\n"
        "绿色=达到存活门槛，橙色=低稳定候选"
    )
    figure.tight_layout()
    path = output / "vision_sep_consensus.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_arc_extraction(
    prep: PreprocessedImage,
    config: VisionConfig,
    footprint: np.ndarray,
    arc_mask: np.ndarray,
    output: Path,
) -> Path:
    plt = _matplotlib(output)
    ys, xs, half = _view(prep)
    snr_mask = prep.residual_snr >= config.arc_snr_low
    residual_view = prep.residual[ys, xs]
    vmin, vmax = _display_limits(residual_view)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    origin = _origin(prep)
    axes[0].imshow(residual_view, origin=origin, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("1. 前景减除残差")
    axes[1].imshow(footprint[ys, xs], origin=origin, cmap="gray")
    axes[1].set_title("2. SEP 候选结构范围")
    axes[2].imshow(snr_mask[ys, xs], origin=origin, cmap="gray")
    axes[2].set_title(f"3. S/N ≥ {config.arc_snr_low:.2f}σ")
    axes[3].imshow(residual_view, origin=origin, cmap="gray", vmin=vmin, vmax=vmax)
    yy, xx = np.where(arc_mask[ys, xs])
    axes[3].scatter(xx, yy, s=3, c="red")
    axes[3].set_title("4. 两者交集 = 测量残差结构像素")
    for axis in axes:
        _mark(axis, half)
    figure.suptitle("候选残差结构像素提取过程（不等同于透镜弧；红十字为中心）")
    figure.tight_layout()
    path = output / "vision_arc_extraction.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_field_context(
    prep: PreprocessedImage,
    config: VisionConfig,
    segments: list[ArcSegment],
    output: Path,
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
    segments: list[ArcSegment],
    output: Path,
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
            "intersect SEP footprint with S/N mask",
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
            "sep_footprint": "SEP footprint used by residual-structure extraction",
            "arc_mask": "pixels used for final arc-structure measurements",
        },
        "notes": [
            "PNG files are human-readable diagnostics; NPZ arrays remain the numerical evidence.",
            "Threshold and SEP results share the same residual and are correlated evidence.",
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
    arc_mask: np.ndarray,
    segments: list[ArcSegment],
    output: Path,
) -> dict[str, str]:
    try:
        threshold_path = write_threshold_process(prep, config, threshold_detections, level_map, output)
        sep_path = write_sep_consensus(prep, config, consensus_detections, output)
        arc_path = write_arc_extraction(prep, config, footprint, arc_mask, output)
        field_path = write_field_context(prep, config, segments, output)
        rgb_path = write_rgb_context(prep, config, segments, output)
    except ImportError:
        return {}
    files = {
        "vision_arrays": str(output / "vision_artifacts.npz"),
        "overview_diagnostic": str(output / "vision_diagnostic.png"),
        "snr_threshold_process": str(threshold_path),
        "sep_consensus_diagnostic": str(sep_path),
        "arc_extraction_diagnostic": str(arc_path),
        "field_context_diagnostic": str(field_path),
        "vision_manifest": str(output / "vision_manifest.json"),
    }
    if rgb_path is not None:
        files["rgb_context_diagnostic"] = str(rgb_path)
    manifest_path = write_manifest(prep, config, files, output)
    files["vision_manifest"] = str(manifest_path)
    return files
