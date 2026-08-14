from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import numpy as np

from ..core.config import VisionConfig
from .detectors import StabilityDetection, threshold_masks
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
    vmin, vmax = _display_limits(residual_view)

    for index, (threshold, mask) in enumerate(zip(thresholds, masks)):
        axes[0, index].imshow(mask[ys, xs], origin="upper", cmap="gray")
        axes[0, index].set_title(f"mask ≥ {threshold:.2f}σ")
        _mark(axes[0, index], half)

        axes[1, index].imshow(
            np.where(mask, prep.residual, np.nan)[ys, xs],
            origin="upper",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, index].set_title(f"残差 × mask ≥ {threshold:.2f}σ")
        _mark(axes[1, index], half)

        if index < columns - 1:
            shell = mask & ~masks[index + 1]
            axes[2, index].imshow(shell[ys, xs], origin="upper", cmap="gray")
            axes[2, index].set_title(
                f"层壳 {threshold:.2f}–{thresholds[index + 1]:.2f}σ"
            )
        else:
            axes[2, index].imshow(
                level_map[ys, xs],
                origin="upper",
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
    axis.imshow(residual_view, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
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
    axes[0].imshow(residual_view, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("1. 前景减除残差")
    axes[1].imshow(footprint[ys, xs], origin="upper", cmap="gray")
    axes[1].set_title("2. SEP 候选结构范围")
    axes[2].imshow(snr_mask[ys, xs], origin="upper", cmap="gray")
    axes[2].set_title(f"3. S/N ≥ {config.arc_snr_low:.2f}σ")
    axes[3].imshow(residual_view, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    yy, xx = np.where(arc_mask[ys, xs])
    axes[3].scatter(xx, yy, s=3, c="red")
    axes[3].set_title("4. 两者交集 = 测量弧像素")
    for axis in axes:
        _mark(axis, half)
    figure.suptitle("弧像素提取过程（红十字为中心）")
    figure.tight_layout()
    path = output / "vision_arc_extraction.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_flowchart(output: Path) -> Path:
    plt = _matplotlib(output)
    figure, axis = plt.subplots(figsize=(15, 5.8))
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 6)
    axis.axis("off")
    nodes = [
        (0.8, 4.4, "FITS / PNG\n输入图像"),
        (3.1, 4.4, "中心定位与\n污染源掩膜"),
        (5.4, 4.4, "前景星系\n椭圆模型"),
        (7.7, 4.4, "Residual\n残差 / S/N"),
        (10.0, 5.0, "S/N 多阈值\n稳定性"),
        (10.0, 3.8, "SEP 多参数\n共识"),
        (12.3, 4.4, "弧像素交集\n与几何测量"),
        (14.2, 4.4, "Observation\n结构化事实"),
        (7.7, 1.5, "NPZ 数值证据\n+ PNG 过程图\n+ Manifest 参数清单"),
    ]
    for x, y, label in nodes:
        axis.text(
            x, y, label, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="#eaf2ff", ec="#315a8a", lw=1.3),
        )

    def arrow(x1, y1, x2, y2):
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))

    arrow(1.65, 4.4, 2.25, 4.4)
    arrow(3.95, 4.4, 4.55, 4.4)
    arrow(6.25, 4.4, 6.85, 4.4)
    arrow(8.55, 4.55, 9.15, 4.9)
    arrow(8.55, 4.25, 9.15, 3.9)
    arrow(10.85, 5.0, 11.45, 4.6)
    arrow(10.85, 3.8, 11.45, 4.2)
    arrow(13.15, 4.4, 13.55, 4.4)
    arrow(7.7, 3.65, 7.7, 2.25)
    axis.text(7.5, 5.65, "LensAgent 视觉层可追溯处理流程", ha="center", fontsize=16, weight="bold")
    axis.text(7.5, 0.35, "视觉层只记录测量事实，不在此阶段输出‘透镜/非透镜’结论。", ha="center", fontsize=11, color="#7a1f1f")
    path = output / "vision_process_flow.png"
    figure.savefig(path, dpi=140, bbox_inches="tight")
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
            "flux": "loaded science image values",
            "rms": "supplied or conservatively estimated per-pixel noise",
            "model": "fitted foreground-galaxy light model",
            "residual": "flux minus foreground model",
            "residual_snr": "residual divided by RMS",
            "source_mask": "contaminating-source exclusion mask",
            "threshold_level_map": "number of accepted S/N threshold levels survived",
            "sep_consensus_footprint": "union footprint of stable SEP detections",
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
    output: Path,
) -> dict[str, str]:
    try:
        threshold_path = write_threshold_process(prep, config, threshold_detections, level_map, output)
        sep_path = write_sep_consensus(prep, config, consensus_detections, output)
        arc_path = write_arc_extraction(prep, config, footprint, arc_mask, output)
        flow_path = write_flowchart(output)
    except ImportError:
        return {}
    files = {
        "vision_arrays": str(output / "vision_artifacts.npz"),
        "overview_diagnostic": str(output / "vision_diagnostic.png"),
        "snr_threshold_process": str(threshold_path),
        "sep_consensus_diagnostic": str(sep_path),
        "arc_extraction_diagnostic": str(arc_path),
        "vision_process_flow": str(flow_path),
        "vision_manifest": str(output / "vision_manifest.json"),
    }
    manifest_path = write_manifest(prep, config, files, output)
    files["vision_manifest"] = str(manifest_path)
    return files
