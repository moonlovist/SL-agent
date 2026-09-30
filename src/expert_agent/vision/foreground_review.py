from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from .display_orientation import match_rgb_to_native
from .preprocess import ImageData, PreprocessedImage


@dataclass(frozen=True)
class ForegroundOverride:
    center_xy: tuple[float, float]
    foreground_seed: np.ndarray
    fit_exclude: np.ndarray
    input_sha256: str
    source_json: Path | None = None
    analysis_exclude: np.ndarray | None = None


def _native_to_fits_display(array: np.ndarray) -> np.ndarray:
    """Show native FITS row 0 at the bottom, matching diagnostic origin='lower'."""
    return np.flip(np.asarray(array), axis=0)


def _native_y_to_display(y: float, height: int) -> float:
    return float(height - 1) - float(y)


def _display_y_to_native(y: float, height: int) -> float:
    return float(height - 1) - float(y)


def _conservative_core_seed(
    prep: PreprocessedImage, candidate: dict
) -> np.ndarray:
    """Return a compact core seed, never the candidate's whole SEP island."""
    from scipy import ndimage

    label = int(candidate["sep_label"])
    selected = np.asarray(prep.segmentation) == label
    cx, cy = map(float, candidate["centroid_xy"])
    yy, xx = np.indices(selected.shape)
    a = max(float(candidate.get("a_px") or 7.0), 3.0)
    b = max(float(candidate.get("b_px") or a), 3.0)
    theta = float(candidate.get("theta_rad") or 0.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx, dy = xx - cx, yy - cy
    xrot = dx * cos_t + dy * sin_t
    yrot = -dx * sin_t + dy * cos_t
    # SEP a/b are second-moment scales.  A 0.9-scale ellipse deliberately
    # selects a high-surface-brightness initialization core rather than
    # claiming that the full low-threshold island is foreground light.
    compact = selected & ((xrot / a) ** 2 + (yrot / b) ** 2 <= 0.9**2)
    compact = ndimage.binary_closing(compact, iterations=1)
    compact = ndimage.binary_fill_holes(compact)
    labels, count = ndimage.label(compact)
    if count:
        nearest_y = int(np.clip(round(cy), 0, compact.shape[0] - 1))
        nearest_x = int(np.clip(round(cx), 0, compact.shape[1] - 1))
        chosen = int(labels[nearest_y, nearest_x])
        if chosen == 0:
            ids = np.arange(1, count + 1)
            centroids = ndimage.center_of_mass(compact, labels, ids)
            chosen = int(ids[np.argmin([
                (px - cx) ** 2 + (py - cy) ** 2 for py, px in centroids
            ])])
        compact = labels == chosen
    if int(compact.sum()) < 12:
        compact = selected & (np.hypot(xx - cx, yy - cy) <= max(4.0, min(a, b)))
    return np.asarray(compact, dtype=bool)


def _refine_center_from_seed(
    prep: PreprocessedImage,
    seed: np.ndarray,
    annotated_xy: tuple[float, float],
    max_shift_px: float = 3.0,
) -> tuple[float, float]:
    """Refine an approximate human point using only the bright seed core."""
    cx0, cy0 = map(float, annotated_xy)
    yy, xx = np.indices(seed.shape)
    radius = np.hypot(xx - cx0, yy - cy0)
    positive = np.clip(
        np.asarray(prep.image.data, float) - float(prep.background_median),
        0.0,
        None,
    )
    local = np.asarray(seed, bool) & (radius <= 8.0) & np.isfinite(positive)
    values = positive[local]
    if values.size < 6 or not np.any(values > 0):
        return cx0, cy0
    threshold = float(np.percentile(values[values > 0], 55.0))
    weights = np.where(local & (positive >= threshold), positive, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return cx0, cy0
    cx = float(np.sum(weights * xx) / total)
    cy = float(np.sum(weights * yy) / total)
    shift = float(np.hypot(cx - cx0, cy - cy0))
    if shift > max_shift_px:
        scale = max_shift_px / shift
        cx = cx0 + (cx - cx0) * scale
        cy = cy0 + (cy - cy0) * scale
    return cx, cy


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_foreground_override(
    path: str | Path,
    image: ImageData,
    center_xy: tuple[float, float],
    foreground_seed: np.ndarray,
    fit_exclude: np.ndarray,
    analysis_exclude: np.ndarray | None = None,
) -> ForegroundOverride:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = np.asarray(foreground_seed, dtype=bool)
    excluded = np.asarray(fit_exclude, dtype=bool)
    analysis_excluded = (
        np.zeros_like(seed, dtype=bool)
        if analysis_exclude is None
        else np.asarray(analysis_exclude, dtype=bool)
    )
    if (
        seed.shape != image.data.shape
        or excluded.shape != image.data.shape
        or analysis_excluded.shape != image.data.shape
    ):
        raise ValueError("Human foreground masks must match the native image shape")
    cx, cy = map(float, center_xy)
    if not (0 <= cx < image.data.shape[1] and 0 <= cy < image.data.shape[0]):
        raise ValueError("Human foreground centre lies outside the native image")
    if not seed.any():
        raise ValueError("Human foreground seed mask is empty")
    header = image.wcs.to_header() if image.wcs is not None else fits.Header()
    mask_path = path.with_name(path.stem + "_masks.fits")
    fits.HDUList([
        fits.PrimaryHDU(header=header),
        fits.ImageHDU(seed.astype(np.uint8), name="FOREGROUND"),
        fits.ImageHDU(excluded.astype(np.uint8), name="FITEXCL"),
        fits.ImageHDU(analysis_excluded.astype(np.uint8), name="ANALYSIS_EXCLUDE"),
    ]).writeto(mask_path, overwrite=True)
    input_hash = _file_sha256(image.path)
    payload = {
        "format_version": 2,
        "object_id": image.object_id,
        "input_path": str(image.path),
        "input_sha256": input_hash,
        "coordinate_system": "native_numpy_pixels_zero_based_xy",
        "center_xy": [cx, cy],
        "masks_fits": mask_path.name,
        "foreground_hdu": "FOREGROUND",
        "fit_exclude_hdu": "FITEXCL",
        "analysis_exclude_hdu": "ANALYSIS_EXCLUDE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "human_in_the_loop_foreground_selection",
        "grading_effect": "none_direct; full_pipeline_rerun_required",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ForegroundOverride(
        (cx, cy), seed, excluded, input_hash, path, analysis_excluded
    )


def load_foreground_override(
    path: str | Path, image: ImageData
) -> ForegroundOverride:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) not in {1, 2}:
        raise ValueError("Unsupported foreground override format")
    expected_hash = _file_sha256(image.path)
    if payload.get("input_sha256") != expected_hash:
        raise ValueError("Foreground override belongs to a different input FITS/PNG")
    mask_path = Path(payload["masks_fits"])
    if not mask_path.is_absolute():
        mask_path = path.parent / mask_path
    with fits.open(mask_path, memmap=False) as hdul:
        seed = np.asarray(hdul[payload.get("foreground_hdu", "FOREGROUND")].data, dtype=bool)
        excluded = np.asarray(hdul[payload.get("fit_exclude_hdu", "FITEXCL")].data, dtype=bool)
        analysis_hdu = payload.get("analysis_exclude_hdu", "ANALYSIS_EXCLUDE")
        analysis_excluded = (
            np.asarray(hdul[analysis_hdu].data, dtype=bool)
            if analysis_hdu in hdul
            else np.zeros_like(seed, dtype=bool)
        )
    if (
        seed.shape != image.data.shape
        or excluded.shape != image.data.shape
        or analysis_excluded.shape != image.data.shape
    ):
        raise ValueError("Foreground override mask shape does not match the input")
    cx, cy = map(float, payload["center_xy"])
    return ForegroundOverride(
        (cx, cy), seed, excluded, expected_hash, path, analysis_excluded
    )


def apply_foreground_override(
    prep: PreprocessedImage,
    override: ForegroundOverride,
    *,
    ring_outer_px: float = 25.0,
) -> PreprocessedImage:
    seed = np.asarray(override.foreground_seed, dtype=bool)
    excluded = np.asarray(override.fit_exclude, dtype=bool)
    analysis_excluded = (
        np.zeros_like(seed, dtype=bool)
        if override.analysis_exclude is None
        else np.asarray(override.analysis_exclude, dtype=bool)
    )
    if (
        seed.shape != prep.image.data.shape
        or excluded.shape != prep.image.data.shape
        or analysis_excluded.shape != prep.image.data.shape
    ):
        raise ValueError("Foreground override mask shape does not match preprocessing")
    annotated_cx, annotated_cy = map(float, override.center_xy)
    cx, cy = _refine_center_from_seed(
        prep, seed, (annotated_cx, annotated_cy)
    )
    segmentation = np.asarray(prep.segmentation, dtype=np.int32).copy()
    manual_label = int(segmentation.max()) + 1
    segmentation[seed] = manual_label
    # Every automatic mask was built around the former automatic target.  It
    # must be rebuilt after a human changes the target; carrying it over can
    # silently hide the newly selected lens environment from all detectors.
    from scipy import ndimage

    non_foreground = (segmentation != 0) & (segmentation != manual_label)
    fit_mask = ndimage.binary_dilation(non_foreground, iterations=3)
    fit_mask[seed] = False
    fit_mask[excluded] = True
    fit_mask[analysis_excluded] = True

    yy, xx = np.indices(seed.shape)
    usable_seed = seed & ~excluded
    values = np.clip(
        np.asarray(prep.image.data, float) - float(prep.background_median),
        0.0,
        None,
    )
    values = np.where(usable_seed, values, 0.0)
    total = float(values.sum())
    ellipticity = prep.foreground_ellipticity
    axis_ratio = prep.foreground_axis_ratio
    position_angle = prep.foreground_position_angle_deg
    if total > 0 and int(usable_seed.sum()) >= 6:
        dx, dy = xx - cx, yy - cy
        covariance = np.array([
            [np.sum(values * dx * dx), np.sum(values * dx * dy)],
            [np.sum(values * dx * dy), np.sum(values * dy * dy)],
        ]) / total
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        major = max(float(eigenvalues[order[0]]), 1e-8)
        minor = max(float(eigenvalues[order[1]]), 1e-8)
        vector = eigenvectors[:, order[0]]
        q = float(np.clip(np.sqrt(minor / major), 0.20, 1.0))
        axis_ratio = 1.0 / q
        ellipticity = 1.0 - q
        position_angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)

    selection = {
        **prep.center_selection,
        "status": "selected",
        "selected_sep_label": manual_label,
        "selected_xy": [int(round(cx)), int(round(cy))],
        "selected_centroid_xy": [cx, cy],
        "selection_basis": "human_in_the_loop_native_pixel_annotation",
        "reason": "human_foreground_override",
        "routing_only": True,
        "affects_grading": False,
        "human_override": {
            "source_json": str(override.source_json) if override.source_json else None,
            "input_sha256": override.input_sha256,
            "foreground_seed_pixel_count": int(seed.sum()),
            "fit_exclude_pixel_count": int(excluded.sum()),
            "analysis_exclude_pixel_count": int(analysis_excluded.sum()),
            "annotated_center_xy": [annotated_cx, annotated_cy],
            "refined_center_xy": [cx, cy],
            "center_refinement_shift_px": float(
                np.hypot(cx - annotated_cx, cy - annotated_cy)
            ),
            "foreground_seed_role": "conservative_initialization_not_full_extent",
        },
    }
    ix = int(np.clip(round(cx), 0, prep.image.data.shape[1] - 1))
    iy = int(np.clip(round(cy), 0, prep.image.data.shape[0] - 1))
    source_mask = np.zeros_like(seed, dtype=bool)
    for label in np.unique(segmentation):
        if label <= 0 or label == manual_label:
            continue
        component = segmentation == label
        component_y, component_x = np.where(component)
        if not component_y.size:
            continue
        separation = float(
            np.hypot(component_x.mean() - cx, component_y.mean() - cy)
        )
        if separation > float(ring_outer_px):
            source_mask |= component
    source_mask = ndimage.binary_dilation(source_mask, iterations=3)
    source_mask |= analysis_excluded
    source_mask[seed] = False
    override_flags = [*prep.flags, "human_foreground_override"]
    if analysis_excluded.any():
        override_flags.append("human_analysis_exclude")
    return replace(
        prep,
        segmentation=segmentation,
        fit_mask=fit_mask,
        source_mask=source_mask,
        nucleus_yx=(iy, ix),
        foreground_detected=True,
        foreground_peak_value=float(prep.image.data[iy, ix]),
        foreground_integrated_flux=total,
        foreground_ellipticity=ellipticity,
        foreground_position_angle_deg=position_angle,
        foreground_axis_ratio=axis_ratio,
        foreground_reliability=1.0,
        center_selection=selection,
        flags=override_flags,
        human_foreground_seed=seed,
        human_fit_exclude=excluded,
        human_analysis_exclude=analysis_excluded,
    )


def _create_napari_viewer(napari_module, *, title: str, ui_theme: str):
    """Create a hidden viewer so partially populated layers never flash onscreen."""
    viewer = napari_module.Viewer(title=title, show=False)
    try:
        viewer.theme = ui_theme
    except (AttributeError, TypeError, ValueError):
        # Older releases may expose only an application-wide theme.  The
        # editor remains usable with Napari's current default in that case.
        pass
    return viewer


def _add_colored_labels(viewer, data, *, name: str, opacity: float, color: str):
    """Add a one-label mask using the current Napari colormap API."""
    try:
        from napari.utils.colormaps import direct_colormap

        colormap = direct_colormap({0: "transparent", 1: color, None: "transparent"})
        return viewer.add_labels(
            data, name=name, opacity=opacity, colormap=colormap
        )
    except ImportError:
        # Compatibility with Napari releases predating DirectLabelColormap.
        return viewer.add_labels(
            data, name=name, opacity=opacity, color={1: color}
        )


def review_foreground_with_napari(
    image: ImageData,
    prep: PreprocessedImage,
    output_json: str | Path,
    existing: ForegroundOverride | None = None,
    ui_theme: str = "dark",
) -> ForegroundOverride:
    """Open the optional native-pixel editor and return an accepted override."""
    try:
        import napari
        from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
    except ImportError as exc:
        raise RuntimeError(
            'Interactive foreground review requires napari and Qt; install the "ui" optional dependency'
        ) from exc

    candidate_records = [
        item for item in prep.center_selection.get("candidates", [])
        if isinstance(item.get("sep_label"), int)
    ]
    selected_label = prep.center_selection.get("selected_sep_label")
    selected_record = next(
        (
            item for item in candidate_records
            if item["sep_label"] == selected_label
        ),
        None,
    )
    suggested_seed = (
        _conservative_core_seed(prep, selected_record)
        if selected_record is not None
        else np.zeros_like(image.data, dtype=bool)
    )
    if not suggested_seed.any():
        cy0, cx0 = prep.nucleus_yx
        yy, xx = np.indices(image.data.shape)
        suggested_seed = np.hypot(xx - cx0, yy - cy0) <= 6.0
    if existing is None:
        center_xy = tuple(map(float, prep.center_selection.get(
            "selected_centroid_xy", [prep.nucleus_yx[1], prep.nucleus_yx[0]]
        )))
        foreground_seed = suggested_seed
        fit_exclude = np.zeros_like(image.data, dtype=bool)
        analysis_exclude = np.zeros_like(image.data, dtype=bool)
    else:
        center_xy = existing.center_xy
        foreground_seed = existing.foreground_seed
        fit_exclude = existing.fit_exclude
        analysis_exclude = (
            np.zeros_like(image.data, dtype=bool)
            if existing.analysis_exclude is None
            else existing.analysis_exclude
        )

    viewer = _create_napari_viewer(
        napari,
        title=f"LensAgent foreground review — {image.object_id}",
        ui_theme=ui_theme,
    )
    height, width = image.data.shape
    finite = image.data[np.isfinite(image.data)]
    low, high = np.percentile(finite, [1.0, 99.7])
    viewer.add_image(
        _native_to_fits_display(image.data),
        name=f"Full {image.band}_FLUX linear display ({width}x{height})",
        colormap="gray", contrast_limits=(float(low), float(high)),
    )
    rgb_path = image.path.with_name("rgb_0.png")
    if rgb_path.exists():
        try:
            from PIL import Image

            rgb_context = np.asarray(Image.open(rgb_path).convert("RGB"))
            matched = match_rgb_to_native(rgb_context, image.data)
            if matched is not None:
                rgb_native, mapping, correlation = matched
                viewer.add_image(
                    _native_to_fits_display(rgb_native),
                    name=(
                        "RGB context (display only; "
                        f"mapping={mapping}, r={correlation:.2f})"
                    ),
                    rgb=True, visible=False,
                )
        except (OSError, ValueError):
            pass
    viewer.add_image(
        _native_to_fits_display(prep.model),
        name="Automatic foreground model", colormap="gray", visible=False
    )
    viewer.add_image(
        _native_to_fits_display(prep.residual_snr),
        name="Automatic residual S/N", colormap="coolwarm",
        contrast_limits=(-5.0, 5.0), visible=False,
    )
    candidates = [
        [_native_y_to_display(item["centroid_xy"][1], height), item["centroid_xy"][0]]
        for item in candidate_records
    ]
    if candidates:
        candidate_layer = viewer.add_points(
            np.asarray(candidates), name="Automatic 2-4 candidates",
            symbol="ring", face_color="yellow", border_color="yellow",
            size=9, opacity=0.70,
        )
    else:
        candidate_layer = None
    center_layer = viewer.add_points(
        np.asarray([[_native_y_to_display(center_xy[1], height), center_xy[0]]]),
        name="1 Foreground center (drag this point)",
        symbol="ring", face_color="lime", border_color="lime",
        size=12, opacity=0.75,
    )
    foreground_layer = _add_colored_labels(
        viewer,
        _native_to_fits_display(foreground_seed).astype(np.uint8),
        name="2 Foreground area (optional paint/erase)", opacity=0.40,
        color="#4FB1B2",
    )
    foreground_layer.selected_label = 1
    foreground_layer.brush_size = 5
    exclude_layer = _add_colored_labels(
        viewer,
        _native_to_fits_display(fit_exclude).astype(np.uint8),
        name="3 Do not fit: arcs/source (optional)", opacity=0.45,
        color="#71b0fd",
    )
    exclude_layer.selected_label = 1
    exclude_layer.brush_size = 4
    analysis_exclude_layer = _add_colored_labels(
        viewer,
        _native_to_fits_display(analysis_exclude).astype(np.uint8),
        name="4 Ignore unrelated regions (optional)", opacity=0.38,
        color="#fdc0e1",
    )
    analysis_exclude_layer.selected_label = 1
    analysis_exclude_layer.brush_size = 8

    state: dict[str, object] = {"accepted": False}
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(QLabel(
        f"当前显示完整 {width}x{height} {image.band}_FLUX（线性 1%–99.7%）；没有裁剪。\n"
        "最简操作：拖动绿色点到正确星系附近，再按下面第一个按钮。\n"
        "一般不需要手工抠图；需要时才编辑第 2 层前景范围，\n"
        "第 3 层保护 arc/环/源像；第 4 层屏蔽亮星等无关区域。RGB 层仅供定位。"
    ))
    status = QLabel(
        f"绿色点初始坐标 x={center_xy[0]:.1f}, y={center_xy[1]:.1f}；"
        "可直接拖动后选择最近对象。"
    )
    layout.addWidget(status)
    reset_full = QPushButton(f"回到完整 FITS 视野（{width}x{height}）")
    layout.addWidget(reset_full)
    toggle_markers = QPushButton("显示/隐藏绿色与黄色定位圈")
    layout.addWidget(toggle_markers)
    fill_nearest = QPushButton("按绿色点选择最近对象并生成第2层保守核心")
    layout.addWidget(fill_nearest)
    accept = QPushButton("接受人工前景并继续完整重跑")
    layout.addWidget(accept)

    def fill_nearest_candidate() -> None:
        points = np.asarray(center_layer.data, dtype=float)
        if points.shape != (1, 2) or not candidate_records:
            status.setText("需要恰好一个绿色点，并且图中必须有自动候选。")
            return
        display_y, px = map(float, points[0])
        py = _display_y_to_native(display_y, height)
        chosen = min(
            candidate_records,
            key=lambda item: (item["centroid_xy"][0] - px) ** 2
            + (item["centroid_xy"][1] - py) ** 2,
        )
        label = int(chosen["sep_label"])
        foreground_layer.data = _native_to_fits_display(
            _conservative_core_seed(prep, chosen)
        ).astype(np.uint8)
        center_layer.visible = False
        if candidate_layer is not None:
            candidate_layer.visible = False
        viewer.layers.selection.active = foreground_layer
        foreground_layer.mode = "paint"
        status.setText(
            f"已用 SEP {label} 生成保守核心；绿色点 x={px:.1f}, y={py:.1f} "
            "保持不变。定位圈已隐藏，现在可直接涂抹第2层。"
        )

    def reset_full_view() -> None:
        viewer.reset_view()
        points = np.asarray(center_layer.data, dtype=float)
        if points.shape == (1, 2):
            display_y, px = map(float, points[0])
            py = _display_y_to_native(display_y, height)
            status.setText(
                f"已回到完整 {width}x{height} 视野；绿色点 x={px:.1f}, y={py:.1f}。"
            )

    def toggle_marker_visibility() -> None:
        visible = not center_layer.visible
        center_layer.visible = visible
        if candidate_layer is not None:
            candidate_layer.visible = visible
        state_text = "显示" if visible else "隐藏"
        status.setText(f"绿色与黄色定位圈已{state_text}；这不会改变任何掩膜。")

    def accept_review() -> None:
        points = np.asarray(center_layer.data, dtype=float)
        if points.shape != (1, 2):
            status.setText("不能接受：绿色中心点必须恰好有一个。")
            return
        seed = _native_to_fits_display(
            np.asarray(foreground_layer.data, dtype=bool)
        )
        if not seed.any():
            status.setText("不能接受：青色前景范围不能为空。")
            return
        state.update(
            accepted=True,
            center_xy=(
                float(points[0, 1]),
                _display_y_to_native(float(points[0, 0]), height),
            ),
            foreground_seed=seed.copy(),
            fit_exclude=_native_to_fits_display(
                np.asarray(exclude_layer.data, dtype=bool)
            ).copy(),
            analysis_exclude=_native_to_fits_display(
                np.asarray(analysis_exclude_layer.data, dtype=bool)
            ).copy(),
            screenshot=viewer.screenshot(canvas_only=False),
        )
        viewer.close()

    reset_full.clicked.connect(reset_full_view)
    toggle_markers.clicked.connect(toggle_marker_visibility)
    fill_nearest.clicked.connect(fill_nearest_candidate)
    accept.clicked.connect(accept_review)
    viewer.window.add_dock_widget(panel, area="right", name="LensAgent instructions")
    viewer.layers.selection.active = center_layer
    center_layer.mode = "select"
    viewer.reset_view()
    viewer.show(block=False)
    print("MANUAL_REVIEW_WINDOW_READY", flush=True)
    napari.run()
    if not state["accepted"]:
        raise RuntimeError("Foreground review window closed without accepting an annotation")
    override = save_foreground_override(
        output_json, image, state["center_xy"],
        state["foreground_seed"], state["fit_exclude"], state["analysis_exclude"],
    )
    screenshot = np.asarray(state["screenshot"])
    from PIL import Image
    Image.fromarray(screenshot).save(Path(output_json).with_name("foreground_review_accepted.png"))
    return override
