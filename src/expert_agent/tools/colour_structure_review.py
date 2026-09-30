"""Human colour-structure annotation UI, isolated from LensAgent grading.

The tool records native-pixel foreground and labelled structure masks plus
human role/source-group notes.  It does not run foreground subtraction, colour
measurement, reasoning, or grading.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from ..vision.display_orientation import match_rgb_to_native
from ..vision.preprocess import ImageData, load_image


SCHEMA = "human_colour_structure_annotation_v1"
ROLE_OPTIONS = (
    ("independent_image_candidate", "独立像候选"),
    ("arc_or_ring_single_structure", "单弧／环整体"),
    ("unrelated_or_contaminant", "无关源／污染"),
    ("unresolved", "无法确定"),
)
SAMPLE_STATUS_OPTIONS = (
    ("in_progress", "标注中"),
    ("accepted", "已完成"),
    ("uncertain", "有疑问，保留"),
    ("skipped", "跳过／无法可靠标注"),
)
DISPLAY_STRETCH_OPTIONS = (
    ("linear", "Linear（线性）"),
    ("asinh_weak", "Asinh 弱（保留亮核）"),
    ("asinh_strong", "Asinh 强（提升暗弧）"),
    ("log", "Log（强调暗信号）"),
    ("sqrt", "Sqrt（中等提亮）"),
    ("mtf", "MTF（中间调强提升）"),
)
PAIR_GEOMETRY_OPTIONS = (
    ("unreviewed", "未复核"),
    ("plausible_same_source", "几何上可能同源"),
    ("not_same_source", "几何上不像同源"),
    ("uncertain", "仅凭几何无法判断"),
)
PAIR_COLOUR_OPTIONS = (
    ("unreviewed", "未复核"),
    ("visually_consistent", "肉眼颜色大致一致"),
    ("visually_different", "肉眼颜色明显不同"),
    ("uncertain", "肉眼无法判断颜色"),
)
PAIR_CONTAMINATION_OPTIONS = (
    ("unreviewed", "未复核"),
    ("none_obvious", "未见明显污染"),
    ("possible", "可能受前景／邻源污染"),
    ("strong", "污染明显，颜色不可用于定标"),
)
PAIR_CORRESPONDENCE_OPTIONS = (
    ("unreviewed", "未复核"),
    ("corresponding", "掩膜大致覆盖对应区域"),
    ("not_corresponding", "掩膜覆盖的物理区域不对应"),
    ("uncertain", "无法判断是否对应"),
)
BOUNDARY_GRADE_OPTIONS = (
    ("unreviewed", "未复核"),
    ("A", "A：仅凭当前成像已明确"),
    ("B", "B：很可能，但证据仍不够明确"),
    ("unknown", "无法判断"),
)
GEOMETRY_FEATURE_OPTIONS = (
    ("clear_long_curved_arc", "清晰、连续且有曲率的长弧"),
    ("clear_ring", "清晰的部分环／完整环"),
    ("convincing_multiple_images", "可信的 fold／cusp／多像关系"),
    ("only_short_or_fragmented", "只有短弧或碎片，不能独立支持A"),
    ("unrelated_structures_combined", "多个无关结构被组合成构型"),
    ("foreground_subtraction_residual", "前景减除残差可能解释"),
    ("ordinary_galaxy_or_neighbour", "普通星系形态／邻源可能解释"),
    ("uncertain_geometry", "几何关系看不清"),
)
BOUNDARY_REASON_OPTIONS = (
    ("arc_too_short_or_discontinuous", "弧太短或不连续"),
    ("curvature_not_reliable", "曲率不可靠"),
    ("no_lens_specific_morphology", "虽有亮结构，但缺少透镜特异形态"),
    ("global_lens_configuration_incoherent", "整体几何构型不自洽，无法组成统一透镜系统"),
    ("foreground_center_relation_unconvincing", "结构与前景偏折中心的关系不合理"),
    ("tangentiality_or_orientation_inconsistent", "切向性／朝向关系不一致"),
    ("multiple_image_relation_not_secure", "多像关系不充分"),
    ("fragment_or_image_identity_uncertain", "无法确认是独立像、同一弧碎片还是无关结构"),
    ("colour_tension_or_conflict", "颜色张力／不一致"),
    ("foreground_contamination", "前景污染严重"),
    ("foreground_lens_identification_uncertain", "前景透镜星系／中心识别不可靠"),
    ("insufficient_snr_or_resolution", "信噪比或分辨率不足"),
    ("competitive_non_lens_explanation", "非透镜解释同样有力"),
)


@dataclass(frozen=True)
class Sample:
    cohort: str
    object_id: str
    directory: Path
    fits_path: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def discover_samples(dataset_root: str | Path) -> list[Sample]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    samples: list[Sample] = []
    for cohort_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        cohort = cohort_dir.name
        for object_dir in sorted(path for path in cohort_dir.iterdir() if path.is_dir()):
            preferred = object_dir / f"{object_dir.name}.fits"
            fits_files = sorted(object_dir.glob("*.fits"))
            fits_path = preferred if preferred.exists() else (fits_files[0] if len(fits_files) == 1 else None)
            if fits_path is None:
                continue
            samples.append(Sample(cohort, object_dir.name, object_dir, fits_path.resolve()))
    if not samples:
        raise FileNotFoundError(f"No one-FITS object directories found under: {root}")
    return samples


def annotation_paths(output_root: str | Path, sample: Sample) -> tuple[Path, Path]:
    directory = Path(output_root).expanduser().resolve() / sample.cohort / sample.object_id
    return directory / "annotation.json", directory / "masks.fits"


def _native_to_display(array: np.ndarray) -> np.ndarray:
    return np.flip(np.asarray(array), axis=0)


def _display_to_native(array: np.ndarray) -> np.ndarray:
    return np.flip(np.asarray(array), axis=0)


def display_stretch(
    data: np.ndarray,
    method: str,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    """Return a monotonic 0..1 display transform without changing science data."""
    values = np.asarray(data, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, dtype=float)
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Display percentiles must satisfy 0 <= low < high <= 100")
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(values.shape, dtype=float)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    if method == "linear":
        shown = normalized
    elif method == "asinh_weak":
        q = 5.0
        shown = np.arcsinh(q * normalized) / np.arcsinh(q)
    elif method == "asinh_strong":
        q = 20.0
        shown = np.arcsinh(q * normalized) / np.arcsinh(q)
    elif method == "log":
        q = 100.0
        shown = np.log1p(q * normalized) / np.log1p(q)
    elif method == "sqrt":
        shown = np.sqrt(normalized)
    elif method == "mtf":
        midpoint = 0.20
        denominator = (2.0 * midpoint - 1.0) * normalized - midpoint
        shown = np.divide(
            (midpoint - 1.0) * normalized,
            denominator,
            out=np.zeros_like(normalized),
            where=np.abs(denominator) > 1e-12,
        )
    else:
        raise ValueError(f"Unknown display stretch: {method}")
    return np.clip(shown, 0.0, 1.0)


def empty_annotation_state(shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "foreground": np.zeros(shape, dtype=bool),
        "structures": np.zeros(shape, dtype=np.int16),
        "records": {},
        "sample_status": "in_progress",
        "sample_note": "",
        "pair_reviews": {},
        "geometry_boundary_review": {},
    }


def load_annotation(
    output_root: str | Path,
    sample: Sample,
    image: ImageData,
) -> dict[str, Any]:
    json_path, masks_path = annotation_paths(output_root, sample)
    if not json_path.exists() and not masks_path.exists():
        return empty_annotation_state(image.data.shape)
    if not json_path.exists() or not masks_path.exists():
        raise ValueError(f"Incomplete annotation pair for {sample.object_id}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported annotation schema for {sample.object_id}")
    if payload.get("input_sha256") != file_sha256(sample.fits_path):
        raise ValueError(f"Annotation input hash mismatch for {sample.object_id}")
    with fits.open(masks_path, memmap=False) as hdus:
        foreground = np.asarray(hdus["FOREGROUND"].data, dtype=bool)
        structures = np.asarray(hdus["STRUCTURES"].data, dtype=np.int16)
    if foreground.shape != image.data.shape or structures.shape != image.data.shape:
        raise ValueError(f"Annotation mask shape mismatch for {sample.object_id}")
    records = {
        int(item["label"]): {
            "role": item.get("role", "unresolved"),
            "source_group": item.get("source_group", ""),
            "note": item.get("note", ""),
        }
        for item in payload.get("structures", [])
    }
    pair_reviews = {
        item["pair_id"]: {
            "first_label": int(item["first_label"]),
            "second_label": int(item["second_label"]),
            "geometry_relation": item.get("geometry_relation", "unreviewed"),
            "visual_colour_relation": item.get("visual_colour_relation", "unreviewed"),
            "contamination": item.get("contamination", "unreviewed"),
            "mask_correspondence": item.get("mask_correspondence", "unreviewed"),
            "note": item.get("note", ""),
        }
        for item in payload.get("pair_reviews", [])
        if item.get("pair_id")
    }
    return {
        "foreground": foreground,
        "structures": structures,
        "records": records,
        "sample_status": payload.get("sample_status", "in_progress"),
        "sample_note": payload.get("sample_note", ""),
        "pair_reviews": pair_reviews,
        "geometry_boundary_review": payload.get("geometry_boundary_review", {}),
    }


def save_annotation(
    output_root: str | Path,
    dataset_root: str | Path,
    sample: Sample,
    image: ImageData,
    foreground: np.ndarray,
    structures: np.ndarray,
    records: dict[int, dict[str, str]],
    sample_status: str,
    sample_note: str,
    pair_reviews: dict[str, dict[str, Any]] | None = None,
    geometry_boundary_review: dict[str, Any] | None = None,
) -> Path:
    foreground = np.asarray(foreground, dtype=bool)
    structures = np.asarray(structures, dtype=np.int16)
    if foreground.shape != image.data.shape or structures.shape != image.data.shape:
        raise ValueError("Annotation masks must match native image shape")
    if np.any(structures < 0):
        raise ValueError("Structure labels must be non-negative integers")
    json_path, masks_path = annotation_paths(output_root, sample)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    header = image.wcs.to_header() if image.wcs is not None else fits.Header()
    header["OBJECTID"] = sample.object_id
    header["COHORT"] = sample.cohort
    header["COORDSYS"] = "NATIVE_NUMPY_ZERO_XY"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(foreground.astype(np.uint8), name="FOREGROUND"),
            fits.ImageHDU(structures.astype(np.int16), name="STRUCTURES"),
        ]
    ).writeto(masks_path, overwrite=True)
    structure_rows = []
    labels = sorted(int(value) for value in np.unique(structures) if int(value) > 0)
    for label in sorted(set(labels).union(records)):
        mask = structures == label
        record = records.get(label, {})
        structure_rows.append(
            {
                "label": label,
                "structure_id": f"S{label}",
                "role": record.get("role", "unresolved"),
                "source_group": record.get("source_group", ""),
                "note": record.get("note", ""),
                "pixel_count": int(mask.sum()),
                "mask_sha256": mask_sha256(mask),
            }
        )
    payload = {
        "schema": SCHEMA,
        "purpose": "colour_and_geometry_boundary_calibration_only_no_agent_grade_effect",
        "object_id": sample.object_id,
        "cohort": sample.cohort,
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "input_path": str(sample.fits_path),
        "input_sha256": file_sha256(sample.fits_path),
        "band_displayed": image.band,
        "native_shape_yx": list(image.data.shape),
        "coordinate_system": "native_numpy_pixels_zero_based_xy",
        "masks_fits": masks_path.name,
        "foreground_hdu": "FOREGROUND",
        "structures_hdu": "STRUCTURES",
        "foreground_pixel_count": int(foreground.sum()),
        "foreground_mask_sha256": mask_sha256(foreground),
        "structure_label_mask_sha256": hashlib.sha256(structures.tobytes()).hexdigest(),
        "sample_status": sample_status,
        "sample_note": sample_note,
        "structures": structure_rows,
        "pair_reviews": [
            {"pair_id": pair_id, **record}
            for pair_id, record in sorted((pair_reviews or {}).items())
        ],
        "geometry_boundary_review": geometry_boundary_review or {},
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "grading_effect": "none",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def load_agent_review_context(
    agent_project: str | Path | None,
    object_id: str,
) -> dict[str, Any]:
    """Load the exact current UI run as read-only context for human review."""
    if agent_project is None:
        return {}
    project = Path(agent_project).expanduser().resolve()
    state_path = project / "lensagent_ui_state.json"
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        item = state.get("objects", {}).get(object_id, {})
        run_id = item.get("current_run_id")
        if not run_id:
            return {}
        summary_path = project / "runs" / run_id / object_id / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {}
    decision = next(
        (
            row.get("content", {})
            for row in reversed(summary.get("reasoning_trace", []))
            if row.get("action") == "decision_conditions"
        ),
        {},
    )
    configurations = [
        row.get("id")
        for row in summary.get("evidence_dossier", {}).get(
            "configuration_hypotheses", []
        )
        if row.get("status") == "supported"
    ]
    return {
        "run_id": run_id,
        "summary_path": str(summary_path),
        "agent_grade": summary.get("expert_assessment", {}).get("vote_category"),
        "clear_system": decision.get("clear_system"),
        "colour_evidence_state": decision.get("colour_evidence_state"),
        "colour_caps_A": decision.get("colour_caps_A"),
        "supported_configurations": configurations,
    }


def write_batch_index(
    output_root: str | Path,
    dataset_root: str | Path,
    samples: list[Sample],
) -> Path:
    output = Path(output_root).expanduser().resolve()
    rows = []
    for sample in samples:
        json_path, _ = annotation_paths(output, sample)
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "cohort": sample.cohort,
                    "object_id": sample.object_id,
                    "status": payload.get("sample_status", "in_progress"),
                    "structure_count": len(payload.get("structures", [])),
                    "foreground_pixel_count": payload.get("foreground_pixel_count", 0),
                    "geometry_boundary_grade": payload.get(
                        "geometry_boundary_review", {}
                    ).get("all_imaging_grade", "unreviewed"),
                    "annotation_json": str(json_path),
                }
            )
        else:
            rows.append(
                {
                    "cohort": sample.cohort,
                    "object_id": sample.object_id,
                    "status": "not_started",
                    "structure_count": 0,
                    "foreground_pixel_count": 0,
                    "geometry_boundary_grade": "unreviewed",
                    "annotation_json": None,
                }
            )
    payload = {
        "schema": "human_colour_structure_batch_index_v1",
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "sample_count": len(samples),
        "objects": rows,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "annotation_index.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _rgb_for_sample(sample: Sample, image: ImageData) -> tuple[np.ndarray, str, float] | None:
    from PIL import Image

    for name in ("rgb_0.png", "rgb_1.png"):
        path = sample.directory / name
        if not path.exists():
            continue
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            matched = match_rgb_to_native(rgb, image.data)
            if matched is not None:
                return matched
        except (OSError, ValueError):
            continue
    return None


def run_review_ui(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    band: str = "VIS",
    theme: str = "dark",
    start_object: str | None = None,
    pair_review_manifest: str | Path | None = None,
    agent_project: str | Path | None = None,
) -> None:
    try:
        import napari
        from qtpy import QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            'Colour-structure review requires napari/Qt; install the "ui" optional dependency'
        ) from exc

    samples = discover_samples(dataset_root)
    pair_manifest_by_object: dict[str, dict[str, Any]] = {}
    if pair_review_manifest is not None:
        manifest_path = Path(pair_review_manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pair_manifest_by_object = {
            item["object_id"]: item for item in manifest.get("objects", [])
        }
        samples = [sample for sample in samples if sample.object_id in pair_manifest_by_object]
        if not samples:
            raise ValueError("Pair-review manifest does not match any dataset objects")
    output_root = Path(output_root).expanduser().resolve()
    index = 0
    if start_object is not None:
        matches = [i for i, sample in enumerate(samples) if sample.object_id == start_object]
        if not matches:
            raise ValueError(f"Unknown --start-object: {start_object}")
        index = matches[0]
    else:
        # Resume an explicitly in-progress object before advancing to the first
        # never-saved sample.  Completed/uncertain/skipped objects remain
        # reachable with the Previous button.
        in_progress = []
        for i, sample in enumerate(samples):
            json_path = annotation_paths(output_root, sample)[0]
            if not json_path.exists():
                continue
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("sample_status") == "in_progress":
                in_progress.append((payload.get("updated_utc", ""), i))
        first_unsaved = next(
            (
                i
                for i, sample in enumerate(samples)
                if not annotation_paths(output_root, sample)[0].exists()
            ),
            None,
        )
        if in_progress:
            index = max(in_progress)[1]
        elif first_unsaved is not None:
            index = first_unsaved

        # Existing colour annotations may all be complete before the geometry
        # boundary protocol is introduced. Resume at the first object whose
        # new boundary judgement is still empty instead of reopening object 1.
        if first_unsaved is None and not pair_manifest_by_object:
            for i, sample in enumerate(samples):
                json_path = annotation_paths(output_root, sample)[0]
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    index = i
                    break
                grade = payload.get("geometry_boundary_review", {}).get(
                    "all_imaging_grade", "unreviewed"
                )
                if grade == "unreviewed":
                    index = i
                    break

    if pair_manifest_by_object and start_object is None:
        # In pair-review mode resume at the first object that still has an
        # unreviewed proposed pair. Sample-level completion predates pair
        # review and therefore cannot be used as the resume criterion.
        for i, sample in enumerate(samples):
            json_path = annotation_paths(output_root, sample)[0]
            existing_pair_ids: set[str] = set()
            if json_path.exists():
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    existing_pair_ids = {
                        item.get("pair_id")
                        for item in payload.get("pair_reviews", [])
                        if item.get("pair_id")
                        and item.get("geometry_relation", "unreviewed") != "unreviewed"
                    }
                except (OSError, json.JSONDecodeError):
                    existing_pair_ids = set()
            proposed_ids = {
                "S{}-S{}".format(*sorted((int(item["first_label"]), int(item["second_label"]))))
                for item in pair_manifest_by_object[sample.object_id].get("pairs", [])
            }
            if proposed_ids - existing_pair_ids:
                index = i
                break

    # napari 0.9 removed the constructor's ``theme=`` keyword while older
    # supported releases accepted it.  Set the property after construction so
    # the standalone reviewer remains compatible across both API shapes.
    viewer = napari.Viewer(title="LensAgent colour-structure annotation")
    try:
        viewer.theme = theme
    except (AttributeError, ValueError):
        pass
    state: dict[str, Any] = {
        "index": index,
        "image": None,
        "sample": None,
        "records": {},
        "pair_reviews": {},
        "geometry_boundary_review": {},
        "agent_context": {},
        "loading": False,
    }

    panel = QtWidgets.QWidget()
    outer_layout = QtWidgets.QVBoxLayout(panel)
    progress = QtWidgets.QLabel()
    instructions = QtWidgets.QLabel(
        "1. Foreground reference 层画前景参照（可选）。\n"
        "2. 在 Structures 层选 S1/S2… 后画精确区域。\n"
        "3. 同一连续弧/环的碎片用同一个 S 标签；独立像用不同标签。\n"
        "4. 若多个独立像可能同源，给它们填相同的同源组，如 SRC1。"
    )
    instructions.setWordWrap(True)
    outer_layout.addWidget(progress)
    outer_layout.addWidget(instructions)
    nav = QtWidgets.QHBoxLayout()
    previous_button = QtWidgets.QPushButton("上一个")
    save_button = QtWidgets.QPushButton("保存")
    next_button = QtWidgets.QPushButton(
        "保存并下一个像对" if pair_manifest_by_object else "保存并下一个"
    )
    nav.addWidget(previous_button)
    nav.addWidget(save_button)
    nav.addWidget(next_button)
    outer_layout.addLayout(nav)
    message = QtWidgets.QLabel()
    message.setWordWrap(True)
    outer_layout.addWidget(message)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll_content = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(scroll_content)
    scroll.setWidget(scroll_content)
    outer_layout.addWidget(scroll, 1)

    display_box = QtWidgets.QGroupBox("图像显示拉伸（不改数据）")
    display_form = QtWidgets.QFormLayout(display_box)
    stretch_combo = QtWidgets.QComboBox()
    for value, title in DISPLAY_STRETCH_OPTIONS:
        stretch_combo.addItem(title, value)
    stretch_combo.setCurrentIndex(stretch_combo.findData("asinh_strong"))
    low_percentile = QtWidgets.QDoubleSpinBox()
    low_percentile.setRange(0.0, 49.9)
    low_percentile.setDecimals(1)
    low_percentile.setSingleStep(0.5)
    low_percentile.setValue(1.0)
    high_percentile = QtWidgets.QDoubleSpinBox()
    high_percentile.setRange(50.0, 100.0)
    high_percentile.setDecimals(1)
    high_percentile.setSingleStep(0.1)
    high_percentile.setValue(99.7)
    apply_stretch = QtWidgets.QPushButton("应用显示拉伸")
    display_form.addRow("方法", stretch_combo)
    display_form.addRow("黑点百分位", low_percentile)
    display_form.addRow("白点百分位", high_percentile)
    display_form.addRow(apply_stretch)
    layout.addWidget(display_box)

    structure_box = QtWidgets.QGroupBox("当前结构备注")
    form = QtWidgets.QFormLayout(structure_box)
    label_spin = QtWidgets.QSpinBox()
    label_spin.setRange(1, 99)
    label_spin.setValue(1)
    role_combo = QtWidgets.QComboBox()
    for value, title in ROLE_OPTIONS:
        role_combo.addItem(title, value)
    source_group = QtWidgets.QLineEdit()
    source_group.setPlaceholderText("仅分离结构可能同源时填，例如 SRC1；不确定留空")
    structure_note = QtWidgets.QPlainTextEdit()
    structure_note.setPlaceholderText("例：西侧反像；与S1几何上可能同源")
    structure_note.setMaximumHeight(78)
    save_structure = QtWidgets.QPushButton("保存当前结构备注")
    form.addRow("标签", label_spin)
    form.addRow("角色", role_combo)
    form.addRow("可能同源组（可留空）", source_group)
    form.addRow("备注", structure_note)
    form.addRow(save_structure)
    layout.addWidget(structure_box)

    pair_box = QtWidgets.QGroupBox("不确定结构的颜色／同源像对复核")
    pair_form = QtWidgets.QFormLayout(pair_box)
    pair_selector = QtWidgets.QComboBox()
    pair_geometry = QtWidgets.QComboBox()
    pair_colour = QtWidgets.QComboBox()
    pair_contamination = QtWidgets.QComboBox()
    pair_correspondence = QtWidgets.QComboBox()
    for value, title in PAIR_GEOMETRY_OPTIONS:
        pair_geometry.addItem(title, value)
    for value, title in PAIR_COLOUR_OPTIONS:
        pair_colour.addItem(title, value)
    for value, title in PAIR_CONTAMINATION_OPTIONS:
        pair_contamination.addItem(title, value)
    for value, title in PAIR_CORRESPONDENCE_OPTIONS:
        pair_correspondence.addItem(title, value)
    pair_diagnostic = QtWidgets.QLabel("未载入像对诊断")
    pair_diagnostic.setWordWrap(True)
    pair_note = QtWidgets.QPlainTextEdit()
    pair_note.setPlaceholderText("只记录你看到的原因；不要根据A/B标签反推答案")
    pair_note.setMaximumHeight(70)
    save_pair = QtWidgets.QPushButton("保存当前像对复核")
    pair_form.addRow("待复核像对", pair_selector)
    pair_form.addRow("自动测量（仅提示）", pair_diagnostic)
    pair_form.addRow("几何关系", pair_geometry)
    pair_form.addRow("肉眼颜色", pair_colour)
    pair_form.addRow("污染风险", pair_contamination)
    pair_form.addRow("掩膜对应性", pair_correspondence)
    pair_form.addRow("备注", pair_note)
    pair_form.addRow(save_pair)
    layout.addWidget(pair_box)

    sample_box = QtWidgets.QGroupBox("整个样本")
    sample_form = QtWidgets.QFormLayout(sample_box)
    sample_status = QtWidgets.QComboBox()
    for value, title in SAMPLE_STATUS_OPTIONS:
        sample_status.addItem(title, value)
    sample_note = QtWidgets.QPlainTextEdit()
    sample_note.setPlaceholderText("例：前景不确定；S1/S2可能是fold pair")
    sample_note.setMaximumHeight(72)
    sample_form.addRow("状态", sample_status)
    sample_form.addRow("备注", sample_note)
    layout.addWidget(sample_box)

    boundary_box = QtWidgets.QGroupBox("几何 A/B 边界复核（不直接修改 Agent 评级）")
    boundary_layout = QtWidgets.QVBoxLayout(boundary_box)
    agent_context_label = QtWidgets.QLabel("当前 Agent 结果：未找到")
    agent_context_label.setWordWrap(True)
    boundary_layout.addWidget(agent_context_label)
    boundary_form = QtWidgets.QFormLayout()
    vis_grade = QtWidgets.QComboBox()
    all_imaging_grade = QtWidgets.QComboBox()
    for value, title in BOUNDARY_GRADE_OPTIONS:
        vis_grade.addItem(title, value)
        all_imaging_grade.addItem(title, value)
    boundary_form.addRow("仅看VIS形态", vis_grade)
    boundary_form.addRow("结合颜色后的成像等级", all_imaging_grade)
    boundary_layout.addLayout(boundary_form)
    feature_checks: dict[str, Any] = {}
    feature_group = QtWidgets.QGroupBox("你实际看到的几何特征（可多选）")
    feature_layout = QtWidgets.QVBoxLayout(feature_group)
    for value, title in GEOMETRY_FEATURE_OPTIONS:
        check = QtWidgets.QCheckBox(title)
        feature_checks[value] = check
        feature_layout.addWidget(check)
    boundary_layout.addWidget(feature_group)
    reason_checks: dict[str, Any] = {}
    reason_group = QtWidgets.QGroupBox("若不是A，原因（可多选）")
    reason_layout = QtWidgets.QVBoxLayout(reason_group)
    for value, title in BOUNDARY_REASON_OPTIONS:
        check = QtWidgets.QCheckBox(title)
        reason_checks[value] = check
        reason_layout.addWidget(check)
    boundary_layout.addWidget(reason_group)
    boundary_note = QtWidgets.QPlainTextEdit()
    boundary_note.setPlaceholderText("只写肉眼判断依据；不需要填写数值阈值")
    boundary_note.setMaximumHeight(72)
    boundary_layout.addWidget(boundary_note)
    layout.addWidget(boundary_box)

    layout.addStretch(1)

    layers: dict[str, Any] = {}

    def combo_set_data(combo, value: str) -> None:
        found = combo.findData(value)
        combo.setCurrentIndex(found if found >= 0 else 0)

    def persist_structure_form() -> None:
        if state["loading"]:
            return
        label_value = int(label_spin.value())
        state["records"][label_value] = {
            "role": str(role_combo.currentData()),
            "source_group": source_group.text().strip(),
            "note": structure_note.toPlainText().strip(),
        }
        if "structures" in layers:
            layers["structures"].selected_label = label_value
        message.setText(f"S{label_value} 备注已记录（仍需保存样本）")

    def selected_pair() -> dict[str, Any] | None:
        return pair_selector.currentData()

    def pair_id(first_label: int, second_label: int) -> str:
        first, second = sorted((int(first_label), int(second_label)))
        return f"S{first}-S{second}"

    def persist_pair_form() -> None:
        if state["loading"]:
            return
        item = selected_pair()
        if not item:
            return
        key = pair_id(item["first_label"], item["second_label"])
        state["pair_reviews"][key] = {
            "first_label": int(item["first_label"]),
            "second_label": int(item["second_label"]),
            "geometry_relation": str(pair_geometry.currentData()),
            "visual_colour_relation": str(pair_colour.currentData()),
            "contamination": str(pair_contamination.currentData()),
            "mask_correspondence": str(pair_correspondence.currentData()),
            "note": pair_note.toPlainText().strip(),
        }
        message.setText(f"{key} 复核已记录（仍需保存样本）")

    def load_pair_form() -> None:
        item = selected_pair()
        state["loading"] = True
        if not item:
            pair_diagnostic.setText("本样本没有待复核像对")
            pair_note.clear()
            pair_box.setEnabled(False)
            for key in ("pair_first", "pair_second"):
                if key in layers:
                    layers[key].visible = False
            state["loading"] = False
            return
        pair_box.setEnabled(True)
        first_label = int(item["first_label"])
        second_label = int(item["second_label"])
        if "structures" in layers:
            labelled = np.asarray(layers["structures"].data)
            if "pair_first" in layers:
                layers["pair_first"].data = (labelled == first_label).astype(np.uint8)
                layers["pair_first"].name = f"当前像对A：S{first_label}（青色）"
                layers["pair_first"].visible = True
            if "pair_second" in layers:
                layers["pair_second"].data = (labelled == second_label).astype(np.uint8)
                layers["pair_second"].name = f"当前像对B：S{second_label}（洋红）"
                layers["pair_second"].visible = True
        key = pair_id(first_label, second_label)
        record = state["pair_reviews"].get(key, {})
        combo_set_data(pair_geometry, record.get("geometry_relation", "unreviewed"))
        combo_set_data(pair_colour, record.get("visual_colour_relation", "unreviewed"))
        combo_set_data(pair_contamination, record.get("contamination", "unreviewed"))
        combo_set_data(pair_correspondence, record.get("mask_correspondence", "unreviewed"))
        pair_note.setPlainText(record.get("note", ""))
        pair_diagnostic.setText(
            f"青色=S{first_label}，洋红=S{second_label}。"
            f"共同可靠波段={item.get('common_bands', '未知')}；"
            f"最大非灰度偏离={item.get('maximum_departure_mag', '未知')} mag；"
            f"最大显著度={item.get('maximum_departure_sigma', '未知')}σ；"
            f"约化χ²={item.get('reduced_chi2', '未知')}；"
            f"当前状态={item.get('current_status', '未知')}。"
        )
        state["loading"] = False

    def load_structure_form(label_value: int) -> None:
        state["loading"] = True
        record = state["records"].get(int(label_value), {})
        combo_set_data(role_combo, record.get("role", "independent_image_candidate"))
        source_group.setText(record.get("source_group", ""))
        structure_note.setPlainText(record.get("note", ""))
        if "structures" in layers:
            layers["structures"].selected_label = int(label_value)
        state["loading"] = False

    def save_current() -> None:
        persist_structure_form()
        persist_pair_form()
        sample = state["sample"]
        image = state["image"]
        if sample is None or image is None:
            return
        path = save_annotation(
            output_root,
            dataset_root,
            sample,
            image,
            _display_to_native(layers["foreground"].data).astype(bool),
            _display_to_native(layers["structures"].data).astype(np.int16),
            state["records"],
            str(sample_status.currentData()),
            sample_note.toPlainText().strip(),
            state["pair_reviews"],
            {
                "vis_morphology_grade": str(vis_grade.currentData()),
                "all_imaging_grade": str(all_imaging_grade.currentData()),
                "geometry_features": sorted(
                    key for key, check in feature_checks.items() if check.isChecked()
                ),
                "non_A_reasons": sorted(
                    key for key, check in reason_checks.items() if check.isChecked()
                ),
                "note": boundary_note.toPlainText().strip(),
                "agent_context_at_review": state.get("agent_context", {}),
            },
        )
        write_batch_index(output_root, dataset_root, samples)
        screenshot = viewer.screenshot(canvas_only=False)
        from PIL import Image

        Image.fromarray(np.asarray(screenshot)).save(path.parent / "review_screenshot.png")
        message.setText(f"已保存：{path}")

    def update_display_stretch() -> None:
        image = state.get("image")
        if image is None or "vis" not in layers:
            return
        try:
            shown = display_stretch(
                image.data,
                str(stretch_combo.currentData()),
                float(low_percentile.value()),
                float(high_percentile.value()),
            )
        except ValueError as exc:
            message.setText(str(exc))
            return
        layers["vis"].data = _native_to_display(shown)
        layers["vis"].contrast_limits = (0.0, 1.0)
        layers["vis"].name = (
            f"{band} display: {stretch_combo.currentData()} "
            f"p{low_percentile.value():.1f}–p{high_percentile.value():.1f}"
        )
        message.setText("已更新显示拉伸；原始FITS和标注掩膜未改变。")

    def load_current() -> None:
        state["loading"] = True
        sample = samples[state["index"]]
        image = load_image(sample.fits_path, band)
        saved = load_annotation(output_root, sample, image)
        state.update(
            sample=sample,
            image=image,
            records=saved["records"],
            pair_reviews=saved["pair_reviews"],
            geometry_boundary_review=saved["geometry_boundary_review"],
        )
        state["agent_context"] = load_agent_review_context(
            agent_project, sample.object_id
        )
        viewer.layers.clear()
        layers.clear()
        shown = display_stretch(
            image.data,
            str(stretch_combo.currentData()),
            float(low_percentile.value()),
            float(high_percentile.value()),
        )
        layers["vis"] = viewer.add_image(
            _native_to_display(shown),
            name=f"{band} display: {stretch_combo.currentData()}",
            colormap="gray",
            contrast_limits=(0.0, 1.0),
        )
        try:
            layers["vis"].interpolation2d = "nearest"
        except (AttributeError, ValueError):
            pass
        matched_rgb = _rgb_for_sample(sample, image)
        if matched_rgb is not None:
            rgb, mapping, correlation = matched_rgb
            layers["rgb"] = viewer.add_image(
                _native_to_display(rgb),
                name=f"RGB context ({mapping}, r={correlation:.2f})",
                rgb=True,
                visible=False,
            )
        layers["foreground"] = viewer.add_labels(
            _native_to_display(saved["foreground"]).astype(np.uint8),
            name="Foreground reference (optional)",
            opacity=0.42,
        )
        layers["foreground"].selected_label = 1
        layers["foreground"].brush_size = 5
        layers["structures"] = viewer.add_labels(
            _native_to_display(saved["structures"]).astype(np.int16),
            name="Structures S1/S2/...",
            opacity=0.48,
        )
        layers["structures"].selected_label = int(label_spin.value())
        layers["structures"].brush_size = 3
        if pair_manifest_by_object:
            empty_pair = np.zeros_like(layers["structures"].data, dtype=np.uint8)
            try:
                layers["pair_first"] = viewer.add_labels(
                    empty_pair.copy(),
                    name="当前像对A（青色）",
                    opacity=0.72,
                    color={1: "cyan"},
                )
                layers["pair_second"] = viewer.add_labels(
                    empty_pair.copy(),
                    name="当前像对B（洋红）",
                    opacity=0.72,
                    color={1: "magenta"},
                )
            except TypeError:
                layers["pair_first"] = viewer.add_labels(
                    empty_pair.copy(), name="当前像对A（青色）", opacity=0.72
                )
                layers["pair_second"] = viewer.add_labels(
                    empty_pair.copy(), name="当前像对B（洋红）", opacity=0.72
                )
        combo_set_data(sample_status, saved["sample_status"])
        sample_note.setPlainText(saved["sample_note"])
        boundary = saved["geometry_boundary_review"]
        combo_set_data(vis_grade, boundary.get("vis_morphology_grade", "unreviewed"))
        combo_set_data(all_imaging_grade, boundary.get("all_imaging_grade", "unreviewed"))
        selected_features = set(boundary.get("geometry_features", []))
        selected_reasons = set(boundary.get("non_A_reasons", []))
        for key, check in feature_checks.items():
            check.setChecked(key in selected_features)
        for key, check in reason_checks.items():
            check.setChecked(key in selected_reasons)
        boundary_note.setPlainText(boundary.get("note", ""))
        context = state["agent_context"]
        if context:
            agent_context_label.setText(
                "当前 Agent：等级={agent_grade}；clear_system={clear_system}；"
                "颜色={colour_evidence_state}；颜色限制A={colour_caps_A}；"
                "支持构型={supported}".format(
                    **context,
                    supported=", ".join(context.get("supported_configurations", [])) or "无",
                )
            )
        else:
            agent_context_label.setText("当前 Agent 结果：未找到；仍可完成专家边界复核")
        pair_selector.clear()
        manifest_item = pair_manifest_by_object.get(sample.object_id, {})
        for item in manifest_item.get("pairs", []):
            title = (
                f"S{item['first_label']}–S{item['second_label']}  "
                f"Δ={item.get('maximum_departure_mag', '未知')} mag"
            )
            pair_selector.addItem(title, item)
        if pair_selector.count() and pair_manifest_by_object:
            for pair_index in range(pair_selector.count()):
                item = pair_selector.itemData(pair_index)
                key = pair_id(item["first_label"], item["second_label"])
                review = state["pair_reviews"].get(key, {})
                if review.get("geometry_relation", "unreviewed") == "unreviewed":
                    pair_selector.setCurrentIndex(pair_index)
                    break
        progress.setText(
            f"{state['index'] + 1}/{len(samples)}  [{sample.cohort}]  {sample.object_id}"
            + (
                f"  | 像对 {pair_selector.currentIndex() + 1}/{pair_selector.count()}"
                if pair_selector.count()
                else ""
            )
        )
        viewer.title = f"Colour structures — [{sample.cohort}] {sample.object_id}"
        load_structure_form(int(label_spin.value()))
        load_pair_form()
        viewer.layers.selection.active = layers["structures"]
        viewer.reset_view()
        message.setText("已加载；尚未保存本次更改。")
        state["loading"] = False

    def previous() -> None:
        save_current()
        state["index"] = max(0, state["index"] - 1)
        load_current()

    def save_next() -> None:
        save_current()
        if pair_manifest_by_object and pair_selector.currentIndex() + 1 < pair_selector.count():
            pair_selector.setCurrentIndex(pair_selector.currentIndex() + 1)
            progress.setText(
                f"{state['index'] + 1}/{len(samples)}  [{state['sample'].cohort}]  "
                f"{state['sample'].object_id}  | 像对 "
                f"{pair_selector.currentIndex() + 1}/{pair_selector.count()}"
            )
            message.setText("上一像对已保存；已切换到本样本的下一像对。")
            return
        if state["index"] + 1 < len(samples):
            state["index"] += 1
            load_current()
        else:
            message.setText("全部样本已到末尾；最后一个已保存。")

    label_spin.valueChanged.connect(load_structure_form)
    pair_selector.currentIndexChanged.connect(load_pair_form)
    apply_stretch.clicked.connect(update_display_stretch)
    save_structure.clicked.connect(persist_structure_form)
    save_pair.clicked.connect(persist_pair_form)
    previous_button.clicked.connect(previous)
    save_button.clicked.connect(save_current)
    next_button.clicked.connect(save_next)
    dock = viewer.window.add_dock_widget(
        panel, area="right", name="结构标注与备注"
    )
    # This panel owns sample navigation and annotation metadata.  Accidentally
    # closing it leaves only the enlarged image canvas and makes the workflow
    # appear broken, so keep it dockable/movable but remove the close button.
    try:
        closable = QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        dock.setFeatures(dock.features() & ~closable)
    except (AttributeError, TypeError):
        pass
    load_current()
    napari.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset root containing cohort/object directories")
    parser.add_argument("--output", default="outputs_colour_structure_annotations")
    parser.add_argument("--band", default="VIS")
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--start-object")
    parser.add_argument(
        "--pair-review-manifest",
        help="JSON manifest limiting the UI to unresolved colour-pair review targets",
    )
    parser.add_argument(
        "--agent-project",
        default="lensagent_ui_outputs",
        help="Read-only LensAgent UI project used to display current grade/configuration context",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="List discovered sample counts without opening the UI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = discover_samples(args.dataset)
    if args.inventory_only:
        counts: dict[str, int] = {}
        for sample in samples:
            counts[sample.cohort] = counts.get(sample.cohort, 0) + 1
        print(json.dumps({"sample_count": len(samples), "cohorts": counts}, ensure_ascii=False))
        return 0
    run_review_ui(
        args.dataset,
        args.output,
        band=args.band,
        theme=args.theme,
        start_object=args.start_object,
        pair_review_manifest=args.pair_review_manifest,
        agent_project=args.agent_project,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
