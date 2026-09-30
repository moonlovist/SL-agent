"""Build reviewable arc/foreground mask proposals from annotated DOCX images.

The generated masks are proposals only.  They are written to a separate
calibration directory and never replace the independent Ci masks or enter the
production agent until a human has inspected the overlays.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    label,
)
from skimage.draw import polygon

from independent_colour_measurement import radial_model


PROPOSALS: dict[str, dict[str, Any]] = {
    "102160056_2663488981675470136": {
        "media": "image1.png",
        "mapping": "full_frame_stretched",
        "red_role": "arc",
        "blue_role": "foreground_group",
        "note": "Two red arc regions and one blue foreground-group region.",
        "review_decision": "discarded_low_snr_user_20260921",
    },
    "102160335_2682061397679348286": {
        "media": "image2.png",
        "mapping": "central_crop_anchor",
        "annotation_center_xy": [169.0, 181.0],
        "annotation_pixels_per_fits_pixel": [8.2, 8.2],
        "red_role": "arc",
        "note": "Red crescent is mapped from a zoomed central diagnostic; confirm its extent carefully.",
        "review_decision": "accepted_user_20260921",
    },
    "102160337_2703837081680486960": {
        "media": "image4.png",
        "mapping": "full_frame_stretched",
        "red_role": "arc",
        "note": "Red arc region on the full-frame RGB image.",
        "agent_structure_extension": "S1MASK",
        "review_decision": "accepted_user_20260921",
    },
    "102160338_2720592793680584236": {
        "media": "image5.png",
        "mapping": "central_crop_anchor",
        "annotation_center_xy": [350.0, 250.0],
        "annotation_pixels_per_fits_pixel": [6.2, 6.2],
        "red_role": "arc",
        "blue_role": "foreground",
        "note": "Two red arc regions and one blue foreground region in a zoomed central diagnostic.",
        "review_decision": "accepted_user_20260921",
    },
}


def normalized_review_status(proposal: dict[str, Any]) -> str:
    decision = str(proposal.get("review_decision", ""))
    if decision.startswith("accepted_"):
        return "accepted"
    if decision.startswith("discarded_"):
        return "discarded"
    return "requires_human_confirmation"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def colour_line(rgb: np.ndarray, colour: str) -> np.ndarray:
    r, g, b = (rgb[..., index].astype(float) for index in range(3))
    if colour == "red":
        return (r >= 150) & (r - g >= 60) & (r - b >= 45)
    if colour == "blue":
        return (b >= 120) & (b - r >= 55) & (b - g >= 20)
    raise ValueError(colour)


def enclosed_regions(line_mask: np.ndarray, min_line_pixels: int = 12) -> np.ndarray:
    """Fill hand-drawn closed outlines while rejecting tiny coloured pixels."""
    cleaned = binary_closing(binary_dilation(line_mask, iterations=1), iterations=2)
    labels, count = label(cleaned)
    output = np.zeros_like(cleaned, dtype=bool)
    for index in range(1, count + 1):
        component = labels == index
        if int(component.sum()) < min_line_pixels:
            continue
        filled = binary_fill_holes(binary_dilation(component, iterations=2))
        if int(filled.sum()) > int(component.sum()) * 1.35:
            output |= binary_erosion(filled, iterations=1)
    return output


def add_group_upper_arc(mask: np.ndarray) -> np.ndarray:
    """Close the deliberately open upper-arc annotation in image1."""
    vertices = np.asarray(
        [
            (254, 38), (276, 24), (306, 23), (338, 34), (371, 54),
            (405, 84), (431, 118), (451, 160), (460, 203),
            (400, 222), (399, 190), (389, 157), (370, 127),
            (345, 103), (318, 84), (291, 70), (258, 68),
        ],
        dtype=float,
    )
    rr, cc = polygon(vertices[:, 1], vertices[:, 0], shape=mask.shape)
    output = mask.copy()
    output[rr, cc] = True
    return output


def map_annotation_to_fits(
    annotation_mask: np.ndarray,
    fits_shape: tuple[int, int],
    center_xy: tuple[float, float],
    proposal: dict[str, Any],
) -> np.ndarray:
    height, width = annotation_mask.shape
    fits_height, fits_width = fits_shape
    yy, xx = np.indices(fits_shape)
    if proposal["mapping"] == "full_frame_stretched":
        annotation_x = xx * (width - 1) / max(fits_width - 1, 1)
        annotation_y = (fits_height - 1 - yy) * (height - 1) / max(fits_height - 1, 1)
    else:
        annotation_center_x, annotation_center_y = proposal["annotation_center_xy"]
        scale_x, scale_y = proposal["annotation_pixels_per_fits_pixel"]
        annotation_x = annotation_center_x + (xx - center_xy[0]) * scale_x
        annotation_y = annotation_center_y - (yy - center_xy[1]) * scale_y
    sample_x = np.rint(annotation_x).astype(int)
    sample_y = np.rint(annotation_y).astype(int)
    valid = (
        (sample_x >= 0)
        & (sample_x < width)
        & (sample_y >= 0)
        & (sample_y < height)
    )
    output = np.zeros(fits_shape, dtype=bool)
    output[valid] = annotation_mask[sample_y[valid], sample_x[valid]]
    return output


def load_current_masks(mask_dir: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    masks = {}
    for path in sorted(mask_dir.glob("C*_mask.fits")):
        mask = np.asarray(fits.getdata(path), dtype=bool)
        if mask.shape == shape:
            masks[path.stem.removesuffix("_mask")] = mask
    return masks


def robust_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 1.0
    median = np.median(finite)
    value = 1.4826 * np.median(np.abs(finite - median))
    return float(value) if value > 0 else float(np.std(finite) or 1.0)


def signal_mask_inside_reviewed_roi(
    vis: np.ndarray,
    roi: np.ndarray,
    foreground_roi: np.ndarray | None,
    center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover positive residual signal inside a human-defined search ROI.

    The ROI states where the reviewed structure is allowed to be.  It is not
    itself a photometric aperture.  A protected radial model is estimated
    without using the ROI, then a deliberately permissive smoothed threshold
    recovers connected signal.  No fallback forces noise pixels into a mask.
    """
    protected = roi.copy()
    if foreground_roi is not None:
        protected |= foreground_roi
    model = radial_model(vis, center_xy, binary_dilation(protected, iterations=2))
    residual = vis - model
    smooth = gaussian_filter(np.where(np.isfinite(residual), residual, 0.0), 1.0)
    control = ~binary_dilation(protected, iterations=3) & np.isfinite(smooth)
    yy, xx = np.indices(vis.shape)
    radius = np.hypot(xx - center_xy[0], yy - center_xy[1])
    local_control = control & (radius <= max(55.0, float(np.max(radius[roi])) + 15.0))
    if int(local_control.sum()) < 200:
        local_control = control
    smooth_sigma = robust_sigma(smooth[local_control])
    raw_sigma = robust_sigma(residual[local_control])
    seeds = roi & (smooth >= 1.15 * smooth_sigma) & (residual >= -0.25 * raw_sigma)
    if foreground_roi is not None:
        seeds &= ~binary_dilation(foreground_roi, iterations=1)
    labels, count = label(seeds)
    kept = np.zeros_like(roi, dtype=bool)
    component_count = 0
    for index in range(1, count + 1):
        component = labels == index
        if int(component.sum()) < 2:
            continue
        component_count += 1
        kept |= component
    grown = binary_dilation(kept, iterations=1) & roi
    grown &= residual >= -0.50 * raw_sigma
    if foreground_roi is not None:
        grown &= ~foreground_roi
    diagnostics = {
        "method": "human_ROI_then_protected_radial_residual_detection",
        "smoothed_seed_threshold_sigma": 1.15,
        "raw_seed_floor_sigma": -0.25,
        "grown_raw_floor_sigma": -0.50,
        "minimum_seed_component_pixels": 2,
        "smooth_control_sigma": smooth_sigma,
        "raw_control_sigma": raw_sigma,
        "seed_component_count": component_count,
        "seed_pixel_count": int(kept.sum()),
        "grown_signal_pixel_count": int(grown.sum()),
    }
    return grown, residual, diagnostics


def save_overlay(
    output_path: Path,
    vis: np.ndarray,
    current_masks: dict[str, np.ndarray],
    arc_roi: np.ndarray,
    arc_signal: np.ndarray,
    foreground: np.ndarray | None,
    residual: np.ndarray,
    center_xy: tuple[float, float],
    object_id: str,
    annotation_rgb: np.ndarray,
    proposal: dict[str, Any],
) -> None:
    sigma = robust_sigma(vis)
    display = np.arcsinh(np.maximum(vis - np.nanmedian(vis), 0.0) / max(sigma, 1e-12))
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    axes[0].imshow(annotation_rgb, origin="upper")
    axes[0].set_title("Human annotation source")
    axes[0].axis("off")

    axes[1].imshow(display, cmap="gray", origin="lower")
    for name, mask in current_masks.items():
        if name == "C0":
            continue
        if np.any(mask):
            axes[1].contour(mask, levels=[0.5], colors="cyan", linewidths=0.7, alpha=0.8)
    axes[1].plot(center_xy[0], center_xy[1], marker="+", color="yellow", markersize=10)
    axes[1].set_title("Existing Ci masks (cyan)")

    axes[2].imshow(display, cmap="gray", origin="lower")
    if np.any(arc_roi):
        axes[2].contour(arc_roi, levels=[0.5], colors="red", linewidths=1.0, linestyles="dashed")
    if np.any(arc_signal):
        axes[2].contour(arc_signal, levels=[0.5], colors="magenta", linewidths=1.5)
    if foreground is not None and np.any(foreground):
        axes[2].contour(foreground, levels=[0.5], colors="dodgerblue", linewidths=1.4)
    axes[2].plot(center_xy[0], center_xy[1], marker="+", color="yellow", markersize=10)
    axes[2].set_title("ROI (dashed) / signal mask (magenta)")

    residual_sigma = robust_sigma(residual)
    residual_display = np.arcsinh(residual / max(residual_sigma, 1e-12))
    limit = float(np.nanpercentile(np.abs(residual_display), 99.0))
    axes[3].imshow(residual_display, cmap="gray", origin="lower", vmin=-limit, vmax=limit)
    if np.any(arc_signal):
        axes[3].contour(arc_signal, levels=[0.5], colors="magenta", linewidths=1.5)
    if np.any(arc_roi):
        axes[3].contour(arc_roi, levels=[0.5], colors="red", linewidths=0.8, linestyles="dashed")
    axes[3].set_title("Foreground residual + signal mask")

    arc_y, arc_x = np.where(arc_roi)
    if foreground is not None and np.any(foreground):
        fg_y, fg_x = np.where(foreground)
        all_x = np.concatenate([arc_x, fg_x]) if arc_x.size else fg_x
        all_y = np.concatenate([arc_y, fg_y]) if arc_y.size else fg_y
    else:
        all_x, all_y = arc_x, arc_y
    if all_x.size:
        margin = 12
        x0, x1 = max(0, all_x.min() - margin), min(vis.shape[1] - 1, all_x.max() + margin)
        y0, y1 = max(0, all_y.min() - margin), min(vis.shape[0] - 1, all_y.max() + margin)
        for axis in axes[1:]:
            axis.set_xlim(x0, x1)
            axis.set_ylim(y0, y1)
    for axis in axes[1:]:
        axis.set_xlabel("FITS x")
        axis.set_ylabel("FITS y")
    fig.suptitle(
        f"{object_id} — reviewed mask: {normalized_review_status(proposal)}; "
        f"{proposal['mapping']}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--agent-refinement-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    objects = {}
    import csv

    with (args.measurements / "objects.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            objects[row["object_id"]] = row
    manifest: dict[str, Any] = {
        "schema": "reviewed_manual_mask_proposals_v2",
        "source_docx": str(args.docx.resolve()),
        "source_docx_sha256": hashlib.sha256(args.docx.read_bytes()).hexdigest(),
        "status": "human_review_recorded",
        "objects": [],
    }
    with zipfile.ZipFile(args.docx) as archive:
        for object_id, proposal in PROPOSALS.items():
            image_bytes = archive.read(f"word/media/{proposal['media']}")
            annotation_rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            red = enclosed_regions(colour_line(annotation_rgb, "red"))
            if object_id == "102160056_2663488981675470136":
                red = add_group_upper_arc(red)
            blue_line = colour_line(annotation_rgb, "blue")
            blue = enclosed_regions(blue_line) if np.any(blue_line) else None

            fits_path = Path(objects[object_id]["fits_path"])
            with fits.open(fits_path, memmap=False) as hdus:
                vis = np.asarray(hdus["VIS_FLUX"].data, dtype=float)
                vis_header = hdus["VIS_FLUX"].header.copy()
            diagnostics_path = args.measurements / "masks" / object_id / "diagnostics.json"
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            center_xy = tuple(float(value) for value in diagnostics["center_xy"])
            arc_roi = map_annotation_to_fits(red, vis.shape, center_xy, proposal)
            foreground_mask = (
                map_annotation_to_fits(blue, vis.shape, center_xy, proposal)
                if blue is not None and np.any(blue)
                else None
            )
            if foreground_mask is not None:
                arc_roi &= ~foreground_mask
            arc_mask, residual, signal_diagnostics = signal_mask_inside_reviewed_roi(
                vis, arc_roi, foreground_mask, center_xy
            )
            agent_refinement = None
            extension = proposal.get("agent_structure_extension")
            if extension and args.agent_refinement_root is not None:
                segmentation_path = (
                    args.agent_refinement_root / object_id / "vision_segmentation.fits"
                )
                if segmentation_path.exists():
                    with fits.open(segmentation_path, memmap=False) as segmentation:
                        agent_mask = np.asarray(segmentation[extension].data, dtype=bool)
                    if agent_mask.shape != vis.shape:
                        raise ValueError(
                            f"{object_id}: agent mask shape {agent_mask.shape} != {vis.shape}"
                        )
                    agent_mask &= arc_roi
                    arc_mask = agent_mask
                    agent_refinement = {
                        "segmentation_path": str(segmentation_path.resolve()),
                        "extension": extension,
                        "pixel_count_before_roi_intersection": int(
                            np.asarray(fits.getdata(segmentation_path, extname=extension), dtype=bool).sum()
                        ),
                        "pixel_count_after_roi_intersection": int(agent_mask.sum()),
                        "method": "LensAgent promoted SEP/geometric structure intersected with reviewed ROI",
                    }
            object_output = args.output / object_id
            object_output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(annotation_rgb).save(object_output / "annotation_source.png")
            arc_header = vis_header.copy()
            arc_header["MASKROLE"] = "ARC_SIGNAL_PROPOSAL"
            review_status = normalized_review_status(proposal)
            arc_header["MASKSTAT"] = review_status.upper()
            arc_header["ANNMEDIA"] = proposal["media"]
            fits.PrimaryHDU(arc_mask.astype(np.uint8), header=arc_header).writeto(
                object_output / "arc_mask_proposal.fits", overwrite=True
            )
            roi_header = vis_header.copy()
            roi_header["MASKROLE"] = "ARC_SEARCH_ROI"
            roi_header["MASKSTAT"] = review_status.upper()
            roi_header["ANNMEDIA"] = proposal["media"]
            fits.PrimaryHDU(arc_roi.astype(np.uint8), header=roi_header).writeto(
                object_output / "arc_search_roi_proposal.fits", overwrite=True
            )
            if foreground_mask is not None:
                foreground_header = vis_header.copy()
                foreground_header["MASKROLE"] = "FOREGROUND_PROPOSAL"
                foreground_header["MASKSTAT"] = review_status.upper()
                foreground_header["ANNMEDIA"] = proposal["media"]
                fits.PrimaryHDU(foreground_mask.astype(np.uint8), header=foreground_header).writeto(
                    object_output / "foreground_mask_proposal.fits", overwrite=True
                )
            current_masks = load_current_masks(args.measurements / "masks" / object_id, vis.shape)
            save_overlay(
                object_output / "proposal_overlay.png",
                vis,
                current_masks,
                arc_roi,
                arc_mask,
                foreground_mask,
                residual,
                center_xy,
                object_id,
                annotation_rgb,
                proposal,
            )
            record = {
                "object_id": object_id,
                **proposal,
                "fits_path": str(fits_path),
                "fits_shape": list(vis.shape),
                "center_xy": list(center_xy),
                "annotation_shape": list(annotation_rgb.shape[:2]),
                "annotation_sha256": sha256_bytes(image_bytes),
                "arc_search_roi_pixel_count": int(arc_roi.sum()),
                "arc_search_roi_sha256": hashlib.sha256(arc_roi.astype(np.uint8).tobytes()).hexdigest(),
                "arc_pixel_count": int(arc_mask.sum()),
                "arc_mask_sha256": hashlib.sha256(arc_mask.astype(np.uint8).tobytes()).hexdigest(),
                "foreground_pixel_count": int(foreground_mask.sum()) if foreground_mask is not None else 0,
                "foreground_mask_sha256": (
                    hashlib.sha256(foreground_mask.astype(np.uint8).tobytes()).hexdigest()
                    if foreground_mask is not None
                    else None
                ),
                "status": review_status,
                "signal_detection": signal_diagnostics,
                "agent_refinement": agent_refinement,
            }
            (object_output / "proposal_metadata.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest["objects"].append(record)
    (args.output / "proposal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"proposal_count": len(manifest["objects"]), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
