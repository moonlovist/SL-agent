"""Independent four-band colour measurements for Euclid-like FITS cutouts.

This script deliberately does not import or call the Lens Expert Agent.  It
implements the documented measurement protocol directly: deterministic VIS
residual masks, common-PSF aperture photometry, continuous-valued colour and
SED comparisons, and aperture/foreground sensitivity records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, label
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve


BANDS = ("VIS", "NIR_Y", "NIR_J", "NIR_H")
COLOURS = (("VIS", "NIR_Y", "I_Y"), ("NIR_Y", "NIR_J", "Y_J"),
           ("NIR_J", "NIR_H", "J_H"), ("VIS", "NIR_J", "I_J"),
           ("VIS", "NIR_H", "I_H"), ("NIR_Y", "NIR_H", "Y_H"))


def f(value: object) -> object:
    """Convert NumPy scalars/non-finite floats to portable CSV/JSON values."""
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def psf_fwhm(psf: np.ndarray) -> float | None:
    p = np.where(np.isfinite(psf), np.maximum(psf, 0.0), 0.0)
    total = p.sum()
    if total <= 0:
        return None
    p /= total
    yy, xx = np.indices(p.shape)
    mx, my = (p * xx).sum(), (p * yy).sum()
    variance = (p * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / 2.0
    return float(2.354820045 * np.sqrt(variance)) if variance > 0 else None


def gaussian_kernel(source_fwhm: float, target_fwhm: float) -> np.ndarray:
    sigma = np.sqrt(max(target_fwhm ** 2 - source_fwhm ** 2, 0.0)) / 2.354820045
    if sigma < 0.05:
        return np.ones((1, 1), float)
    half = max(2, int(np.ceil(4 * sigma)))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
    return kernel / kernel.sum()


def aperture_error(rms: np.ndarray, kernel: np.ndarray, aperture: np.ndarray) -> float:
    weights = fftconvolve(aperture.astype(float), kernel[::-1, ::-1], mode="same")
    return float(np.sqrt(np.sum((weights * rms) ** 2)))


def radial_model(data: np.ndarray, center: tuple[float, float], exclude: np.ndarray) -> np.ndarray:
    """Circular median profile, a declared contamination diagnostic not a fit."""
    yy, xx = np.indices(data.shape)
    radius = np.hypot(xx - center[0], yy - center[1])
    max_radius = min(70.0, float(radius.max()))
    bins = np.arange(0.0, max_radius + 1.0, 1.0)
    rs, values = [], []
    valid = np.isfinite(data) & ~exclude
    for low, high in zip(bins[:-1], bins[1:]):
        use = valid & (radius >= low) & (radius < high)
        if use.sum() >= 15:
            rs.append((low + high) / 2)
            values.append(float(np.median(data[use])))
    if len(rs) < 5:
        return np.full(data.shape, np.nanmedian(data), dtype=float)
    return np.interp(radius, rs, values, left=values[0], right=values[-1])


def component_masks(vis: np.ndarray, rms: np.ndarray) -> tuple[tuple[float, float], dict[str, np.ndarray], np.ndarray]:
    """Find C0 and residual Ci masks from VIS only, without lens interpretation."""
    h, w = vis.shape
    yy, xx = np.indices(vis.shape)
    nominal = (w / 2.0, h / 2.0)
    # These supplied cutouts are target-centred.  Restricting C0 selection to
    # the central 18 px prevents a brighter, off-target neighbour from being
    # silently relabelled as the central component.
    central = np.hypot(xx - nominal[0], yy - nominal[1]) <= 18.0
    smooth = gaussian_filter(np.where(np.isfinite(vis), vis, np.nanmedian(vis)), 1.2)
    peak_y, peak_x = np.unravel_index(np.argmax(np.where(central, smooth, -np.inf)), vis.shape)
    center = (float(peak_x), float(peak_y))
    radius = np.hypot(xx - center[0], yy - center[1])
    c0 = radius <= 4.0
    model0 = radial_model(vis, center, c0)
    residual0 = vis - model0
    noise = np.maximum(rms, robust_sigma(residual0[(radius >= 8) & (radius <= 55)]))
    first = (residual0 / noise >= 2.5) & (radius >= 5.0) & (radius <= 35.0)
    # Remove isolated single-pixel threshold crossings, then re-estimate profile.
    labels, count = label(first)
    keep = np.zeros_like(first)
    for idx in range(1, count + 1):
        item = labels == idx
        if item.sum() >= 3:
            keep |= item
    model = radial_model(vis, center, c0 | binary_dilation(keep, iterations=2))
    residual = vis - model
    threshold = (residual / noise >= 2.5) & (radius >= 5.0) & (radius <= 35.0)
    labels, count = label(threshold)
    detections: list[tuple[float, np.ndarray]] = []
    for idx in range(1, count + 1):
        item = labels == idx
        if item.sum() < 3:
            continue
        ys, xs = np.where(item)
        angle = float(np.mod(np.degrees(np.arctan2(ys.mean() - center[1], xs.mean() - center[0])), 360.0))
        detections.append((angle, item))
    masks = {"C0": c0}
    for idx, (_, item) in enumerate(sorted(detections, key=lambda v: v[0]), start=1):
        masks[f"C{idx}"] = item
    return center, masks, model


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()


def local_background(data: np.ndarray, mask: np.ndarray, exclusions: np.ndarray, which: str) -> tuple[float | None, float | None, int]:
    if which == "inner":
        ring = binary_dilation(mask, iterations=4) & ~binary_dilation(mask, iterations=1)
    else:
        ring = binary_dilation(mask, iterations=8) & ~binary_dilation(mask, iterations=4)
    use = ring & ~exclusions & np.isfinite(data)
    if use.sum() < 20:
        return None, None, int(use.sum())
    return float(np.median(data[use])), robust_sigma(data[use]), int(use.sum())


def record_photometry(data: np.ndarray, rms: np.ndarray, kernel: np.ndarray, mask: np.ndarray,
                      exclusions: np.ndarray) -> tuple[dict[str, float | int | None], dict[str, float | int | None]]:
    raw = float(np.sum(data[mask]))
    err_diag = aperture_error(rms, kernel, mask)
    inner_bg, inner_scatter, inner_n = local_background(data, mask, exclusions, "inner")
    outer_bg, outer_scatter, outer_n = local_background(data, mask, exclusions, "outer")
    bg = inner_bg if inner_bg is not None else outer_bg
    scatter = inner_scatter if inner_scatter is not None else outer_scatter
    bg_n = inner_n if inner_bg is not None else outer_n
    flux = raw - bg * mask.sum() if bg is not None else None
    bg_error = (mask.sum() * scatter / np.sqrt(bg_n)) if scatter is not None and bg_n else 0.0
    error = float(np.hypot(err_diag, bg_error)) if flux is not None else None
    snr = float(flux / error) if error and error > 0 and flux is not None else None
    reliable = bool(snr is not None and snr >= 3.0 and flux is not None and flux > 0)
    magnitude = float(25.0 - 2.5 * np.log10(flux)) if reliable else None
    magnitude_error = float(2.5 / np.log(10) / snr) if reliable else None
    upper_limit = float(3.0 * error) if error and error > 0 and not reliable else None
    primary = {
        "flux_raw": raw, "flux_local_bg_subtracted": flux, "flux_error": error,
        "snr": snr, "local_background": bg, "local_background_scatter": scatter,
        "local_background_pixel_count": bg_n, "negative_pixel_fraction": float(np.mean(data[mask] < 0)),
        "masked_pixel_fraction": float(np.mean(~np.isfinite(data[mask]))), "magnitude": magnitude,
        "magnitude_error": magnitude_error, "upper_limit": upper_limit,
    }
    alternate = {"local_background_outer": outer_bg, "local_background_outer_scatter": outer_scatter,
                 "local_background_outer_pixel_count": outer_n}
    return primary, alternate


def colours(phot: dict[str, dict[str, float | int | None]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for a, b, name in COLOURS:
        ma, mb = phot[a]["magnitude"], phot[b]["magnitude"]
        ea, eb = phot[a]["magnitude_error"], phot[b]["magnitude_error"]
        result[f"colour_{name}"] = float(ma - mb) if ma is not None and mb is not None else None
        result[f"colour_{name}_error"] = float(np.hypot(ea, eb)) if ea is not None and eb is not None else None
    return result


def sed_fit(one: dict[str, dict], two: dict[str, dict]) -> dict[str, float | int | None]:
    usable = [b for b in BANDS if one[b]["flux_local_bg_subtracted"] is not None and
              two[b]["flux_local_bg_subtracted"] is not None and one[b]["flux_error"] and two[b]["flux_error"]]
    if len(usable) < 2:
        return {"best_alpha": None, "same_sed_chi2": None, "same_sed_dof": None,
                "same_sed_reduced_chi2": None, "fit_band_count": len(usable),
                **{f"residual_{b}": None for b in BANDS}, **{f"residual_sigma_{b}": None for b in BANDS}}
    f1 = np.array([one[b]["flux_local_bg_subtracted"] for b in usable], float)
    f2 = np.array([two[b]["flux_local_bg_subtracted"] for b in usable], float)
    e1 = np.array([one[b]["flux_error"] for b in usable], float)
    e2 = np.array([two[b]["flux_error"] for b in usable], float)
    scale = max(1.0, np.nanmax(np.abs(f1)) / max(np.nanmax(np.abs(f2)), 1e-12))
    def chi2(alpha: float) -> float:
        return float(np.sum((f1 - alpha * f2) ** 2 / (e1 * e1 + alpha * alpha * e2 * e2)))
    optimum = minimize_scalar(chi2, bounds=(-10 * scale, 10 * scale), method="bounded")
    alpha, value = float(optimum.x), float(optimum.fun)
    output: dict[str, float | int | None] = {"best_alpha": alpha, "same_sed_chi2": value,
        "same_sed_dof": len(usable) - 1, "same_sed_reduced_chi2": value / (len(usable) - 1),
        "fit_band_count": len(usable)}
    for band in BANDS:
        if band not in usable:
            output[f"residual_{band}"] = output[f"residual_sigma_{band}"] = None
        else:
            i = usable.index(band)
            residual = float(f1[i] - alpha * f2[i])
            output[f"residual_{band}"] = residual
            output[f"residual_sigma_{band}"] = float(residual / np.sqrt(e1[i] ** 2 + alpha ** 2 * e2[i] ** 2))
    return output


def quality(component: str, phot: dict[str, dict], central_fraction: float, neighbour_fraction: float,
            wcs_ok: bool, edge: bool) -> str:
    reliable = sum(phot[b]["magnitude"] is not None for b in BANDS)
    if reliable < 2:
        return "INSUFFICIENT_BANDS"
    if edge:
        return "EDGE_AFFECTED"
    if not wcs_ok:
        return "WCS_UNCERTAIN"
    if component != "C0" and central_fraction > 0.5:
        return "CENTRAL_CONTAMINATED"
    if neighbour_fraction > 0.15:
        return "NEIGHBOUR_CONTAMINATED"
    if reliable < 4:
        return "LOW_SNR"
    return "HIGH_QUALITY"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f(value) for key, value in row.items()})


def analyse_one(cohort: str, folder: Path, output: Path) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    fits_path = next(folder.glob("*.fits"))
    object_id = folder.name
    object_dir = output / "masks" / object_id
    object_dir.mkdir(parents=True, exist_ok=True)
    with fits.open(fits_path, memmap=False) as hdus:
        flux, rms, psfs, headers = {}, {}, {}, {}
        for band in BANDS:
            flux[band] = np.asarray(hdus[f"{band}_FLUX"].data, float)
            rms[band] = np.asarray(hdus[f"{band}_RMS"].data, float)
            psfs[band] = np.asarray(hdus[f"{band}_PSF"].data, float)
            headers[band] = hdus[f"{band}_FLUX"].header
        shape = flux["VIS"].shape
        zps = {b: headers[b].get("MAGZERO") for b in BANDS}
        units = {b: headers[b].get("BUNIT") for b in BANDS}
        wcs = {b: WCS(headers[b]).celestial for b in BANDS}
    if any(zps[b] is None for b in BANDS):
        raise ValueError(f"{object_id}: missing MAGZERO")
    scaled_flux = {b: flux[b] * 10 ** (-0.4 * (float(zps[b]) - 25.0)) for b in BANDS}
    scaled_rms = {b: rms[b] * 10 ** (-0.4 * (float(zps[b]) - 25.0)) for b in BANDS}
    fwhm = {b: psf_fwhm(psfs[b]) for b in BANDS}
    if any(v is None for v in fwhm.values()):
        raise ValueError(f"{object_id}: invalid PSF")
    target = max(fwhm.values())
    kernels = {b: gaussian_kernel(fwhm[b], target) for b in BANDS}
    matched = {b: fftconvolve(scaled_flux[b], kernels[b], mode="same") for b in BANDS}
    center, masks, initial_model = component_masks(scaled_flux["VIS"], scaled_rms["VIS"])
    all_masks = np.logical_or.reduce(list(masks.values()))
    candidate_union = np.logical_or.reduce([mask for key, mask in masks.items() if key != "C0"]) if len(masks) > 1 else np.zeros(shape, bool)
    radial_models = {b: radial_model(matched[b], center, binary_dilation(candidate_union, iterations=2)) for b in BANDS}
    yy, xx = np.indices(shape)
    distance = np.hypot(xx - center[0], yy - center[1])
    c0_overlap_zone = distance <= 7.0
    finite = {b: np.isfinite(scaled_flux[b]) & np.isfinite(scaled_rms[b]) & (scaled_rms[b] > 0) for b in BANDS}
    wcs_ok = all(wcs[b].has_celestial for b in BANDS)
    grid_y, grid_x = np.mgrid[0:shape[0]:complex(0, 5), 0:shape[1]:complex(0, 5)]
    offsets: dict[str, float | None] = {}
    for band in BANDS:
        try:
            bx, by = wcs[band].world_to_pixel_values(*wcs["VIS"].pixel_to_world_values(grid_x, grid_y))
            offsets[band] = float(np.nanmax(np.hypot(bx - grid_x, by - grid_y)))
        except Exception:
            offsets[band] = None
            wcs_ok = False
    pixel_scale = abs(headers["VIS"].get("CDELT1", np.nan)) * 3600.0
    if not np.isfinite(pixel_scale):
        pixel_scale = None
    component_rows, component_phot = [], {}
    diagnostics = {"object_id": object_id, "cohort": cohort, "fits_path": str(fits_path), "center_xy": center,
        "mask_method": "VIS circular-radial-profile residual threshold at 2.5 sigma, connected components >=3 pixels",
        "target_psf_fwhm_px": target, "psf_fwhm_px": fwhm, "wcs_grid_max_offset_px": offsets, "components": {}}
    for cid, base_mask in masks.items():
        fits.PrimaryHDU(base_mask.astype(np.uint8), header=fits.Header({"COMPID": cid, "MSOURCE": "VIS_RESIDUAL_THRESHOLD" if cid != "C0" else "CENTRAL_FIXED_CIRCLE"})).writeto(object_dir / f"{cid}_mask.fits", overwrite=True)
        ys, xs = np.where(base_mask)
        centroid = (float(xs.mean()), float(ys.mean()))
        radius = float(np.hypot(centroid[0] - center[0], centroid[1] - center[1]))
        angle = float(np.mod(np.degrees(np.arctan2(centroid[1] - center[1], centroid[0] - center[0])), 360.0))
        # A proximity proxy must refer to another saved component; blank sky in
        # an aperture's surrounding ring is not neighbour contamination.
        other_components = all_masks & ~base_mask
        neighbour = binary_dilation(base_mask, iterations=3) & other_components & ~c0_overlap_zone
        neighbour_fraction = float(neighbour.sum() / max(1, binary_dilation(base_mask, iterations=3).sum()))
        central_fraction = float(np.mean(c0_overlap_zone[base_mask]))
        edge = bool(np.any((xs < 2) | (xs >= shape[1]-2) | (ys < 2) | (ys >= shape[0]-2)))
        exclusions = binary_dilation(all_masks, iterations=2) & ~base_mask
        phot, alternate = {}, {}
        for band in BANDS:
            p, a = record_photometry(matched[band], scaled_rms[band], kernels[band], base_mask, exclusions)
            phot[band], alternate[band] = p, a
        component_phot[cid] = phot
        c = colours(phot)
        norm = sum(max(float(phot[b]["flux_local_bg_subtracted"] or 0.0), 0.0) for b in BANDS)
        row: dict[str, object] = {"cohort": cohort, "object_id": object_id, "component_id": cid,
            "component_type": "central" if cid == "C0" else "candidate_residual", "mask_path": str((object_dir / f"{cid}_mask.fits").resolve()),
            "mask_sha256": mask_hash(base_mask), "mask_source": "central_fixed_circle" if cid == "C0" else "VIS_residual_threshold_segmentation",
            "centroid_x_px": centroid[0], "centroid_y_px": centroid[1], "pixel_count": int(base_mask.sum()),
            "radius_from_C0_px": radius, "radius_from_C0_arcsec": radius * pixel_scale if pixel_scale else None,
            "position_angle_deg": angle, "C0_overlap_fraction": central_fraction, "neighbour_proximity_fraction": neighbour_fraction,
            "edge_affected": edge, "colour_quality": quality(cid, phot, central_fraction, neighbour_fraction, wcs_ok, edge),
            "usable_colour_band_count": sum(phot[b]["magnitude"] is not None for b in BANDS), **c}
        for band in BANDS:
            for key, value in phot[band].items(): row[f"{band}_{key}"] = value
            row[f"{band}_normalised_flux"] = float(phot[band]["flux_local_bg_subtracted"] / norm) if norm > 0 and phot[band]["flux_local_bg_subtracted"] is not None else None
            model_flux = float(np.sum(radial_models[band][base_mask]))
            row[f"{band}_radial_central_model_flux"] = model_flux
            row[f"{band}_central_model_fraction_of_raw"] = float(model_flux / phot[band]["flux_raw"]) if phot[band]["flux_raw"] not in (None, 0) else None
        component_rows.append(row)
        diagnostics["components"][cid] = {"mask_sha256": row["mask_sha256"], "photometry": phot, "alternate_background": alternate, "colours": c}
    # Pair and candidate-centre continuous comparisons.
    pairs, centres = [], []
    for first, second in combinations([x for x in masks if x != "C0"], 2):
        fit = sed_fit(component_phot[first], component_phot[second])
        overlap = float((masks[first] & masks[second]).sum() / max(1, min(masks[first].sum(), masks[second].sum())))
        row = {"cohort": cohort, "object_id": object_id, "component_1": first, "component_2": second,
               "aperture_overlap_fraction": overlap, "same_geometric_hypothesis": None, "possible_arc_fragment_pair": None,
               "measurement_quality": f"{next(x['colour_quality'] for x in component_rows if x['component_id']==first)}|{next(x['colour_quality'] for x in component_rows if x['component_id']==second)}", **fit}
        for band in BANDS:
            a, b = component_phot[first][band], component_phot[second][band]
            row[f"flux_ratio_{band}"] = float(a["flux_local_bg_subtracted"] / b["flux_local_bg_subtracted"]) if b["flux_local_bg_subtracted"] not in (None, 0) and a["flux_local_bg_subtracted"] is not None else None
        c1, c2 = colours(component_phot[first]), colours(component_phot[second])
        max_delta, max_sigma = 0.0, 0.0
        for _, _, name in COLOURS:
            delta = float(c1[f"colour_{name}"] - c2[f"colour_{name}"]) if c1[f"colour_{name}"] is not None and c2[f"colour_{name}"] is not None else None
            error = float(np.hypot(c1[f"colour_{name}_error"], c2[f"colour_{name}_error"])) if c1[f"colour_{name}_error"] is not None and c2[f"colour_{name}_error"] is not None else None
            row[f"delta_colour_{name}"] = delta; row[f"delta_colour_{name}_error"] = error
            if delta is not None and error: max_delta=max(max_delta,abs(delta)); max_sigma=max(max_sigma,abs(delta/error))
        row["maximum_colour_departure_mag"] = max_delta if max_delta else None; row["maximum_colour_departure_sigma"] = max_sigma if max_sigma else None
        pairs.append(row)
    c0colours = colours(component_phot["C0"])
    for cid in [x for x in masks if x != "C0"]:
        fit = sed_fit(component_phot[cid], component_phot["C0"])
        candidate_colours = colours(component_phot[cid])
        row = {"cohort": cohort, "object_id": object_id, "candidate_id": cid, "centre_id": "C0", **fit}
        for _, _, name in COLOURS:
            delta = float(candidate_colours[f"colour_{name}"] - c0colours[f"colour_{name}"]) if candidate_colours[f"colour_{name}"] is not None and c0colours[f"colour_{name}"] is not None else None
            error = float(np.hypot(candidate_colours[f"colour_{name}_error"], c0colours[f"colour_{name}_error"])) if candidate_colours[f"colour_{name}_error"] is not None and c0colours[f"colour_{name}_error"] is not None else None
            row[f"delta_colour_{name}"] = delta; row[f"delta_colour_{name}_error"] = error; row[f"delta_colour_{name}_significance"] = float(delta/error) if delta is not None and error else None
        candidate_row = next(x for x in component_rows if x["component_id"] == cid)
        row.update({"candidate_C0_overlap_fraction": candidate_row["C0_overlap_fraction"], "radius_from_C0_px": candidate_row["radius_from_C0_px"], "central_contamination_proxy": candidate_row["VIS_central_model_fraction_of_raw"], "measurement_quality": candidate_row["colour_quality"]})
        centres.append(row)
    # Sensitivities: masks +/- 1/2 pixels, background choices, and central profile subtraction.
    robust = []
    all_candidate_mask = candidate_union
    wider_target = target * 1.10
    wider_kernels = {b: gaussian_kernel(fwhm[b], wider_target) for b in BANDS}
    wider_matched = {b: fftconvolve(scaled_flux[b], wider_kernels[b], mode="same") for b in BANDS}
    for cid, base in masks.items():
        variants = {"nominal": base, "erode_1px": binary_erosion(base, iterations=1), "dilate_1px": binary_dilation(base, iterations=1), "dilate_2px": binary_dilation(base, iterations=2)}
        nominal_colours = None
        for name, variant in variants.items():
            if variant.sum() < 3: continue
            exclusions = binary_dilation(all_masks, iterations=2) & ~variant
            phot = {b: record_photometry(matched[b], scaled_rms[b], kernels[b], variant, exclusions)[0] for b in BANDS}
            colour_values = colours(phot)
            if name == "nominal": nominal_colours = colour_values
            row = {"cohort": cohort, "object_id": object_id, "component_id": cid, "setting": name, "background_region": "inner_annulus", "foreground_subtracted": False, "psf_target_fwhm_px": target, "pixel_count": int(variant.sum())}
            for b in BANDS: row[f"{b}_flux_local_bg_subtracted"] = phot[b]["flux_local_bg_subtracted"]
            for _, _, cname in COLOURS:
                row[f"colour_{cname}"] = colour_values[f"colour_{cname}"]
                row[f"delta_from_nominal_{cname}"] = float(colour_values[f"colour_{cname}"] - nominal_colours[f"colour_{cname}"]) if nominal_colours and colour_values[f"colour_{cname}"] is not None and nominal_colours[f"colour_{cname}"] is not None else None
            robust.append(row)
        # Nominal mask with alternate background and radial foreground subtraction.
        exclusions = binary_dilation(all_masks, iterations=2) & ~base
        alternate_phot = {b: record_photometry(matched[b], scaled_rms[b], kernels[b], base, exclusions)[0] for b in BANDS}
        for b in BANDS:
            bg, sc, n = local_background(matched[b], base, exclusions, "outer")
            if bg is not None:
                alternate_phot[b]["flux_local_bg_subtracted"] = float(alternate_phot[b]["flux_raw"] - bg * base.sum())
        for setting, foreground_subtracted in (("outer_background", False), ("radial_C0_subtraction", True)):
            phot = alternate_phot if setting == "outer_background" else {}
            if foreground_subtracted:
                profile_exclude = binary_dilation(all_candidate_mask, iterations=2)
                for b in BANDS:
                    fg = radial_model(matched[b], center, profile_exclude)
                    residual = matched[b] - fg
                    p, _ = record_photometry(residual, scaled_rms[b], kernels[b], base, exclusions)
                    phot[b] = p
            cv = colours(phot)
            row={"cohort":cohort,"object_id":object_id,"component_id":cid,"setting":setting,"background_region":"outer_annulus" if setting=="outer_background" else "inner_annulus","foreground_subtracted":foreground_subtracted,"psf_target_fwhm_px":target,"pixel_count":int(base.sum())}
            for b in BANDS: row[f"{b}_flux_local_bg_subtracted"] = phot[b]["flux_local_bg_subtracted"]
            for _,_,cname in COLOURS:
                row[f"colour_{cname}"]=cv[f"colour_{cname}"]
                row[f"delta_from_nominal_{cname}"]=float(cv[f"colour_{cname}"]-nominal_colours[f"colour_{cname}"]) if nominal_colours and cv[f"colour_{cname}"] is not None and nominal_colours[f"colour_{cname}"] is not None else None
            robust.append(row)
        wide_phot = {b: record_photometry(wider_matched[b], scaled_rms[b], wider_kernels[b], base, exclusions)[0] for b in BANDS}
        wide_colours = colours(wide_phot)
        row={"cohort":cohort,"object_id":object_id,"component_id":cid,"setting":"psf_target_plus10pct","background_region":"inner_annulus","foreground_subtracted":False,"psf_target_fwhm_px":wider_target,"pixel_count":int(base.sum())}
        for b in BANDS: row[f"{b}_flux_local_bg_subtracted"] = wide_phot[b]["flux_local_bg_subtracted"]
        for _,_,cname in COLOURS:
            row[f"colour_{cname}"] = wide_colours[f"colour_{cname}"]
            row[f"delta_from_nominal_{cname}"] = float(wide_colours[f"colour_{cname}"]-nominal_colours[f"colour_{cname}"]) if nominal_colours and wide_colours[f"colour_{cname}"] is not None and nominal_colours[f"colour_{cname}"] is not None else None
        robust.append(row)
    # Summarise continuous sensitivity ranges without imposing a rating threshold.
    for row in component_rows:
        relevant = [x for x in robust if x["object_id"] == row["object_id"] and x["component_id"] == row["component_id"]]
        def maximum(settings: set[str]) -> float | None:
            values = [abs(float(item[key])) for item in relevant if item["setting"] in settings for key in item if key.startswith("delta_from_nominal_") and item[key] not in (None, "")]
            return max(values) if values else None
        row["aperture_sensitivity_max_colour_change_mag"] = maximum({"erode_1px", "dilate_1px", "dilate_2px"})
        row["local_background_sensitivity_max_colour_change_mag"] = maximum({"outer_background"})
        row["foreground_subtraction_sensitivity_max_colour_change_mag"] = maximum({"radial_C0_subtraction"})
        row["psf_matching_sensitivity_max_colour_change_mag"] = maximum({"psf_target_plus10pct"})
        row["stability_status"] = "UNASSESSED_NO_CALIBRATED_THRESHOLD"
    # Object row and a transparent diagnostic overlay.
    all_finite = all(np.all(finite[b]) for b in BANDS)
    object_row = {"cohort": cohort, "object_id": object_id, "fits_path": str(fits_path), "fits_sha256": hashlib.sha256(fits_path.read_bytes()).hexdigest(),
        "available_bands": ";".join(BANDS), "usable_band_count": len(BANDS), "candidate_structure_count": len(masks)-1,
        "measurable_candidate_pair_count": len(pairs), "overall_colour_quality": "HIGH_QUALITY" if all(x["colour_quality"]=="HIGH_QUALITY" for x in component_rows) else "MIXED_OR_LIMITED",
        "primary_failure_reason": "none" if len(masks)>1 else "no_VIS_residual_candidate_above_documented_threshold",
        "image_shape": f"{shape[1]}x{shape[0]}", "center_x_px": center[0], "center_y_px": center[1], "all_pixels_finite": all_finite,
        "wcs_all_valid": wcs_ok, "pixel_scale_arcsec": pixel_scale, "common_psf_target_fwhm_px": target,
        "psf_matching_method": "Gaussian kernel from positive-PSF second-moment FWHM; no deconvolution",
        "saturation_detected": False, "edge_truncation_detected": any(r["edge_affected"] for r in component_rows)}
    for b in BANDS:
        object_row[f"{b}_magzero"] = zps[b]; object_row[f"{b}_bunit"] = units[b]; object_row[f"{b}_psf_fwhm_px"] = fwhm[b]; object_row[f"{b}_wcs_max_offset_px"] = offsets[b]
        object_row[f"{b}_psf_flux_ratio_after_before"] = float(matched[b][20:-20,20:-20].sum() / scaled_flux[b][20:-20,20:-20].sum()) if scaled_flux[b][20:-20,20:-20].sum() != 0 else None
    diagnostics["object_row"] = object_row
    (object_dir / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=f), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5,5)); scale=robust_sigma(scaled_flux["VIS"]); ax.imshow(np.arcsinh(np.maximum(scaled_flux["VIS"],0)/max(scale,1e-9)), cmap="gray", origin="lower")
    for cid, mask in masks.items():
        ax.contour(mask, levels=[0.5], linewidths=1.0); ys,xs=np.where(mask); ax.text(xs.mean(),ys.mean(),cid,color="cyan",fontsize=8)
    ax.set_title(object_id); ax.set_xlim(center[0]-45,center[0]+45); ax.set_ylim(center[1]-45,center[1]+45); fig.tight_layout(); fig.savefig(object_dir/"VIS_mask_overlay.png",dpi=160); plt.close(fig)
    return [object_row], component_rows, pairs, centres, robust


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = ([], [], [], [], [])
    selection = {}
    for cohort in ("A", "B"):
        folders = sorted(p for p in (args.data_root / cohort).iterdir() if p.is_dir())[-10:]
        selection[cohort] = [p.name for p in folders]
        for folder in folders:
            result = analyse_one(cohort, folder, args.output)
            for target, source in zip(rows, result): target.extend(source)
    for name, data in zip(("objects.csv", "components.csv", "component_pairs.csv", "candidate_centre.csv", "robustness.csv"), rows):
        write_csv(args.output / name, data)
    (args.output / "selection_and_method.json").write_text(json.dumps({"selection": selection, "method": "Independent script; no Lens Expert Agent code or outputs used.", "limitations": ["Masks are deterministic VIS residual threshold segmentations, not human annotations.", "Flux errors propagate supplied RMS through the applied kernel and local-background uncertainty, but do not include unprovided input covariance or PSF-model uncertainty.", "No Q1 grade, expert score, or colour-consistency decision threshold is calculated."]}, ensure_ascii=False, indent=2), encoding="utf-8")
    # A compact all-object visual index makes the saved measurement regions
    # inspectable without changing any of the photometric products.
    figure, axes = plt.subplots(4, 5, figsize=(15, 12))
    for axis, row in zip(axes.ravel(), rows[0]):
        image = plt.imread(args.output / "masks" / row["object_id"] / "VIS_mask_overlay.png")
        axis.imshow(image)
        axis.set_title(f"{row['cohort']} {row['object_id']}", fontsize=7)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "mask_overview.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"objects":len(rows[0]),"components":len(rows[1]),"pairs":len(rows[2]),"candidate_centre":len(rows[3]),"robustness":len(rows[4])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
