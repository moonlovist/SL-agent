from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from ..core.config import VisionConfig


class VisionDependencyError(RuntimeError):
    pass


@dataclass
class ImageData:
    path: Path
    object_id: str
    input_format: str
    band: str
    data: np.ndarray
    rms: np.ndarray
    supplied_rms: bool
    finite_fraction: float
    wcs: WCS | None
    pixel_scale_arcsec: float | None
    psf: np.ndarray | None


@dataclass
class PreprocessedImage:
    image: ImageData
    model: np.ndarray
    residual: np.ndarray
    residual_snr: np.ndarray
    source_mask: np.ndarray
    fit_mask: np.ndarray
    segmentation: np.ndarray
    nucleus_yx: tuple[int, int]
    background_median: float
    background_rms: float
    residual_median: float
    residual_robust_rms: float
    model_method: str
    model_success: bool
    foreground_detected: bool
    foreground_peak_value: float
    foreground_integrated_flux: float | None
    foreground_ellipticity: float | None
    foreground_position_angle_deg: float | None
    foreground_axis_ratio: float | None
    foreground_reliability: float
    multiple_core_count: int | None
    tidal_asymmetry: float | None
    core_residual_fraction: float | None
    flags: list[str]
    center_selection: dict = field(default_factory=dict)
    source_protection_weight: np.ndarray | None = None
    foreground_fit_candidates: dict = field(default_factory=dict)
    detection_calibration: dict = field(default_factory=dict)
    human_foreground_seed: np.ndarray | None = None
    human_fit_exclude: np.ndarray | None = None
    human_analysis_exclude: np.ndarray | None = None


def robust_rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return 1.4826 * mad if mad > 0 else float(np.std(finite))


def resolve_input(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Input does not exist: {candidate}")
    preferred = candidate / f"{candidate.name}.fits"
    if preferred.exists():
        return preferred
    fits_files = sorted(candidate.glob("*.fits"))
    if len(fits_files) == 1:
        return fits_files[0]
    raise FileNotFoundError(
        f"Expected one FITS file in {candidate}; found {len(fits_files)}"
    )


def load_image(path: str | Path, band: str) -> ImageData:
    resolved = resolve_input(path)
    suffix = resolved.suffix.lower()
    object_id = resolved.stem
    if suffix in {".fits", ".fit", ".fts"}:
        with fits.open(resolved, memmap=False) as hdul:
            flux_name = f"{band}_FLUX"
            rms_name = f"{band}_RMS"
            if flux_name in hdul:
                data = np.asarray(hdul[flux_name].data, dtype="=f8")
                try:
                    candidate_wcs = WCS(hdul[flux_name].header)
                    wcs = candidate_wcs.celestial if candidate_wcs.has_celestial else None
                    scales = (
                        proj_plane_pixel_scales(wcs) * 3600.0
                        if wcs is not None
                        else np.asarray([])
                    )
                    pixel_scale_arcsec = (
                        float(np.sqrt(scales[0] * scales[1]))
                        if scales.size == 2 and np.all(np.isfinite(scales)) and np.all(scales > 0)
                        else None
                    )
                except Exception:
                    wcs = None
                    pixel_scale_arcsec = None
                psf_name = f"{band}_PSF"
                psf = (
                    np.asarray(hdul[psf_name].data, dtype="=f8")
                    if psf_name in hdul and hdul[psf_name].data is not None
                    else None
                )
                if rms_name in hdul:
                    rms = np.asarray(hdul[rms_name].data, dtype="=f8")
                    supplied_rms = True
                else:
                    sigma = robust_rms(data)
                    rms = np.full(data.shape, sigma, dtype=float)
                    supplied_rms = False
            elif hdul[0].data is not None and hdul[0].data.ndim == 2:
                data = np.asarray(hdul[0].data, dtype="=f8")
                sigma = robust_rms(data)
                rms = np.full(data.shape, sigma, dtype=float)
                supplied_rms = False
                try:
                    candidate_wcs = WCS(hdul[0].header)
                    wcs = candidate_wcs.celestial if candidate_wcs.has_celestial else None
                    scales = (
                        proj_plane_pixel_scales(wcs) * 3600.0
                        if wcs is not None
                        else np.asarray([])
                    )
                    pixel_scale_arcsec = (
                        float(np.sqrt(scales[0] * scales[1]))
                        if scales.size == 2 and np.all(np.isfinite(scales)) and np.all(scales > 0)
                        else None
                    )
                except Exception:
                    wcs = None
                    pixel_scale_arcsec = None
                psf = None
            else:
                raise ValueError(
                    f"FITS has no 2-D primary image or {flux_name} extension"
                )
        input_format = "fits"
    elif suffix == ".png":
        try:
            from PIL import Image
        except ImportError as exc:
            raise VisionDependencyError("PNG input requires Pillow") from exc
        rgb = np.asarray(Image.open(resolved).convert("L"), dtype=float) / 255.0
        data = rgb
        sigma = robust_rms(data)
        rms = np.full(data.shape, max(sigma, 1e-8), dtype=float)
        supplied_rms = False
        input_format = "png"
        wcs = None
        pixel_scale_arcsec = None
        psf = None
    else:
        raise ValueError(f"Unsupported input format: {resolved.suffix}")

    if data.ndim != 2 or rms.shape != data.shape:
        raise ValueError("Flux and RMS data must be matching 2-D arrays")
    finite = np.isfinite(data) & np.isfinite(rms)
    finite_fraction = float(finite.mean())
    if not finite.any():
        raise ValueError("Input contains no finite pixels")
    data = np.where(np.isfinite(data), data, np.nanmedian(data[finite]))
    positive_rms = rms[np.isfinite(rms) & (rms > 0)]
    fallback_rms = float(np.median(positive_rms)) if positive_rms.size else robust_rms(data)
    rms = np.where(np.isfinite(rms) & (rms > 0), rms, max(fallback_rms, 1e-8))
    return ImageData(
        path=resolved,
        object_id=object_id,
        input_format=input_format,
        band=band,
        data=data,
        rms=rms,
        supplied_rms=supplied_rms,
        finite_fraction=finite_fraction,
        wcs=wcs,
        pixel_scale_arcsec=pixel_scale_arcsec,
        psf=psf,
    )


def _smooth(values: np.ndarray, preferred: int) -> np.ndarray:
    from scipy.signal import savgol_filter

    count = len(values)
    if count < 5:
        return values
    window = min(preferred, count if count % 2 else count - 1)
    window = max(5, window)
    if window > count:
        return values
    return savgol_filter(values, window, 2, mode="interp")


def _radial_fallback_model(
    data: np.ndarray, mask: np.ndarray, nucleus_yx: tuple[int, int]
) -> np.ndarray:
    yy, xx = np.indices(data.shape)
    radius = np.hypot(xx - nucleus_yx[1], yy - nucleus_yx[0])
    max_radius = int(np.ceil(radius.max()))
    profile = np.zeros(max_radius + 1, dtype=float)
    valid = ~mask & np.isfinite(data)
    global_median = float(np.median(data[valid])) if valid.any() else 0.0
    for index in range(max_radius + 1):
        ring = valid & (radius >= index) & (radius < index + 1)
        profile[index] = float(np.median(data[ring])) if ring.any() else np.nan
    good = np.isfinite(profile)
    if good.sum() >= 2:
        profile = np.interp(np.arange(profile.size), np.flatnonzero(good), profile[good])
    else:
        profile[:] = global_median
    return np.interp(radius, np.arange(profile.size), profile)


def _ellipse_model(
    data: np.ndarray,
    mask: np.ndarray,
    nucleus_yx: tuple[int, int],
    initial_eps: float,
    initial_pa: float,
) -> np.ndarray:
    from photutils.isophote import Ellipse, EllipseGeometry

    fit_data = np.ma.array(data, mask=mask)
    geometry = EllipseGeometry(
        x0=float(nucleus_yx[1]),
        y0=float(nucleus_yx[0]),
        sma=5.0,
        eps=float(np.clip(initial_eps, 0.05, 0.7)),
        pa=float(initial_pa),
    )
    maxsma = min(60.0, 0.45 * min(data.shape))
    isolist = Ellipse(fit_data, geometry=geometry).fit_image(
        maxsma=maxsma, sclip=3.0, n_clip=2
    )
    table = isolist.to_table()
    eps_column = "ellipticity" if "ellipticity" in table.colnames else "eps"
    ok = (
        np.isfinite(table["intens"])
        & (table["intens"] > 0)
        & np.isfinite(table["sma"])
    )
    if int(ok.sum()) < 5:
        raise RuntimeError("Too few valid isophotes")
    sma = np.asarray(table["sma"][ok], dtype=float)
    intens = _smooth(np.asarray(table["intens"][ok], dtype=float), 7)
    eps = np.clip(_smooth(np.asarray(table[eps_column][ok], dtype=float), 5), 0, 0.7)
    pa = _smooth(np.asarray(table["pa"][ok], dtype=float), 5)

    yy, xx = np.indices(data.shape)
    dx = xx - geometry.x0
    dy = yy - geometry.y0
    circular_radius = np.hypot(dx, dy)
    eps_pixel = np.interp(circular_radius, sma, eps)
    pa_pixel = np.interp(circular_radius, sma, pa)
    cos_pa, sin_pa = np.cos(pa_pixel), np.sin(pa_pixel)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    elliptical_radius = np.sqrt(x_rot**2 + (y_rot / (1 - eps_pixel)) ** 2)
    model = np.interp(elliptical_radius, sma, intens, left=intens[0], right=0.0)
    model[elliptical_radius > sma.max()] = 0.0
    return model


def _target_morphology_metrics(data, rms, obj, config):
    """Measure inexpensive raw-image configuration proxies for routing only."""
    from scipy import ndimage

    cx, cy = float(obj["x"]), float(obj["y"])
    yy, xx = np.indices(data.shape)
    radius = np.hypot(xx - cx, yy - cy)
    names = obj.dtype.names or ()
    size = float(
        np.sqrt(max(float(obj["a"] * obj["b"]), 1.0))
        if "a" in names and "b" in names
        else 1.0
    )
    inner = max(2.5, min(0.7 * size, 7.0))
    outer = min(float(config.ring_outer_px), max(10.0, 2.8 * size))
    smoothed = ndimage.gaussian_filter(np.asarray(data, float), 1.0)
    annulus = (radius >= inner) & (radius <= outer)
    if int(annulus.sum()) < 30:
        return dict(angular_coverage=0.0, multiplicity=0.0, annular_contrast=0.0)
    radial_bin = np.floor(radius).astype(int)
    profile = np.zeros(int(np.ceil(outer)) + 2, dtype=float)
    for index in range(profile.size):
        ring = radial_bin == index
        profile[index] = float(np.median(smoothed[ring])) if ring.any() else 0.0
    departure = smoothed - profile[np.clip(radial_bin, 0, profile.size - 1)]
    local_noise = max(
        float(np.median(np.asarray(rms)[annulus])),
        robust_rms(departure[annulus]),
        1e-8,
    )
    positive = annulus & (departure > 1.5 * local_noise)
    angle = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    angle_bin = np.floor(
        angle / (2 * np.pi) * config.target_morphology_angle_bins
    ).astype(int)
    occupied = sum(
        np.count_nonzero(positive & (angle_bin == index)) >= 2
        for index in range(config.target_morphology_angle_bins)
    )
    labels, count = ndimage.label(positive)
    component_count = sum(
        np.count_nonzero(labels == label) >= 3 for label in range(1, count + 1)
    )
    contrast = float(np.percentile(departure[annulus], 95) / local_noise)
    return {
        "angular_coverage": float(occupied / config.target_morphology_angle_bins),
        "multiplicity": float(min(component_count, 4) / 4.0),
        "annular_contrast": float(np.clip((contrast - 1.0) / 5.0, 0.0, 1.0)),
    }


def select_foreground_center(data, objects, segmentation, config, rms=None):
    """Choose one analysis target from at most four extended objects.

    This is a pre-modelling routing choice.  Its metrics never enter lens
    evidence or grading and it never forks the downstream analysis.
    """
    from scipy import ndimage

    height, width = data.shape
    reference = config.target_center_xy or (width // 2, height // 2)
    rx, ry = map(float, reference)
    if not (np.isfinite(rx) and np.isfinite(ry) and 0 <= rx < width and 0 <= ry < height):
        raise ValueError("Target centre must be finite and inside the image")
    record = {
        "reference_xy": [rx, ry],
        "reference_source": "explicit" if config.target_center_xy else "cutout_center",
        "candidates": [],
        "routing_only": True,
        "affects_grading": False,
    }
    if not len(objects):
        record.update(
            status="selected",
            reason="no_detection_center_fallback",
            selected_xy=[int(round(rx)), int(round(ry))],
            selected_centroid_xy=[rx, ry],
        )
        return (int(round(ry)), int(round(rx))), None, record

    distances = np.hypot(objects["x"] - rx, objects["y"] - ry)
    noise = np.asarray(
        rms if rms is not None else np.full(data.shape, robust_rms(data)), float
    )
    local_fallback = None
    if (
        config.target_center_xy is None
        and float(np.min(distances)) > config.center_max_offset_px
    ):
        # A bright star's diffraction spike can bridge a real central galaxy
        # into a remote low-threshold SEP island.  Retry target deblending at a
        # high significance only for routing; residual detection still uses
        # the full multi-threshold pipeline below.
        try:
            import sep

            local_data = np.ascontiguousarray(data, dtype="=f8")
            local_noise = np.ascontiguousarray(noise, dtype="=f8")
            local_background = sep.Background(local_data)
            local_objects, local_segmentation = sep.extract(
                local_data - local_background.back(),
                5.0,
                err=local_noise,
                minarea=config.sep_min_area,
                deblend_nthresh=64,
                deblend_cont=0.0001,
                segmentation_map=True,
            )
            if len(local_objects):
                local_distances = np.hypot(
                    local_objects["x"] - rx, local_objects["y"] - ry
                )
                local_index = int(np.argmin(local_distances))
                local_area = float(
                    local_objects["a"][local_index]
                    * local_objects["b"][local_index]
                )
                if (
                    local_distances[local_index] <= config.center_max_offset_px
                    and local_area >= config.target_candidate_min_area_px
                ):
                    local_label = local_index + 1
                    local_mask = local_segmentation == local_label
                    peak = np.argmax(np.where(local_mask, data, -np.inf))
                    local_py, local_px = np.unravel_index(peak, data.shape)
                    obj = local_objects[local_index]
                    local_fallback = {
                        "sep_label": f"H5-{local_label}",
                        "centroid_xy": [float(obj["x"]), float(obj["y"])],
                        "offset_px": float(local_distances[local_index]),
                        "a_px": float(obj["a"]),
                        "b_px": float(obj["b"]),
                        "theta_rad": float(obj["theta"]),
                        "flux": float(obj["flux"]),
                        "peak_xy": [int(local_px), int(local_py)],
                        "selection_source": "high_sigma_central_deblend",
                        **_target_morphology_metrics(data, noise, obj, config),
                    }
        except Exception:
            local_fallback = None
    if config.target_center_xy is not None:
        pool = [int(np.argmin(distances))]
        basis = "explicit_nearest_detected_source"
    else:
        names = objects.dtype.names or ()
        area = (
            np.maximum(np.asarray(objects["a"] * objects["b"], float), 1e-6)
            if "a" in names and "b" in names
            else np.ones(len(objects), dtype=float)
        )
        flux = (
            np.maximum(np.abs(np.asarray(objects["flux"], float)), 1e-6)
            if "flux" in names
            else np.maximum(np.asarray([data[int(round(y)), int(round(x))] for x, y in zip(objects["x"], objects["y"])]), 1e-6)
        )
        eligible = np.flatnonzero(area >= float(config.target_candidate_min_area_px))
        if not eligible.size:
            eligible = np.arange(len(objects))
        shortlist_rank = (
            np.log1p(area[eligible])
            + 0.25 * np.log1p(flux[eligible])
            + 0.6 * np.exp(-distances[eligible] / max(config.center_search_half_size, 1))
        )
        limit = max(1, min(int(config.target_candidate_max), 4))
        pool = [
            int(eligible[index])
            for index in np.argsort(shortlist_rank)[::-1][:limit]
        ]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= config.center_max_offset_px and nearest not in pool:
            pool = [*pool[: max(limit - 1, 0)], nearest]
        basis = "raw_morphology_among_extended_shortlist"

    scored = []
    for index in pool:
        metrics = _target_morphology_metrics(data, noise, objects[index], config)
        centrality = float(
            np.exp(-distances[index] / max(config.center_search_half_size, 1))
        )
        score = (
            0.50 * metrics["angular_coverage"]
            + 0.25 * metrics["multiplicity"]
            + 0.20 * metrics["annular_contrast"]
            + 0.05 * centrality
        )
        record["candidates"].append(
            {
                "sep_label": int(index + 1),
                "centroid_xy": [float(objects[index]["x"]), float(objects[index]["y"])],
                "offset_px": float(distances[index]),
                "a_px": float(objects[index]["a"]) if "a" in (objects.dtype.names or ()) else None,
                "b_px": float(objects[index]["b"]) if "b" in (objects.dtype.names or ()) else None,
                "theta_rad": float(objects[index]["theta"]) if "theta" in (objects.dtype.names or ()) else None,
                "flux": float(objects[index]["flux"]) if "flux" in (objects.dtype.names or ()) else None,
                **metrics,
                "centrality_prior": centrality,
                "routing_score": float(score),
            }
        )
        scored.append((score, centrality, -int(index), int(index)))
    if local_fallback is not None:
        local_centrality = float(
            np.exp(-local_fallback["offset_px"] / max(config.center_search_half_size, 1))
        )
        local_fallback["centrality_prior"] = local_centrality
        local_fallback["routing_score"] = float(
            0.50 * local_fallback["angular_coverage"]
            + 0.25 * local_fallback["multiplicity"]
            + 0.20 * local_fallback["annular_contrast"]
            + 0.05 * local_centrality
        )
        record["candidates"].append(local_fallback)
    chosen = max(scored)[-1]
    chosen_local = False
    record["candidates"].sort(
        key=lambda item: (-item["routing_score"], item["sep_label"])
    )
    if config.target_center_xy is None:
        central = [
            item
            for item in record["candidates"]
            if item["offset_px"] <= config.center_max_offset_px
        ]
        if central:
            central_best = max(
                central,
                key=lambda item: (
                    item["routing_score"],
                    -item["sep_label"] if isinstance(item["sep_label"], int) else 0,
                ),
            )
            overall = next(
                item for item in record["candidates"] if item["sep_label"] == chosen + 1
            )
            if overall["sep_label"] != central_best["sep_label"]:
                central_area = max(
                    float(central_best.get("a_px") or 1.0)
                    * float(central_best.get("b_px") or 1.0),
                    1e-6,
                )
                overall_area = max(
                    float(overall.get("a_px") or 1.0)
                    * float(overall.get("b_px") or 1.0),
                    1e-6,
                )
                central_flux = abs(float(central_best.get("flux") or central_area))
                overall_flux = abs(float(overall.get("flux") or overall_area))
                separation = float(
                    np.hypot(
                        float(overall["centroid_xy"][0]) - float(central_best["centroid_xy"][0]),
                        float(overall["centroid_xy"][1]) - float(central_best["centroid_xy"][1]),
                    )
                )
                ellipse_overlap = bool(
                    separation
                    <= 1.2
                    * (
                        float(central_best.get("a_px") or np.sqrt(central_area))
                        + float(overall.get("a_px") or np.sqrt(overall_area))
                    )
                )
                footprint_overlap = False
                if isinstance(central_best["sep_label"], int) and isinstance(overall["sep_label"], int):
                    central_pixels = segmentation == int(central_best["sep_label"])
                    overall_pixels = segmentation == int(overall["sep_label"])
                    footprint_overlap = bool(
                        np.any(
                            ndimage.binary_dilation(central_pixels, iterations=2)
                            & ndimage.binary_dilation(overall_pixels, iterations=2)
                        )
                    )
                # Low-threshold SEP footprints can nearly touch through diffuse
                # sky or a diffraction bridge.  That is not sufficient evidence
                # that the two extended light distributions overlap.  Require
                # the measured ellipses themselves to overlap; keep footprint
                # proximity only as an audit value.
                light_profiles_overlap = ellipse_overlap
                offcenter_override = bool(
                    overall["offset_px"] <= config.center_blend_radius_px
                    and overall_area >= 2.0 * central_area
                    and overall_flux >= 2.0 * central_flux
                    and light_profiles_overlap
                    and overall["routing_score"] >= central_best["routing_score"] + 0.05
                    and overall["angular_coverage"]
                    >= central_best["angular_coverage"] + 0.10
                )
                if not offcenter_override:
                    if central_best.get("selection_source") == "high_sigma_central_deblend":
                        chosen_local = True
                    else:
                        chosen = int(central_best["sep_label"] - 1)
                record["central_prior_gate"] = {
                    "central_sep_label": central_best["sep_label"],
                    "raw_winner_sep_label": overall["sep_label"],
                    "offcenter_override_accepted": offcenter_override,
                    "larger": bool(overall_area >= 2.0 * central_area),
                    "brighter": bool(overall_flux >= 2.0 * central_flux),
                    "light_profiles_overlap": light_profiles_overlap,
                    "segmentation_footprints_near": footprint_overlap,
                    "centroid_separation_px": separation,
                    "rule": "offcenter_must_be_larger_brighter_overlapping_and_have_clear_morphology_advantage",
                }

    if chosen_local and local_fallback is not None:
        px, py = local_fallback["peak_xy"]
        record.update(
            status="selected",
            selected_sep_label=local_fallback["sep_label"],
            selected_xy=[int(px), int(py)],
            selected_centroid_xy=local_fallback["centroid_xy"],
            offset_px=float(np.hypot(px - rx, py - ry)),
            selection_basis="high_sigma_central_deblend_after_low_threshold_bridge",
            reason="single_analysis_target_selected_before_foreground_modelling",
            local_deblended_target={
                key: local_fallback[key]
                for key in ("centroid_xy", "peak_xy", "a_px", "b_px", "theta_rad", "flux")
            },
        )
        return (int(py), int(px)), None, record

    allowed = segmentation == chosen + 1
    if allowed.any():
        peak = np.argmax(np.where(allowed, data, -np.inf))
        py, px = np.unravel_index(peak, data.shape)
    else:
        px, py = int(round(objects[chosen]["x"])), int(round(objects[chosen]["y"]))
    record.update(
        status="selected",
        selected_sep_label=chosen + 1,
        selected_xy=[int(px), int(py)],
        selected_centroid_xy=[float(objects[chosen]["x"]), float(objects[chosen]["y"])],
        offset_px=float(np.hypot(px - rx, py - ry)),
        selection_basis=basis,
        reason="single_analysis_target_selected_before_foreground_modelling",
    )
    return (int(py), int(px)), chosen, record


def preprocess(image: ImageData, config: VisionConfig) -> PreprocessedImage:
    try:
        import sep
        from scipy import ndimage
    except ImportError as exc:
        raise VisionDependencyError(
            "Vision pipeline requires scipy and sep; install project dependencies"
        ) from exc

    sep.set_sub_object_limit(100000)
    data = np.ascontiguousarray(image.data, dtype="=f8")
    rms = np.ascontiguousarray(image.rms, dtype="=f8")
    height, width = data.shape

    background = sep.Background(data)
    background_array = background.back()
    background_subtracted = data - background_array
    objects, segmentation = sep.extract(
        background_subtracted,
        config.source_detect_sigma,
        err=rms,
        deblend_nthresh=16 if image.input_format == "png" else 32,
        deblend_cont=0.01 if image.input_format == "png" else 0.005,
        segmentation_map=True,
    )
    nucleus, central_index, center_selection = select_foreground_center(
        data, objects, segmentation, config, rms=rms
    )
    source_mask = np.zeros(data.shape, dtype=bool)
    fit_mask = np.zeros(data.shape, dtype=bool)
    initial_eps, initial_pa = 0.2, 0.0
    foreground_detected = False
    foreground_flux: float | None = None
    foreground_axis_ratio: float | None = None
    foreground_reliability = 0.25
    multiple_core_count: int | None = None
    if len(objects):
        distances = (objects["x"] - nucleus[1]) ** 2 + (objects["y"] - nucleus[0]) ** 2
        foreground_detected = True
        local_target = center_selection.get("local_deblended_target")
        if local_target is not None:
            yy, xx = np.indices(data.shape)
            target_radius = max(
                config.near_core_radius_px,
                2.5 * np.sqrt(local_target["a_px"] * local_target["b_px"]),
            )
            protected_target = (
                np.hypot(xx - nucleus[1], yy - nucleus[0]) <= target_radius
            )
            fit_mask = segmentation != 0
            fit_mask = ndimage.binary_dilation(fit_mask, iterations=3)
            fit_mask[protected_target] = False
            source_mask = segmentation != 0
            source_mask = ndimage.binary_dilation(source_mask, iterations=3)
            source_mask[
                np.hypot(xx - nucleus[1], yy - nucleus[0])
                <= config.ring_outer_px
            ] = False
            initial_eps = float(
                np.clip(
                    1 - local_target["b_px"] / max(local_target["a_px"], 1e-9),
                    0.05,
                    0.7,
                )
            )
            initial_pa = float(local_target["theta_rad"])
            foreground_flux = float(local_target["flux"])
            foreground_axis_ratio = float(
                local_target["a_px"] / max(local_target["b_px"], 1e-9)
            )
            foreground_reliability = float(
                np.clip(
                    abs(foreground_flux) / max(20.0 * background.globalrms, 1e-9),
                    0.25,
                    1.0,
                )
            )
            multiple_core_count = None
        else:
            central_label = central_index + 1
        # Fit the foreground using the central object only, so candidate arcs do
        # not bias its isophotes. During residual detection, however, preserve
        # non-central sources inside the analysis annulus and mask only external
        # contaminants. The prototype used one mask for both purposes and could
        # erase the very arcs it intended to measure.
            fit_mask = (segmentation != 0) & (segmentation != central_label)
            fit_mask = ndimage.binary_dilation(fit_mask, iterations=3)
            external_labels = (
                np.where(distances > config.ring_outer_px**2)[0] + 1
            ).tolist()
            source_mask = np.isin(segmentation, external_labels)
            source_mask = ndimage.binary_dilation(source_mask, iterations=3)
            initial_eps = float(
                np.clip(
                    1 - objects["b"][central_index] / max(objects["a"][central_index], 1e-9),
                    0.05,
                    0.7,
                )
            )
            initial_pa = float(objects["theta"][central_index])
            foreground_flux = float(objects["flux"][central_index])
            foreground_axis_ratio = float(
                objects["a"][central_index] / max(objects["b"][central_index], 1e-9)
            )
            foreground_reliability = float(
                np.clip(abs(foreground_flux) / max(20.0 * background.globalrms, 1e-9), 0.25, 1.0)
            )
            # Keep the core screen inside the subtraction-sensitive central scale.
            # A wider aperture can incorrectly count the very annular structures
            # under investigation as additional galaxy nuclei.
            core_radius = config.near_core_radius_px
            central_flux_scale = max(abs(foreground_flux), 1e-9)
            plausible_secondary = (
                (distances <= core_radius**2)
                & (np.arange(len(objects)) != central_index)
                & (np.abs(objects["flux"]) >= 0.1 * central_flux_scale)
            )
            multiple_core_count = int(np.count_nonzero(plausible_secondary))

    flags: list[str] = []
    if image.input_format == "png":
        model = _radial_fallback_model(data, fit_mask, nucleus)
        model_method = "robust_radial_png"
        model_success = True
        flags.append("png_reduced_measurement_mode")
    else:
        try:
            model = _ellipse_model(data, fit_mask, nucleus, initial_eps, initial_pa)
            model_method = "photutils_isophote_smoothed"
            model_success = True
        except Exception as exc:
            model = _radial_fallback_model(data, fit_mask, nucleus)
            model_method = "robust_radial_fallback"
            model_success = False
            flags.append(f"ellipse_model_failed:{type(exc).__name__}")

    residual = data - model
    residual_snr = np.divide(residual, rms, out=np.zeros_like(residual), where=rms > 0)
    yy, xx = np.indices(data.shape)
    radius = np.hypot(xx - nucleus[1], yy - nucleus[0])
    background_region = (radius > config.ring_outer_px * 1.5) & ~source_mask
    if not background_region.any():
        background_region = ~source_mask
    symmetric_half_y = min(nucleus[0], data.shape[0] - 1 - nucleus[0])
    symmetric_half_x = min(nucleus[1], data.shape[1] - 1 - nucleus[1])
    if symmetric_half_y >= 1 and symmetric_half_x >= 1:
        symmetric = residual[
            nucleus[0] - symmetric_half_y : nucleus[0] + symmetric_half_y + 1,
            nucleus[1] - symmetric_half_x : nucleus[1] + symmetric_half_x + 1,
        ]
        rotated = np.rot90(symmetric, 2)
        asymmetry_denominator = float(np.sum(np.abs(symmetric)) + np.sum(np.abs(rotated)))
        tidal_asymmetry = (
            float(np.sum(np.abs(symmetric - rotated)) / asymmetry_denominator)
            if asymmetry_denominator > 0
            else 0.0
        )
    else:
        tidal_asymmetry = None
    core_region = radius <= config.near_core_radius_px
    comparison_region = radius <= config.ring_outer_px
    residual_denominator = float(np.sum(np.abs(residual[comparison_region])))
    core_residual_fraction = (
        float(np.sum(np.abs(residual[core_region])) / residual_denominator)
        if residual_denominator > 0
        else 0.0
    )
    return PreprocessedImage(
        image=image,
        model=model,
        residual=residual,
        residual_snr=residual_snr,
        source_mask=source_mask,
        fit_mask=fit_mask,
        segmentation=segmentation,
        nucleus_yx=nucleus,
        background_median=float(np.median(background_array)),
        background_rms=float(background.globalrms),
        residual_median=float(np.median(residual[background_region])),
        residual_robust_rms=robust_rms(residual[background_region]),
        model_method=model_method,
        model_success=model_success,
        foreground_detected=foreground_detected,
        foreground_peak_value=float(data[nucleus]),
        foreground_integrated_flux=foreground_flux,
        foreground_ellipticity=initial_eps if foreground_detected else None,
        foreground_position_angle_deg=(
            float(np.degrees(initial_pa) % 180.0) if foreground_detected else None
        ),
        foreground_axis_ratio=foreground_axis_ratio,
        foreground_reliability=foreground_reliability,
        multiple_core_count=multiple_core_count,
        tidal_asymmetry=tidal_asymmetry,
        core_residual_fraction=core_residual_fraction,
        flags=flags,
        center_selection=center_selection,
    )
