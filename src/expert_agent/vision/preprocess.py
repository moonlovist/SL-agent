from __future__ import annotations

from dataclasses import dataclass
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
    cy, cx = height // 2, width // 2
    half = min(config.center_search_half_size, cy, cx, height - cy, width - cx)
    central = data[cy - half : cy + half, cx - half : cx + half]
    local_y, local_x = np.unravel_index(int(np.argmax(central)), central.shape)
    nucleus = (int(local_y + cy - half), int(local_x + cx - half))

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
        central_index = int(np.argmin(distances))
        foreground_detected = True
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
    )
