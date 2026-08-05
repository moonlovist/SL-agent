from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

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
    else:
        raise ValueError(f"Unsupported input format: {resolved.suffix}")

    if data.ndim != 2 or rms.shape != data.shape:
        raise ValueError("Flux and RMS data must be matching 2-D arrays")
    finite = np.isfinite(data) & np.isfinite(rms)
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
    if len(objects):
        distances = (objects["x"] - nucleus[1]) ** 2 + (objects["y"] - nucleus[0]) ** 2
        central_index = int(np.argmin(distances))
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
        flags=flags,
    )
