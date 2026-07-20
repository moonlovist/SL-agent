import re
import numpy as np
from scipy.optimize import curve_fit
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Rest-frame line tables (single source of truth — keep in sync with VI.py)
# ---------------------------------------------------------------------------

EMISSION_LINES = {
    "Lyα":       1216.0,
    "C IV":      1549.0,
    "He II":     1640.4,
    "C III]":    1909.0,
    "Mg II":     2800.0,
    "[Ne V]":    3426.0,
    "[O II]":    3727.0,
    "Hε":        3970.1,
    "Hδ":        4102.9,
    "Hγ":        4341.7,
    "Hβ":        4862.7,
    "[O III]a":  4960.3,
    "[O III]b":  5008.2,
    "[N II]a":   6549.8,
    "Hα":        6564.6,
    "[N II]b":   6585.3,
    "[S II]a":   6718.3,
    "[S II]b":   6732.7,
}

EMISSION_LINE_WIDTHS = {
    "Lyα":       "broad",
    "C IV":      "broad",
    "C III]":    "broad",
    "He II":     "both",
    "Mg II":     "broad",
    "Hε":        "both",
    "Hδ":        "both",
    "Hγ":        "both",
    "Hβ":        "both",
    "Hα":        "both",
    "[Ne V]":    "narrow",
    "[O II]":    "narrow",
    "[O III]a":  "narrow",
    "[O III]b":  "narrow",
    "[N II]a":   "narrow",
    "[N II]b":   "narrow",
    "[S II]a":   "narrow",
    "[S II]b":   "narrow",
}

ABSORPTION_LINES = {
    "Ca K_abs":      3934.8,
    "Ca H_abs":      3969.6,
    "G-band_abs":    4305.6,
    "Mg I_abs":      5176.7,
    "Mg II_abs":     2800.0,
    "Na D_abs":      5895.6,
    "CaT1_abs":      8498.0,
    "CaT2_abs":      8542.0,
    "CaT3_abs":      8662.0,
    "Hε_abs":        3970.1,
    "Hδ_abs":        4102.9,
    "Hγ_abs":        4341.7,
    "Hβ_abs":        4862.7,
    "Hα_abs":        6564.6,
}

_WIDTH_3SIGMA_MAP = {
    "broad":     90.0,
    "narrow":    25.0,
    "both":      50.0,
    "absorption": 20.0,
}


# ---------------------------------------------------------------------------
# Tool 1: load_spectrum
# ---------------------------------------------------------------------------

@tool
def load_spectrum(npz_path: str) -> dict:
    """Load a cleaned spectrum .npz file and return summary statistics.

    The .npz must contain arrays: wavelength, flux, snr (and optionally ivar).

    Parameters
    ----------
    npz_path : str
        Absolute path to the .npz file.

    Returns
    -------
    dict with keys:
        n_points : int          — number of wavelength bins
        wavelength_min : float  — minimum wavelength (Å)
        wavelength_max : float  — maximum wavelength (Å)
        snr_median : float      — median SNR across the spectrum
        n_arms : str or None    — number of distinct bands (if detectable)
    """
    data = np.load(npz_path)
    wl = data["wavelength"]
    snr = data.get("snr")

    # Detect bands by looking for large gaps in wavelength coverage
    gaps = np.diff(wl)
    n_arms = int(np.sum(gaps > 100)) + 1  # >100 Å gap = band boundary

    return {
        "n_points": int(len(wl)),
        "wavelength_min": float(wl[0]),
        "wavelength_max": float(wl[-1]),
        "snr_median": float(np.median(snr)) if snr is not None else None,
        "n_arms": str(n_arms) if n_arms > 1 else None,
    }


# ---------------------------------------------------------------------------
# Tool 2: predict_lines
# ---------------------------------------------------------------------------

@tool
def predict_lines(
    redshift: float,
    line_type: str = "all",
    wavelength_min: float = None,
    wavelength_max: float = None,
) -> dict:
    """Predict observed-frame wavelengths for rest-frame spectral lines at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift hypothesis z.
    line_type : str
        "emission", "absorption", or "all" (default).
    wavelength_min, wavelength_max : float, optional
        If provided, only return lines whose predicted λ_obs falls within [min, max].

    Returns
    -------
    dict with keys:
        redshift : float
        n_lines : int
        lines : list[dict]
            Each dict has: name, rest_wl, obs_wl, width_class, type
    """
    lines_out = []

    if line_type in ("emission", "all"):
        for name, rest_wl in EMISSION_LINES.items():
            obs_wl = rest_wl * (1.0 + redshift)
            if wavelength_min is not None and wavelength_max is not None:
                if obs_wl < wavelength_min or obs_wl > wavelength_max:
                    continue
            lines_out.append({
                "name": name,
                "rest_wl": round(rest_wl, 1),
                "obs_wl": round(obs_wl, 1),
                "width_class": EMISSION_LINE_WIDTHS.get(name, "narrow"),
                "type": "emission",
            })

    if line_type in ("absorption", "all"):
        for name, rest_wl in ABSORPTION_LINES.items():
            obs_wl = rest_wl * (1.0 + redshift)
            if wavelength_min is not None and wavelength_max is not None:
                if obs_wl < wavelength_min or obs_wl > wavelength_max:
                    continue
            lines_out.append({
                "name": name,
                "rest_wl": round(rest_wl, 1),
                "obs_wl": round(obs_wl, 1),
                "width_class": "absorption",
                "type": "absorption",
            })

    lines_out.sort(key=lambda x: x["obs_wl"])
    return {
        "redshift": redshift,
        "n_lines": len(lines_out),
        "lines": lines_out,
    }


# ---------------------------------------------------------------------------
# Tool 3: fit_peak
# ---------------------------------------------------------------------------

def _gaussian_plus_linear(x, amp, center, sigma, slope, intercept):
    return amp * np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + slope * x + intercept


def _do_fit_peak(
    wl_full: np.ndarray,
    flux_full: np.ndarray,
    center_guess: float,
    width_3sigma: float,
    line_type: str = "emission",
    window_half: float = 200.0,
) -> dict:
    """Core fitting logic — works on in-memory arrays (no file I/O)."""
    mask = (wl_full >= center_guess - window_half) & (wl_full <= center_guess + window_half)
    wl = wl_full[mask]
    flux = flux_full[mask]

    if len(wl) < 10:
        return {
            "center": None, "center_err": None,
            "amplitude": None, "amplitude_err": None,
            "sigma": None, "fwhm": None, "fwhm_km_s": None,
            "delta_chi2_per_n": None, "local_rms": None, "local_snr": None,
            "n_points": len(wl),
            "flags": ["too_few_points"],
            "message": f"Only {len(wl)} points in window — need at least 10.",
        }

    lin_coeffs = np.polyfit(wl, flux, 1)
    slope0, intercept0 = lin_coeffs[0], lin_coeffs[1]
    flux_at_center = np.interp(center_guess, wl, flux)
    linear_at_center = slope0 * center_guess + intercept0
    amp0 = flux_at_center - linear_at_center

    if line_type == "absorption":
        amp0 = min(amp0, -1e-6); amp_lower, amp_upper = -np.inf, 0.0
    else:
        amp0 = max(amp0, 1e-6);  amp_lower, amp_upper = 0.0, np.inf

    sigma0 = np.clip(width_3sigma / 3.0, 2.0, window_half / 2)
    p0 = [amp0, center_guess, sigma0, slope0, intercept0]
    bounds = (
        [amp_lower, center_guess - 50, 1.0,      -np.inf, -np.inf],
        [amp_upper, center_guess + 50, window_half, np.inf,  np.inf],
    )

    try:
        popt, pcov = curve_fit(_gaussian_plus_linear, wl, flux, p0=p0, bounds=bounds, maxfev=5000)
    except Exception as e:
        return {
            "center": None, "center_err": None,
            "amplitude": None, "amplitude_err": None,
            "sigma": None, "fwhm": None, "fwhm_km_s": None,
            "delta_chi2_per_n": None, "local_rms": None, "local_snr": None,
            "n_points": len(wl),
            "flags": ["fit_failed"],
            "message": f"curve_fit failed: {e}",
        }

    amp, center, sigma, slope, intercept = popt
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [None] * 5

    fitted = _gaussian_plus_linear(wl, *popt)
    linear_only = slope * wl + intercept
    residuals = flux - fitted

    local_rms = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    if local_rms < 1e-10:
        local_rms = np.std(residuals) or 1e-10

    n = len(wl)
    chi2_full = np.sum((residuals / local_rms) ** 2)
    chi2_linear = np.sum(((flux - linear_only) / local_rms) ** 2)
    delta_chi2_per_n = round((chi2_linear - chi2_full) / n, 3)
    local_snr = round(abs(amp) / local_rms, 2)
    fwhm = sigma * 2.35482
    fwhm_km_s = fwhm / center * 2.99792458e5 if center != 0 else None

    flags = []
    if abs(center - center_guess) > 50:
        flags.append("center_deviation_large")
    if delta_chi2_per_n < 0:
        flags.append("negative_delta_chi2")

    return {
        "center": round(center, 3),
        "center_err": round(perr[1], 4) if perr[1] is not None else None,
        "amplitude": round(amp, 6),
        "amplitude_err": round(perr[0], 6) if perr[0] is not None else None,
        "sigma": round(sigma, 3),
        "fwhm": round(fwhm, 3),
        "fwhm_km_s": round(fwhm_km_s, 1) if fwhm_km_s is not None else None,
        "delta_chi2_per_n": delta_chi2_per_n,
        "local_rms": round(local_rms, 6),
        "local_snr": local_snr,
        "n_points": n,
        "flags": flags,
        "message": (
            f"Fit {'OK' if not flags else 'with warnings'}. "
            f"center={center:.2f}±{perr[1]:.3f} Å, amp={amp:.4f}, "
            f"FWHM={fwhm:.1f} Å ({fwhm_km_s:.0f} km/s), "
            f"S/N={local_snr:.1f}, Δχ²/n={delta_chi2_per_n:.1f}"
        ),
    }


@tool
def fit_peak(
    npz_path: str,
    center_guess: float,
    width_3sigma: float,
    line_type: str = "emission",
    window_half: float = 200.0,
) -> dict:
    """Fit a Gaussian + linear baseline model around a predicted line position.

    Parameters
    ----------
    npz_path : str
        Path to the cleaned spectrum .npz file.
    center_guess : float
        Predicted observed wavelength (Å) — initial guess for the Gaussian center.
    width_3sigma : float
        Expected physical width (3σ) of the line in Å. Typical values:
        broad lines (Lyα, C IV, C III], Mg II) → 90
        narrow lines (Ne [V], O [II], O [III])  → 25
        both (Balmer, He II)                     → 50
        absorption                                → 20
    line_type : str
        "emission" (default) or "absorption". Constrains amplitude sign:
        emission → amp ≥ 0, absorption → amp ≤ 0.
    window_half : float
        Half-width of the fitting window in Å (default 200).

    Returns
    -------
    dict — see _do_fit_peak for field descriptions.
    """
    data = np.load(npz_path)
    return _do_fit_peak(data["wavelength"], data["flux"], center_guess,
                        width_3sigma, line_type, window_half)


# ---------------------------------------------------------------------------
# Tool 4: fit_doublet — two-component Gaussian fitting for close line pairs
# ---------------------------------------------------------------------------

def _doublet_model(x, amp1, center1, sigma1, amp2, center2, sigma2, slope, intercept):
    """Two Gaussians + linear baseline."""
    g1 = amp1 * np.exp(-((x - center1) / sigma1) ** 2 / 2)
    g2 = amp2 * np.exp(-((x - center2) / sigma2) ** 2 / 2)
    return g1 + g2 + slope * x + intercept


# Known doublet rest-frame parameters (name -> (wl1, wl2, sep, amp_ratio_hint))
_DOUBLET_PARAMS = {
    "Ca H/K":   (3934.8, 3969.6, 34.8, 0.5, "absorption"),
    "O III":    (4960.3, 5008.2, 47.9, 3.0, "emission"),
    "S II":     (6718.3, 6732.7, 14.4, 1.0, "emission"),
    "N II":     (6549.8, 6585.3, 35.5, 3.0, "emission"),
    "Na D":     (5891.6, 5897.6,  6.0, 2.0, "absorption"),
}


def _do_fit_doublet(
    wl_full: np.ndarray,
    flux_full: np.ndarray,
    center_guess_1: float,
    center_guess_2: float,
    line_type: str = "emission",
    width_3sigma: float = 25.0,
    window_half: float = 300.0,
    separation_rest: float = None,
    separation_tolerance: float = 5.0,
    amp_ratio_expected: float = None,
) -> dict:
    """Core doublet fitting — works on in-memory arrays (no file I/O)."""
    mid = (center_guess_1 + center_guess_2) / 2.0
    half = max(window_half, abs(center_guess_2 - center_guess_1) * 2.0)
    mask = (wl_full >= mid - half) & (wl_full <= mid + half)
    wl = wl_full[mask]
    flux = flux_full[mask]

    min_pts = 20
    if len(wl) < min_pts:
        return {
            "component_1": None, "component_2": None,
            "delta_chi2_per_n": None, "local_rms": None, "local_snr": None,
            "separation_check": None, "amp_ratio_check": None,
            "n_points": len(wl),
            "flags": ["too_few_points"],
            "message": f"Only {len(wl)} points in window — need at least {min_pts}.",
        }

    # ── Initial guess ─────────────────────────────────────────
    lin_coeffs = np.polyfit(wl, flux, 1)
    slope0, intercept0 = lin_coeffs[0], lin_coeffs[1]
    linear_at_c1 = slope0 * center_guess_1 + intercept0
    linear_at_c2 = slope0 * center_guess_2 + intercept0
    flux_at_c1 = np.interp(center_guess_1, wl, flux)
    flux_at_c2 = np.interp(center_guess_2, wl, flux)
    amp01 = flux_at_c1 - linear_at_c1
    amp02 = flux_at_c2 - linear_at_c2

    if line_type == "absorption":
        amp01 = min(amp01, -1e-6)
        amp02 = min(amp02, -1e-6)
    else:
        amp01 = max(amp01, 1e-6)
        amp02 = max(amp02, 1e-6)

    sigma0 = np.clip(width_3sigma / 3.0, 2.0, half / 4.0)
    p0 = [amp01, center_guess_1, sigma0, amp02, center_guess_2, sigma0,
          slope0, intercept0]

    # ── Bounds ─────────────────────────────────────────────────
    if line_type == "absorption":
        amp_lo, amp_hi = -np.inf, 0.0
    else:
        amp_lo, amp_hi = 0.0, np.inf

    bounds = (
        [amp_lo, center_guess_1 - 50, 1.0,  amp_lo, center_guess_2 - 50, 1.0,  -np.inf, -np.inf],
        [amp_hi, center_guess_1 + 50, half, amp_hi, center_guess_2 + 50, half,  np.inf,  np.inf],
    )

    # ── Fit ────────────────────────────────────────────────────
    try:
        popt, pcov = curve_fit(
            _doublet_model, wl, flux,
            p0=p0, bounds=bounds, maxfev=10000,
        )
    except Exception as e:
        # Fallback: fit each component individually
        flags = ["doublet_fit_failed"]
        msg = f"Doublet fit failed: {e}. "
        c1_result = _try_fit_single(wl, flux, center_guess_1, width_3sigma,
                                    line_type, window_half)
        c2_result = _try_fit_single(wl, flux, center_guess_2, width_3sigma,
                                    line_type, window_half)
        if c1_result and c2_result:
            flags.append("fallback_individual")
            msg += "Fell back to individual single-Gaussian fits."
        else:
            msg += "Individual fits also failed."
        return _build_doublet_result(None, None, len(wl), flags, msg,
                                     c1_result, c2_result)

    amp1, c1, s1, amp2, c2, s2, slope, intercept = popt
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [None] * 8

    def _comp_stats(i, amp, center, sigma, perr_amp, perr_center):
        if center == 0:
            return None
        fwhm = sigma * 2.35482
        fwhm_kms = fwhm / center * 2.99792458e5
        return {
            "center": round(float(center), 3),
            "center_err": round(float(perr_center), 4) if perr_center is not None else None,
            "amplitude": round(float(amp), 6),
            "amplitude_err": round(float(perr_amp), 6) if perr_amp is not None else None,
            "sigma": round(float(sigma), 3),
            "fwhm": round(float(fwhm), 3),
            "fwhm_km_s": round(float(fwhm_kms), 1) if fwhm_kms is not None else None,
        }

    c1_info = _comp_stats(1, amp1, c1, s1, perr[0], perr[1])
    c2_info = _comp_stats(2, amp2, c2, s2, perr[3], perr[4])

    # ── Statistics ─────────────────────────────────────────────
    fitted = _doublet_model(wl, *popt)
    linear_only = slope * wl + intercept
    residuals = flux - fitted

    local_rms = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    if local_rms < 1e-10:
        local_rms = np.std(residuals) or 1e-10

    n = len(wl)
    chi2_full = np.sum((residuals / local_rms) ** 2)
    chi2_linear = np.sum(((flux - linear_only) / local_rms) ** 2)
    delta_chi2_per_n = round((chi2_linear - chi2_full) / n, 3)

    max_amp = max(abs(amp1), abs(amp2))
    local_snr = round(max_amp / local_rms, 2) if local_rms > 0 else None

    # ── Separation check ───────────────────────────────────────
    sep_check = None
    if separation_rest is not None and c1_info and c2_info:
        obs_sep = abs(c1_info["center"] - c2_info["center"])
        sep_check = {
            "observed_sep_A": round(obs_sep, 2),
            "expected_sep_A": round(float(separation_rest), 2),
            "match": obs_sep <= separation_rest + separation_tolerance,
        }

    # ── Amplitude ratio check ──────────────────────────────────
    amp_check = None
    if amp_ratio_expected is not None and c1_info and c2_info:
        if abs(c1_info["amplitude"]) > 1e-10:
            obs_ratio = abs(c2_info["amplitude"]) / abs(c1_info["amplitude"])
        else:
            obs_ratio = None
        amp_check = {
            "observed_ratio": round(obs_ratio, 2) if obs_ratio is not None else None,
            "expected_ratio": round(float(amp_ratio_expected), 2),
        }

    # ── Flags ──────────────────────────────────────────────────
    flags = []
    for i, ci, c_guess in [(1, c1_info, center_guess_1), (2, c2_info, center_guess_2)]:
        if ci and abs(ci["center"] - c_guess) > 50:
            flags.append(f"center_{i}_deviation_large")
    if delta_chi2_per_n < 0:
        flags.append("negative_delta_chi2")

    msg_parts = [f"Fit {'OK' if not flags else 'with warnings'}."]
    if c1_info:
        msg_parts.append(
            f"C1: {c1_info['center']:.2f}±{c1_info.get('center_err', 0) or 0:.3f} Å, "
            f"amp={c1_info['amplitude']:.4f}, FWHM={c1_info['fwhm']:.1f} Å"
        )
    if c2_info:
        msg_parts.append(
            f"C2: {c2_info['center']:.2f}±{c2_info.get('center_err', 0) or 0:.3f} Å, "
            f"amp={c2_info['amplitude']:.4f}, FWHM={c2_info['fwhm']:.1f} Å"
        )
    msg_parts.append(f"S/N={local_snr:.1f}, Δχ²/n={delta_chi2_per_n:.1f}")
    if sep_check:
        status = "✓" if sep_check["match"] else "✗"
        msg_parts.append(f"sep={sep_check['observed_sep_A']:.1f} Å {status}")

    return {
        "component_1": c1_info,
        "component_2": c2_info,
        "delta_chi2_per_n": delta_chi2_per_n,
        "local_rms": round(local_rms, 6),
        "local_snr": local_snr,
        "separation_check": sep_check,
        "amp_ratio_check": amp_check,
        "n_points": n,
        "flags": flags,
        "message": " ".join(msg_parts),
    }


@tool
def fit_doublet(
    npz_path: str,
    center_guess_1: float,
    center_guess_2: float,
    line_type: str = "emission",
    width_3sigma: float = 25.0,
    window_half: float = 300.0,
    separation_rest: float = None,
    separation_tolerance: float = 5.0,
    amp_ratio_expected: float = None,
) -> dict:
    """Fit a doublet (two Gaussians + linear baseline) around a close line pair.

    (See _do_fit_doublet for full parameter docs.)
    """
    data = np.load(npz_path)
    return _do_fit_doublet(data["wavelength"], data["flux"],
                           center_guess_1, center_guess_2,
                           line_type, width_3sigma, window_half,
                           separation_rest, separation_tolerance, amp_ratio_expected)


def _try_fit_single(wl, flux, center_guess, width_3sigma, line_type, window_half):
    """Attempt a single-Gaussian fit. Returns dict or None on failure."""
    half = min(window_half, 200.0)
    mask = (wl >= center_guess - half) & (wl <= center_guess + half)
    w = wl[mask]; f = flux[mask]
    if len(w) < 10:
        return None
    lin = np.polyfit(w, f, 1)
    s0, i0 = lin[0], lin[1]
    fa = np.interp(center_guess, w, f)
    la = s0 * center_guess + i0
    a0 = fa - la
    if line_type == "absorption":
        a0 = min(a0, -1e-6); lo, hi = -np.inf, 0.0
    else:
        a0 = max(a0, 1e-6); lo, hi = 0.0, np.inf
    sig0 = np.clip(width_3sigma / 3.0, 2.0, half / 2)
    try:
        popt, _ = curve_fit(
            _gaussian_plus_linear, w, f,
            p0=[a0, center_guess, sig0, s0, i0],
            bounds=([lo, center_guess - 50, 1.0, -np.inf, -np.inf],
                    [hi, center_guess + 50, half, np.inf, np.inf]),
            maxfev=5000,
        )
        amp, c, s, _, _ = popt
        fwhm = s * 2.35482
        return {
            "center": round(float(c), 3),
            "center_err": None,
            "amplitude": round(float(amp), 6),
            "amplitude_err": None,
            "sigma": round(float(s), 3),
            "fwhm": round(float(fwhm), 3),
            "fwhm_km_s": round(fwhm / c * 2.99792458e5, 1) if c != 0 else None,
        }
    except Exception:
        return None


def _build_doublet_result(c1_info, c2_info, n_pts, flags, msg,
                          fallback_c1=None, fallback_c2=None):
    """Build a consistent return dict for fit_doublet (normal + fallback paths)."""
    return {
        "component_1": c1_info or fallback_c1,
        "component_2": c2_info or fallback_c2,
        "delta_chi2_per_n": None,
        "local_rms": None,
        "local_snr": None,
        "separation_check": None,
        "amp_ratio_check": None,
        "n_points": n_pts,
        "flags": flags,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# Tool 6: read_spectrum_region (tool 5 was fit_doublet)
# ---------------------------------------------------------------------------

@tool
def read_spectrum_region(
    npz_path: str,
    wl_min: float,
    wl_max: float,
    stride: int = 1,
) -> dict:
    """Read a raw slice of the cleaned spectrum for manual inspection.

    Use this to investigate anomalies: DLA damping wings, Lyα forest,
    associated absorption, or any region where fit_peak returns suspicious results.

    Parameters
    ----------
    npz_path : str
        Path to the cleaned spectrum .npz file.
    wl_min, wl_max : float
        Wavelength range of interest (Å).
    stride : int
        Downsampling step. Default 1 (no downsampling). Use 2–5 for large regions.

    Returns
    -------
    dict with keys:
        wl_range : [float, float]  — wavelength bounds (Å)
        n : int                    — number of points returned
        wl : list[float]           — wavelength array (Å, 3 d.p.)
        fl : list[float]           — flux array (4 d.p.)
    """
    data = np.load(npz_path)
    wl_full = data["wavelength"]
    flux_full = data["flux"]

    mask = (wl_full >= wl_min) & (wl_full <= wl_max)
    wl = wl_full[mask][::stride]
    fl = flux_full[mask][::stride]

    return {
        "wl_range": [wl_min, wl_max],
        "n": len(wl),
        "wl": [round(float(w), 3) for w in wl],
        "fl": [round(float(f), 4) for f in fl],
    }


# ---------------------------------------------------------------------------
# Tool 7: write_report
# ---------------------------------------------------------------------------

@tool
def write_report(file_path: str, content: str) -> dict:
    """Write a natural-language analysis report to a markdown file.

    Use this to save your findings as a human-readable report. Write the report
    BEFORE outputting the final JSON block — the JSON is for downstream
    pipelines, the report is for astronomers to read and audit.

    Parameters
    ----------
    file_path : str
        Absolute path for the output .md file.
    content : str
        Markdown content of the report.

    Returns
    -------
    dict with keys:
        path : str        — absolute path of the written file
        size_bytes : int  — file size in bytes
    """
    import os
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {
        "path": file_path,
        "size_bytes": len(content.encode("utf-8")),
    }


# ---------------------------------------------------------------------------
# Tool 8: write_lines_csv
# ---------------------------------------------------------------------------

@tool
def write_lines_csv(file_path: str, lines: list) -> dict:
    """Write CWT feature evaluation results to a CSV file.

    Each row represents one line that was evaluated (status: LIKELY, MARGINAL,
    NOT_FOUND, or MASKED). The CSV is the definitive structured record of the
    analysis — downstream pipelines read this, not the JSON in the chat response.

    Parameters
    ----------
    file_path : str
        Absolute path for the output .csv file.
    lines : list[dict]
        Each dict must have the following keys:
            name : str              — line name (e.g. "Lyα", "C IV")
            rest_wavelength : float — rest-frame wavelength in Å
            predicted_obs : float   — predicted observed wavelength at this z
            fitted_center : float   — CWT feature wavelength in Å (or null)
            fitted_center_err : float — wavelength uncertainty (or null)
            amplitude : float       — CWT amplitude (or null)
            amplitude_err : float   — amplitude uncertainty (or null)
            fitted_sigma : float    — Gaussian σ in Å (or null). Used by downstream
                                      global fit to model the line profile
            fwhm_km_s : float       — FWHM in km/s (or null)
            ridge_length : int      — CWT ridge persistence (scales spanned, or null)
            cwt_snr : float         — max SNR along the CWT ridge (or null)
            status : str            — LIKELY, MARGINAL, NOT_FOUND, or MASKED

    Returns
    -------
    dict with keys:
        path : str        — absolute path of the written file
        n_lines : int     — number of lines written
    """
    import csv
    import os

    columns = [
        "name", "rest_wavelength", "predicted_obs", "fitted_center",
        "fitted_center_err", "amplitude", "amplitude_err", "fitted_sigma",
        "fwhm_km_s", "ridge_length", "cwt_snr", "status",
    ]

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lines)

    return {
        "path": file_path,
        "n_lines": len(lines),
    }


# ---------------------------------------------------------------------------
# Tool 9: write_synthesis_csv — hypothesis-level synthesis results
# ---------------------------------------------------------------------------

@tool
def write_synthesis_csv(file_path: str, rows: list) -> dict:
    """Write the final line catalog for the confirmed redshift hypothesis.

    One row per evaluated spectral line from the best hypothesis.
    This is the definitive line catalog — downstream code should not need
    to consult individual harness CSVs.

    Parameters
    ----------
    file_path : str
        Absolute path for the output .csv file.
    rows : list[dict]
        Each dict must have the following keys:
            name : str                — line name (e.g. "Lyα", "Ca K_abs")
            rest_wavelength : float   — rest-frame λ (Å)
            predicted_obs : float     — predicted observed λ at this z
            fitted_center : float or null — CWT or fitted center (Å)
            fitted_center_err : float or null — wavelength uncertainty (Å)
            amplitude : float or null
            amplitude_err : float or null
            fitted_sigma : float or null
            fwhm_km_s : float or null
            ridge_length : int or null   — CWT only
            cwt_snr : float or null      — CWT only
            delta_chi2_per_n : float or null — fit-derived only
            local_snr : float or null        — fit-derived only
            source : str               — "CWT", "fit_peak", or "fit_doublet"
            implied_z : float or null
            status : str               — LIKELY, MARGINAL, NOT_FOUND, or MASKED
            is_anchor : bool           — true if this line anchors the systemic z
    """
    import csv
    import os

    columns = [
        "name", "rest_wavelength", "predicted_obs",
        "fitted_center", "fitted_center_err",
        "amplitude", "amplitude_err", "fitted_sigma", "fwhm_km_s",
        "ridge_length", "cwt_snr", "delta_chi2_per_n", "local_snr",
        "source", "implied_z", "status", "is_anchor",
    ]

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return {"path": file_path, "n_rows": len(rows)}


# ---------------------------------------------------------------------------
# Shared CSV utilities (used by both synthesize.py and AnalysisAuditor.py)
# ---------------------------------------------------------------------------

def _resolve_csv_path(harness_dir: str, idx: int) -> str:
    """Return the best available lines CSV for a hypothesis.

    Prefers ``{idx}_lines_cleaned.csv`` (post-FeatureAuditor) over the
    original ``{idx}_lines.csv``.  Falls back to the original when the
    cleaned version doesn't exist.
    """
    import os as _os
    cleaned = _os.path.join(harness_dir, f"single_hypothesis/{idx}_lines_cleaned.csv")
    if _os.path.exists(cleaned):
        return cleaned
    return _os.path.join(harness_dir, f"single_hypothesis/{idx}_lines.csv")


def _build_line_tables(harness_results: list, harness_dir: str) -> str:
    """Build per-hypothesis line tables from lines.csv (preferring cleaned).

    The synthesis agent needs the raw measurement data (fitted_center_err,
    fitted_sigma, fwhm_km_s, etc.) to populate write_synthesis_csv accurately.
    These tables provide every column that the CSV output requires.

    When ``{idx}_lines_cleaned.csv`` exists (post-FeatureAuditor), it is
    used instead of the original.  Features removed by FeatureAuditor are
    shown with a ~~strikethrough~~ note; flagged features show the flag text.
    """
    import csv as _csv
    import os as _os

    sections = []

    for r in harness_results:
        idx = r["hypothesis_idx"]
        csv_path = _resolve_csv_path(harness_dir, idx)
        if not _os.path.exists(csv_path):
            sections.append(
                f"### H{idx}\n\n*No lines.csv found.*\n"
            )
            continue

        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            sections.append(
                f"### H{idx}\n\n*No lines evaluated.*\n"
            )
            continue

        # Detect whether FeatureAuditor columns are present
        has_audit = any("feature_audit" in r for r in rows)

        # Identify columns present
        cols = list(rows[0].keys())

        header = (
            "| " + " | ".join(cols) + " |\n"
            "|" + "|".join(["------"] * len(cols)) + "|"
        )

        body_lines = []
        for row in rows:
            vals = []
            for c in cols:
                v = (row.get(c) or "").strip()
                if c == "feature_audit" and v == "REMOVED":
                    v = "REMOVED (noise/artifact — do NOT use)"
                elif c == "feature_audit" and v == "FLAGGED":
                    v = "FLAGGED (see feature_audit_flag)"
                vals.append(v if v else "—")
            body_lines.append("| " + " | ".join(vals) + " |")

        note = ""
        if has_audit:
            note = (
                " *(FeatureAuditor-verified: REMOVED features should be "
                "excluded from synthesis.)*"
            )

        sections.append(
            f"### H{idx}{note}\n\n"
            f"{header}\n"
            + "\n".join(body_lines)
            + "\n"
        )

    if not sections:
        return "\n## Per-Hypothesis Line Data\n\n*No line catalogs available.*\n"

    return (
        "\n## Per-Hypothesis Line Data\n\n"
        "Full measurement tables from each harness run.  Reference these when "
        "calling `write_synthesis_csv` — every column the CSV requires is "
        "present here.  When FeatureAuditor has run, the `feature_audit` and "
        "`feature_audit_flag` columns indicate which features were verified / "
        "flagged / removed.\n\n"
        + "\n".join(sections)
    )


# ---------------------------------------------------------------------------
# Tool 10: grep_kb — search knowledge base and skill files
# ---------------------------------------------------------------------------

from pathlib import Path as _Path

_BASE_DIR = _Path(__file__).resolve().parent
_KB_DIR = _BASE_DIR / "kb"

_GREP_FILES: dict[str, _Path] = {
    "kb/classification.md": _KB_DIR / "classification.md",
    "kb/ionization.md": _KB_DIR / "ionization.md",
    "kb/lines.md": _KB_DIR / "lines.md",
    "kb/composite_profile.md": _KB_DIR / "composite_profile.md",
}

_grep_cache: dict[str, str] | None = None


def _load_grep_files() -> dict[str, str]:
    """Lazy-load and cache the searchable files."""
    global _grep_cache
    if _grep_cache is None:
        _grep_cache = {}
        for name, path in _GREP_FILES.items():
            if path.exists():
                _grep_cache[name] = path.read_text(encoding="utf-8")
    return _grep_cache


@tool
def grep_kb(pattern: str, A: int = 0, B: int = 0, C: int = 0) -> dict:
    """Search the knowledge base and skill files with optional context lines.

    Use this to check what the pipeline methodology, physics rules, and
    classification criteria actually say. Essential for distinguishing
    ``skill_gap`` (rule missing from doc) from ``llm_error`` (LLM ignored
    a documented rule).

    Parameters
    ----------
    pattern : str
        grep-compatible regex pattern (case-insensitive). Examples:
        "abstain", "doublet", "Ca K", "Dn4000", "ELG|LRG|QSO", "fatal"
    A : int (default 0)
        Lines to show AFTER each match (like grep -A).
    B : int (default 0)
        Lines to show BEFORE each match (like grep -B).
    C : int (default 0)
        Lines to show before AND after each match (like grep -C).
        If both C and A/B are given, A and B take precedence.

    Returns
    -------
    dict mapping filename to a list of match blocks, where each block is
    a list of "L<num>: <text>" strings. Returns {"(no matches)": []} if
    nothing matched.
    """
    files = _load_grep_files()
    context_before = max(B, C)
    context_after = max(A, C)

    results: dict[str, list] = {}

    for fname, content in files.items():
        lines = content.split("\n")
        n_lines = len(lines)

        # Find match line numbers (1-indexed)
        match_line_nums: set[int] = set()
        for i, line in enumerate(lines, start=1):
            if re.search(pattern, line, re.IGNORECASE):
                match_line_nums.add(i)

        if not match_line_nums:
            continue

        # Expand to context windows, merge overlapping
        windows: list[tuple[int, int]] = []
        for ln in sorted(match_line_nums):
            start = max(1, ln - context_before)
            end = min(n_lines, ln + context_after)
            windows.append((start, end))

        # Merge overlapping windows
        merged: list[tuple[int, int]] = []
        for start, end in windows:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Format each block
        blocks: list[list[str]] = []
        for start, end in merged:
            block: list[str] = []
            for ln in range(start, end + 1):
                marker = ">" if ln in match_line_nums else " "
                block.append(f"L{ln}{marker}: {lines[ln - 1].strip()[:200]}")
            blocks.append(block)

        # Cap total output per file (30 match lines max, then 20 blocks max)
        total_matches = sum(
            sum(1 for line in block if line[4] == ">") for block in blocks
        )
        if total_matches > 30 or len(blocks) > 20:
            blocks = blocks[:20]

        results[fname] = blocks

    if not results:
        return {"(no matches)": []}

    return results


# ---------------------------------------------------------------------------
# Tool 11: compute_redshift
# ---------------------------------------------------------------------------

@tool
def compute_redshift(observed_wavelength: float, rest_wavelength: float) -> dict:
    """Compute the redshift implied by an observed spectral line.

    z = λ_obs / λ_rest − 1

    Use this to check whether a fitted line center is consistent with the
    verification window: compute z from the fitted center, then check if
    z ∈ [z_min, z_max].

    Parameters
    ----------
    observed_wavelength : float
        The observed wavelength in Å (e.g. the fitted center from fit_peak).
    rest_wavelength : float
        The rest-frame wavelength of the line in Å.

    Returns
    -------
    dict with keys:
        redshift : float       — implied redshift
        rest_wavelength : float
        observed_wavelength : float
        in_window : bool or null — True/False if z_min, z_max are known, else None
    """
    z = observed_wavelength / rest_wavelength - 1.0
    return {
        "redshift": round(z, 6),
        "rest_wavelength": rest_wavelength,
        "observed_wavelength": observed_wavelength,
    }


# ---------------------------------------------------------------------------
# Tool 12: detect_oii_slope_change — unresolved [O II] doublet morphology
# ---------------------------------------------------------------------------

def _detect_oii_slope_change_core(
    wavelength: np.ndarray,
    flux: np.ndarray,
    target_wl: float,
    search_window: float = 25.0,
    rise_search_n: int = 20,
    dip_threshold: float = 0.20,
    recovery_threshold: float = 2.0,
) -> dict:
    """Core detection logic for the [O II] unresolved doublet slope-change signature.

    Returns a dict with detection result and diagnostic metrics.
    """
    # 1. Locate the peak in a window around the target wavelength
    mask = (wavelength >= target_wl - search_window) & (wavelength <= target_wl + search_window)
    w = wavelength[mask]
    f = flux[mask]

    if len(w) < 10:
        return {
            "detected": False,
            "reason": "Too few points in search window",
            "n_window_points": len(w),
        }

    peak_idx = int(np.argmax(f))
    peak_wl = float(w[peak_idx])
    peak_flux = float(f[peak_idx])

    # 2. Extract the rising edge: ~5 to `rise_search_n` pixels blueward of the peak
    rise_start = max(0, peak_idx - rise_search_n)
    rise_end = peak_idx
    rise_f = f[rise_start:rise_end + 1]
    rise_w = w[rise_start:rise_end + 1]

    if len(rise_f) < 6:
        return {
            "detected": False,
            "reason": f"Rising edge too short ({len(rise_f)} pixels)",
            "peak_wl": peak_wl,
            "peak_flux": peak_flux,
        }

    # 3. Compute discrete derivative on the rising edge
    #    Exclude the last 2 pixels (too close to peak, affected by peak curvature)
    deriv = np.diff(rise_f)[:-2]
    deriv_wl = rise_w[1:len(deriv) + 1]  # wavelength at each derivative point

    if len(deriv) < 5:
        return {
            "detected": False,
            "reason": f"Derivative array too short ({len(deriv)} pixels)",
            "peak_wl": peak_wl,
            "peak_flux": peak_flux,
        }

    # 4. Find the maximum derivative (the "steep" phase)
    max_deriv_idx = int(np.argmax(deriv))
    max_deriv = float(deriv[max_deriv_idx])
    max_deriv_wl = float(deriv_wl[max_deriv_idx])

    # 5. After the max derivative, find the minimum (the "shallow" phase / dip)
    post_peak_deriv = deriv[max_deriv_idx + 1:]
    if len(post_peak_deriv) < 2:
        return {
            "detected": False,
            "reason": "Insufficient pixels after max derivative for dip",
            "peak_wl": peak_wl,
            "peak_flux": peak_flux,
            "max_deriv": max_deriv,
            "max_deriv_wl": max_deriv_wl,
        }

    dip_idx = max_deriv_idx + 1 + int(np.argmin(post_peak_deriv))
    dip_deriv = float(deriv[dip_idx])
    dip_wl = float(deriv_wl[dip_idx])

    # 6. After the dip, find the maximum derivative (the "recovery" phase)
    post_dip_deriv = deriv[dip_idx + 1:]
    recovery_deriv = float(np.max(post_dip_deriv)) if len(post_dip_deriv) > 0 else 0.0
    if len(post_dip_deriv) > 0:
        recovery_idx = dip_idx + 1 + int(np.argmax(post_dip_deriv))
        recovery_wl = float(deriv_wl[recovery_idx])
    else:
        recovery_wl = None

    # 7. Apply detection criteria
    # Two distinct [O II] signatures, both physically meaningful:
    #
    #   VALLEY: The flux REVERSES on the rising edge (dip_deriv < 0).
    #           This is the classic inter-component gap — [O II]a peaks
    #           and flux briefly drops before [O II]b takes over. This
    #           is the STRONGEST [O II] signature.
    #
    #   SLOPE-CHANGE: The flux never reverses (dip_deriv >= 0), but the
    #           rise rate has a deep local minimum — the derivative drops
    #           to a small positive value, then recovers. Detected when
    #           dip_ratio = dip_deriv / max_deriv < dip_threshold (0.20).
    #
    dip_ratio = dip_deriv / max_deriv if max_deriv > 1e-10 else 1.0
    if dip_deriv < 0:
        # Valley detected — flux reversal is unambiguous [O II] evidence
        recovery_ok = True  # negative derivative is its own confirmation
        detected = True
        signature_type = "valley"
    else:
        # Slope-change — check if the derivative dip is deep enough
        signature_type = "slope-change"
        if len(post_dip_deriv) >= 2:
            recovery_ok = recovery_deriv > dip_deriv * recovery_threshold if dip_deriv > 0 else False
            detected = dip_ratio < dip_threshold and recovery_ok
        else:
            recovery_ok = None  # too close to peak to assess
            detected = dip_ratio < dip_threshold

    # 8. Compute FWHM estimate from the peak profile
    #    Half-max on the rising side
    fwhm_A = None
    fwhm_km_s = None
    if peak_flux > 0:
        # Find continuum from local minimum before the rise
        pre_rise = f[:rise_start + 1]
        continuum = float(np.min(pre_rise)) if len(pre_rise) > 0 else float(f[0])
        half_max = continuum + (peak_flux - continuum) / 2.0

        # Left crossing
        for i in range(peak_idx - 1, 0, -1):
            if f[i] <= half_max <= f[i + 1]:
                left_wl = w[i] + (w[i + 1] - w[i]) * (half_max - f[i]) / (f[i + 1] - f[i])
                break
        else:
            left_wl = None

        # Right crossing
        for i in range(peak_idx, len(f) - 1):
            if f[i] >= half_max >= f[i + 1]:
                right_wl = w[i] + (w[i + 1] - w[i]) * (f[i] - half_max) / (f[i] - f[i + 1])
                break
        else:
            right_wl = None

        if left_wl is not None and right_wl is not None:
            fwhm_A = round(float(right_wl - left_wl), 2)
            fwhm_km_s = round(float(fwhm_A / peak_wl * 2.99792458e5), 0)

    return {
        "detected": detected,
        "signature_type": signature_type if detected else None,
        "peak_wl": peak_wl,
        "peak_flux": round(peak_flux, 4),
        "dip_ratio": round(dip_ratio, 3),
        "dip_threshold": dip_threshold,
        "max_deriv": round(max_deriv, 6),
        "max_deriv_wl": max_deriv_wl,
        "dip_deriv": round(dip_deriv, 6),
        "dip_wl": dip_wl,
        "recovery_deriv": round(recovery_deriv, 6),
        "recovery_wl": recovery_wl,
        "recovery_ok": recovery_ok,
        "fwhm_A": fwhm_A,
        "fwhm_km_s": fwhm_km_s,
        "deriv_profile": [round(float(d), 6) for d in deriv],
        "deriv_wl": [round(float(dw), 1) for dw in deriv_wl],
        "n_rise_pixels": len(rise_f),
        "n_deriv_points": len(deriv),
        "n_post_dip": len(post_dip_deriv),
    }


@tool
def detect_oii_slope_change(
    npz_path: str,
    target_wl: float,
    search_window: float = 25.0,
) -> dict:
    """Detect the [O II] unresolved doublet slope-change signature at a target wavelength.

    [O II] 3727 is a close doublet (3726.0/3729.0 Å, rest sep 2.8 Å) unresolved
    at DESI resolution. It appears as a single peak in the CWT catalog, but the
    raw spectrum carries a characteristic signature on the rising edge: the
    discrete derivative (flux[i] - flux[i-1]) shows a **dip** — it rises to a
    peak, drops significantly, then rises again before the flux peaks. A true
    single line ([O III]b, Hβ) cannot produce this pattern.

    Use this to discriminate [O II] from [O III]b when the SAME observed feature
    is claimed as [O II] by one hypothesis and [O III]b by another.

    Parameters
    ----------
    npz_path : str
        Path to the cleaned spectrum .npz file.
    target_wl : float
        The observed wavelength (Å) where [O II] is claimed.
    search_window : float
        Half-width of the search window around target_wl (default 25 Å).

    Returns
    -------
    dict with keys:
        detected : bool            — True if slope-change signature found
        peak_wl : float            — actual peak wavelength (Å)
        peak_flux : float          — peak flux
        dip_ratio : float          — min_deriv / max_deriv (< 0.30 = detected)
        max_deriv : float          — maximum derivative on rising edge
        max_deriv_wl : float       — wavelength of max derivative
        dip_deriv : float          — minimum derivative after the max
        dip_wl : float             — wavelength of the dip
        recovery_deriv : float     — max derivative after the dip
        fwhm_A : float or None     — FWHM estimate (Å)
        fwhm_km_s : float or None  — FWHM estimate (km/s)
        deriv_profile : list[float] — full derivative array on rising edge
        deriv_wl : list[float]     — wavelengths for derivative array
    """
    data = np.load(npz_path)
    return _detect_oii_slope_change_core(
        data["wavelength"], data["flux"], target_wl, search_window,
    )


# ---------------------------------------------------------------------------
# Tool 13: compute_redshift_error — redshift uncertainty from wavelength error
# ---------------------------------------------------------------------------

@tool
def compute_redshift_error(rest_wavelength: float, wavelength_error: float) -> dict:
    """Compute the redshift uncertainty σ_z from a line's wavelength error.

    σ_z = wavelength_error / rest_wavelength

    This is the propagation of the wavelength measurement uncertainty into
    redshift space.  Use the ANCHOR line (lowest-ionization confirmed line)
    for the most reliable σ_z estimate.

    Parameters
    ----------
    rest_wavelength : float
        Rest-frame wavelength of the line (Å).
    wavelength_error : float
        Wavelength measurement uncertainty (Å), from the CWT catalog
        (``wavelength_err`` column in emission/absorption CSV).

    Returns
    -------
    dict with keys:
        sigma_z : float           — redshift uncertainty (σ_z)
        rest_wavelength : float
        wavelength_error : float
    """
    sigma_z = wavelength_error / rest_wavelength
    return {
        "sigma_z": round(sigma_z, 6),
        "rest_wavelength": rest_wavelength,
        "wavelength_error": wavelength_error,
    }

