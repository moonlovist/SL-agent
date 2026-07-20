"""
CWT-based feature detector using Ricker (Mexican hat) wavelet.
Multi-scale ridge detection replaces the Gaussian consensus voting in
simple_feature_finder.

The Ricker wavelet (second derivative of Gaussian) is a natural matched
filter for Gaussian-like spectral lines. At each scale the CWT coefficients
form a scale-space representation where:
  - Positive coefficients -> emission-like features
  - Negative coefficients -> absorption-like features

Ridge detection in this scale-space identifies features that persist across
multiple scales, providing a principled confidence metric.
"""

import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.355


# ---------------------------------------------------------------------------
# Local noise estimation (same as simple_feature_finder)
# ---------------------------------------------------------------------------

def _local_noise_mad(signal, half_window=50):
    """Sliding-window MAD noise estimate, robust against line features."""
    n = len(signal)
    noise = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        noise[i] = 1.4826 * np.median(np.abs(signal[lo:hi] - np.median(signal[lo:hi])))
    return np.maximum(noise, np.median(noise) * 0.1)


# ---------------------------------------------------------------------------
# CWT ridge detection core
# ---------------------------------------------------------------------------

def _cwt_ridge_detection(signal, wavelength, scales, snr_thresh=5.0,
                         min_ridge_length=2, tol_pix=10):
    """
    Multi-scale CWT ridge detection using the Ricker wavelet.

    Parameters
    ----------
    signal : ndarray
        For emission pass ``residual = flux - continuum``.
        For absorption pass ``-residual`` (troughs become peaks).
    wavelength : ndarray
    scales : ndarray
        Wavelet scales in pixel units (log-spaced recommended).
    snr_thresh : float
        Minimum |CWT| / local_noise ratio.
    min_ridge_length : int
        Minimum number of scales a feature must persist across.
    tol_pix : int
        Max pixel drift allowed when linking detections across adjacent scales.

    Returns
    -------
    list of dict with keys: wavelength, amplitude, sigma_init, fwhm_pix,
                             ridge_length, snr
    """
    n = len(signal)
    if n < 10:
        return []

    noise = _local_noise_mad(signal)
    mean_dwave = float(np.median(np.diff(wavelength))) if len(wavelength) > 1 else 1.0

    # ---- CWT with Ricker (Mexican hat) wavelet via PyWavelets ----
    cwt_mat, _ = pywt.cwt(signal, scales, 'mexh')  # (n_scales, n_pixels)

    # ---- Find local maxima at each scale ----
    detections = []  # (pixel_idx, scale_idx, cwt_value, snr)
    for i in range(len(scales)):
        row = cwt_mat[i, :]
        peaks_idx, _ = find_peaks(row, width=1)
        for pi in peaks_idx:
            local_snr = float(row[pi]) / noise[pi] if noise[pi] > 0 else 0.0
            if local_snr >= snr_thresh:
                detections.append((int(pi), i, float(row[pi]), local_snr))

    if not detections:
        return []

    # ---- Build ridges by linking detections across adjacent scales ----
    detections.sort(key=lambda x: (x[1], x[0]))  # sort by (scale_idx, pixel)

    ridges = []  # list of [(pixel, scale_idx, val, snr), ...]
    for det in detections:
        pi, si, val, snr = det
        matched = False
        for ridge in ridges:
            last_pi, last_si, _, _ = ridge[-1]
            if last_si == si - 1 and abs(pi - last_pi) <= tol_pix:
                ridge.append(det)
                matched = True
                break
            # Same-scale collision (should be rare with width=1):
            # keep the higher-SNR detection
            if last_si == si and abs(pi - last_pi) <= tol_pix:
                if snr > ridge[-1][3]:
                    ridge[-1] = det
                matched = True
                break
        if not matched:
            ridges.append([det])

    # ---- Extract representative parameters from each ridge ----
    results = []
    for ridge in ridges:
        if len(ridge) < min_ridge_length:
            continue

        pixels = [d[0] for d in ridge]
        scales_detected = [scales[d[1]] for d in ridge]

        rep_idx = int(np.median(pixels))
        rep_wave = float(wavelength[rep_idx])
        rep_amp = float(signal[rep_idx])

        med_scale = float(np.median(scales_detected))
        fwhm_pix = max(med_scale * SIGMA_TO_FWHM, 2.0)
        sigma_init = med_scale * mean_dwave * 0.85

        max_snr = max(d[3] for d in ridge)

        results.append({
            'wavelength': rep_wave,
            'amplitude': rep_amp,
            'sigma_init': sigma_init,
            'fwhm_pix': fwhm_pix,
            'ridge_length': len(ridge),
            'snr': max_snr,
        })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_features_cwt(wavelength, flux, snr_thresh=5.0, min_ridge_length=2,
                      n_scales=24, min_scale=1.0, max_scale=80.0,
                      verbose=True):
    """
    CWT-based emission and absorption line detection.

    Parameters
    ----------
    wavelength : ndarray
    flux : ndarray
        Original spectrum flux (NOT residual).
    snr_thresh : float
        Minimum CWT SNR to keep a detection.
    min_ridge_length : int
        Minimum number of scales for a valid ridge (≥ 2).
    n_scales : int
        Number of logarithmically-spaced wavelet scales.
    min_scale, max_scale : float
        Scale range in pixel units.  min_scale ≈ narrowest line FWHM / 2.355.
    verbose : bool

    Returns
    -------
    records_emission, records_absorption : list[dict]
    """
    n = len(flux)
    if n < 10:
        return [], []

    # Rough continuum via wide median filter -> residual
    median_width = max(51, n // 20)
    if median_width % 2 == 0:
        median_width += 1
    baseline = median_filter(flux, size=median_width)
    residual = flux - baseline  # emission > 0, absorption < 0

    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
    mean_dwave = np.median(np.diff(wavelength))

    # Emission: positive features in residual
    em_candidates = _cwt_ridge_detection(
        residual, wavelength, scales,
        snr_thresh=snr_thresh, min_ridge_length=min_ridge_length)

    # Absorption: positive features in -residual (troughs -> peaks)
    ab_candidates = _cwt_ridge_detection(
        -residual, wavelength, scales,
        snr_thresh=snr_thresh, min_ridge_length=min_ridge_length)

    def _build_records(candidates, feature_type):
        records = []
        for c in candidates:
            amp = c['amplitude']
            if feature_type == 'absorption':
                amp = -abs(amp)
            fwhm_a = c['fwhm_pix'] * mean_dwave
            fwhm_km_s = fwhm_a / c['wavelength'] * 3e5 if c['wavelength'] > 0 else 0
            records.append({
                'wavelength': c['wavelength'],
                'wavelength_err': float(mean_dwave),
                'FWHM_A': float(fwhm_a),
                'FWHM_km_s': float(fwhm_km_s),
                'amplitude': float(amp),
                'amplitude_err': float(amp * 0.15),
                'width_class': ('narrow' if fwhm_km_s < 1000 else
                                'intermediate' if fwhm_km_s < 2000 else 'broad'),
                'feature_type': feature_type,
                'ridge_length': c['ridge_length'],
                'snr': c['snr'],
                'left_neighbor': 'None',
                'right_neighbor': 'None',
                'is_pseudo_peak': False,
                'pseudo_reason': '',
                'covered_troughs': 0,
                'trough_centers': '',
            })
        return records

    records_em = _build_records(em_candidates, 'emission')
    records_ab = _build_records(ab_candidates, 'absorption')

    if verbose:
        print(f"[cwt_finder] CWT ridge detection "
              f"(SNR≥{snr_thresh}, ridge≥{min_ridge_length}, "
              f"scales={n_scales}×[{min_scale:.0f}..{max_scale:.0f}]px): "
              f"{len(records_em)} peaks, {len(records_ab)} troughs")
        if records_em:
            rl_dist = {}
            for r in records_em:
                rl = r['ridge_length']
                rl_dist[rl] = rl_dist.get(rl, 0) + 1
            print(f"  peak ridge_length distribution: {rl_dist}")
        if records_ab:
            rl_dist = {}
            for r in records_ab:
                rl = r['ridge_length']
                rl_dist[rl] = rl_dist.get(rl, 0) + 1
            print(f"  trough ridge_length distribution: {rl_dist}")

    return records_em, records_ab


# ---------------------------------------------------------------------------
# Drop-in replacement for run_simple_feature_detection
# ---------------------------------------------------------------------------

def run_cwt_feature_detection(
    output_dir, file_name, wavelength, flux,
    ivar=None, effective_snr=None,
    n_iterations=1,
    absorption_detection_params=None,
    emission_detection_params=None,
    verbose=True,
):
    """
    CWT-based feature detection -- drop-in replacement for
    ``run_simple_feature_detection``.  Interface fully compatible.

    Returns
    -------
    dict with keys: df_emission, df_absorption, records_emission,
                    records_absorption, wave_remaining, flux_remaining
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    snr_thresh = 5.0
    min_ridge_length = 2
    n_scales = 24
    min_scale = 1.0
    max_scale = 80.0

    if emission_detection_params:
        snr_thresh = emission_detection_params.get('snr_thresh', snr_thresh)
        min_ridge_length = emission_detection_params.get('min_ridge_length', min_ridge_length)
        n_scales = emission_detection_params.get('n_scales', n_scales)
        min_scale = emission_detection_params.get('min_scale', min_scale)
        max_scale = emission_detection_params.get('max_scale', max_scale)

    records_em, records_ab = find_features_cwt(
        wavelength, flux,
        snr_thresh=snr_thresh,
        min_ridge_length=min_ridge_length,
        n_scales=n_scales,
        min_scale=min_scale,
        max_scale=max_scale,
        verbose=verbose)

    # ---- Build DataFrames (columns aligned with simple_feature_finder output) ----
    def _to_df(records):
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df['amplitude_rank'] = df['amplitude'].rank(ascending=False, method='min').astype(int)
        df = df.sort_values('wavelength').reset_index(drop=True)
        return df

    df_em = _to_df(records_em)
    df_ab = _to_df(records_ab)

    def _save_csv(df, suffix):
        if len(df) == 0:
            return
        out_cols = ['index', 'amplitude_rank', 'wavelength', 'wavelength_err',
                    'FWHM_A', 'FWHM_km_s', 'amplitude', 'amplitude_err',
                    'integrated_flux', 'flux_at_center',
                    'ridge_length', 'snr',
                    'width_class', 'feature_type',
                    'left_neighbor', 'right_neighbor',
                    'is_pseudo_peak', 'pseudo_reason',
                    'covered_troughs', 'trough_centers']
        df_out = pd.DataFrame(index=range(len(df)))
        for col in out_cols:
            if col in df.columns:
                df_out[col] = df[col]
            elif col == 'index':
                df_out[col] = range(len(df))
            elif col == 'amplitude_rank':
                df_out[col] = df['amplitude'].rank(ascending=False, method='min').astype(int)
            elif col == 'integrated_flux':
                df_out[col] = df['amplitude'] * df['FWHM_A'] / SIGMA_TO_FWHM * np.sqrt(2 * np.pi)
            elif col == 'flux_at_center':
                df_out[col] = df['amplitude']
            else:
                df_out[col] = df.get(col, '')
        path = os.path.join(output_dir, "visual_interpreter", f"{file_name}_{suffix}.csv")
        df_out.to_csv(path, index=False)

    _save_csv(df_em, 'emission')
    _save_csv(df_ab, 'absorption')

    return {
        'df_emission': df_em,
        'df_absorption': df_ab,
        'records_emission': records_em,
        'records_absorption': records_ab,
        'wave_remaining': wavelength.copy(),
        'flux_remaining': flux.copy(),
    }
