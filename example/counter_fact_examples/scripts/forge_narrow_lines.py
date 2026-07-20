#!/usr/bin/env python3
r"""Narrow the broad QSO lines for counterfactual testing.

Strategy
--------
Part 1 — Flatten three broad-line intervals with linear interpolation
between interval endpoints, plus ivar-based noise.

Part 2 — Construct narrow triangular peaks at the original line centres.
For a triangular profile the FWHM equals the half-base width, so the
saddle points (where the profile meets the continuum) are at
λ_peak ± Δλ_FWHM, where Δλ_FWHM = λ_peak × v_FWHM / c.

    Intervals flattened          Narrow peaks added
    ──────────────────          ──────────────────
    4600 – 5000 Å               4767.2 Å, FWHM 700  km/s  (Lyα   →  ~14.3 Å)
    5900 – 6250 Å               6072.8 Å, FWHM 620  km/s  (C IV  →  ~16.6 Å)
    7300 – 7520 Å               7479.2 Å, FWHM 550  km/s  (Mg II →  ~18.7 Å)

The NPZ is modified **in place** inside the QSO_116_narrow directory (a
separate copy from QSO_116, kept for this experiment).
"""

import os
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(_SCRIPT_DIR)
ORI_NPZ = os.path.join(_EXAMPLES_DIR, "ori_data", "visual_interpreter", "116_spectrum.npz")
NPZ_PATH = os.path.join(_EXAMPLES_DIR, "QSO_116_narrow", "visual_interpreter", "116_spectrum.npz")

# ── constants ──────────────────────────────────────────────────────────────
C_KMS = 299792.458  # speed of light in km/s
RANDOM_SEED = 42

# Quiet region for ivar-based noise estimation
QUIET_LO = 7900.0
QUIET_HI = 8400.0

# Part 1: intervals to flatten (replace broad line with continuum + noise)
INTERVALS: list[tuple[float, float, str]] = [
    (4600.0, 5000.0, "Lyα region"),
    (5900.0, 6250.0, "C IV region"),
    (7300.0, 7520.0, "Mg II region"),
]

# Part 2: narrow peaks to construct (wl Å, FWHM km/s, label)
PEAKS: list[dict] = [
    {"wl": 4767.2, "fwhm_kms": 300, "label": "Lyα narrow"},
    {"wl": 6072.8, "fwhm_kms": 300, "label": "C IV narrow"},
    {"wl": 7479.2, "fwhm_kms": 300, "label": "Mg II narrow"},
]

# ── load ───────────────────────────────────────────────────────────────────
print(f"Loading original: {ORI_NPZ}")
data = np.load(ORI_NPZ, allow_pickle=False)
wl = data["wavelength"]  # shape (7608,)
n_px = len(wl)

# Work on copies; keep originals for boundary values and ivar-based noise
flux_orig = data["flux"].copy()
snr_orig = data["snr"].copy()
ivar_orig = data["ivar"].copy()

flux = flux_orig.copy()
snr = snr_orig.copy()
ivar = ivar_orig.copy()

rng = np.random.RandomState(RANDOM_SEED)

print(f"  Wavelength: {wl[0]:.1f} – {wl[-1]:.1f} Å, {n_px} px")
print(f"  Flux range: [{flux.min():.3f}, {flux.max():.3f}]")

# ── compute uniform noise sigma from quiet region ivar ─────────────────
quiet_mask = (wl >= QUIET_LO) & (wl <= QUIET_HI)
quiet_ivar_median = float(np.median(ivar_orig[quiet_mask]))
noise_sigma = 1.0 / np.sqrt(max(quiet_ivar_median, 1e-10))
print(f"  Quiet region [{QUIET_LO}, {QUIET_HI}]: median ivar={quiet_ivar_median:.2f} → σ={noise_sigma:.4f}")
print()

# ════════════════════════════════════════════════════════════════════════════
# Part 1 — Flatten broad-line intervals
# ════════════════════════════════════════════════════════════════════════════

for lo, hi, label in INTERVALS:
    mask = (wl >= lo) & (wl <= hi)
    n_masked = mask.sum()

    # Boundary indices (pixels just outside the interval)
    idx_lo = int(np.where(wl < lo)[0][-1])
    idx_hi = int(np.where(wl > hi)[0][0])
    wl_lo, fl_lo = float(wl[idx_lo]), float(flux_orig[idx_lo])
    wl_hi, fl_hi = float(wl[idx_hi]), float(flux_orig[idx_hi])

    print(f"── {label}: [{lo}, {hi}] Å, {n_masked} px")
    print(f"   Boundary lo: idx={idx_lo}, λ={wl_lo:.1f}, flux={fl_lo:.4f}")
    print(f"   Boundary hi: idx={idx_hi}, λ={wl_hi:.1f}, flux={fl_hi:.4f}")

    wl_masked = wl[mask]

    # ── linear interpolation between interval endpoints ─────────────────
    continuum = np.interp(wl_masked, [wl_lo, wl_hi], [fl_lo, fl_hi])

    # ── uniform noise from quiet-region ivar ─────────────────────────────
    noise = rng.normal(0.0, noise_sigma, size=n_masked)

    flux[mask] = continuum + noise
    print(f"   Continuum range: [{continuum.min():.4f}, {continuum.max():.4f}]")
    print(f"   Noise σ:         {noise_sigma:.4f} (uniform, from ivar @ {QUIET_LO}-{QUIET_HI})")

    # ── interpolate snr and ivar linearly ───────────────────────────────
    for arr, orig in [(snr, snr_orig), (ivar, ivar_orig)]:
        v_lo = max(0.0, float(orig[idx_lo]))
        v_hi = max(0.0, float(orig[idx_hi]))
        arr[mask] = np.interp(wl_masked, [wl_lo, wl_hi], [v_lo, v_hi])
        arr_name = "snr" if arr is snr else "ivar"
        print(f"   {arr_name} interp: [{v_lo:.2f}, {v_hi:.2f}] → [{arr[mask].min():.2f}, {arr[mask].max():.2f}]")

    print()

# ════════════════════════════════════════════════════════════════════════════
# Part 2 — Construct narrow triangular peaks
# ════════════════════════════════════════════════════════════════════════════

print("── Constructing narrow peaks ──\n")

for p in PEAKS:
    wl_peak = float(p["wl"])
    fwhm_kms = float(p["fwhm_kms"])
    label = p["label"]

    # FWHM in wavelength: Δλ = λ × v/c
    dlam = wl_peak * fwhm_kms / C_KMS
    wl_left = wl_peak - dlam
    wl_right = wl_peak + dlam

    # Nearest pixel indices
    idx_peak = int(np.argmin(np.abs(wl - wl_peak)))
    idx_left = int(np.argmin(np.abs(wl - wl_left)))
    idx_right = int(np.argmin(np.abs(wl - wl_right)))

    # Guard against degenerate indices
    if idx_left >= idx_right:
        print(f"WARNING: {label} — idx_left={idx_left} >= idx_right={idx_right}, skipping")
        continue
    if idx_left >= idx_peak or idx_right <= idx_peak:
        print(f"WARNING: {label} — peak {idx_peak} not between saddles [{idx_left}, {idx_right}], skipping")
        continue

    peak_mask = (wl >= wl[idx_left]) & (wl <= wl[idx_right])
    n_peak = peak_mask.sum()

    # Continuum level at saddle points (from Part 1 result)
    cont_left = float(flux[idx_left])
    cont_right = float(flux[idx_right])

    # Peak = original flux value
    peak_val = float(flux_orig[idx_peak])

    wl_seg = wl[peak_mask]

    # Piecewise linear: left-saddle → peak  and  peak → right-saddle
    tri = np.empty(n_peak, dtype=np.float64)

    # Left segment (λ_left ≤ λ ≤ λ_peak)
    left_seg = wl_seg <= wl[idx_peak]
    tri[left_seg] = np.interp(
        wl_seg[left_seg],
        [wl[idx_left], wl[idx_peak]],
        [cont_left, peak_val],
    )

    # Right segment (λ_peak < λ ≤ λ_right)
    right_seg = wl_seg > wl[idx_peak]
    tri[right_seg] = np.interp(
        wl_seg[right_seg],
        [wl[idx_peak], wl[idx_right]],
        [peak_val, cont_right],
    )

    # ── uniform noise from quiet-region ivar ─────────────────────────────
    noise = rng.normal(0.0, noise_sigma, size=n_peak)

    flux[peak_mask] = tri + noise
    # Recompute SNR from new flux and uniform noise level
    snr[peak_mask] = np.abs(tri) / max(noise_sigma, 1e-10)

    print(f"   {label}: λ_peak={wl[idx_peak]:.3f} Å (idx={idx_peak})")
    print(f"     Δλ_FWHM = {dlam:.2f} Å  ({fwhm_kms:.0f} km/s)")
    print(f"     Saddles: λ_left={wl[idx_left]:.3f}, λ_right={wl[idx_right]:.3f}")
    print(f"     Continuum: {cont_left:.4f} / {cont_right:.4f}   Peak: {peak_val:.4f}")
    print(f"     Noise σ:       {noise_sigma:.4f} (uniform)")
    print(f"     Resulting flux range: [{flux[peak_mask].min():.4f}, {flux[peak_mask].max():.4f}]")
    print()

# ════════════════════════════════════════════════════════════════════════════
# Visual check
# ════════════════════════════════════════════════════════════════════════════

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    # ── Full spectrum overview ──
    ax = axes[0]
    ax.plot(wl, flux_orig, alpha=0.35, label="original", lw=0.4)
    ax.plot(wl, flux, alpha=0.85, label="narrow-forged", lw=0.4)
    for lo, hi, _label in INTERVALS:
        ax.axvspan(lo, hi, color="red", alpha=0.10)
    ax.set_ylabel("Flux")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("QSO_116 — broad lines replaced with narrow triangular peaks")

    # ── Zoom on each interval ──
    zoom_regions = [
        (4400, 5200, "Lyα 4600–5000 Å"),
        (5700, 6450, "C IV 5900–6250 Å"),
        (7100, 7720, "Mg II 7300–7520 Å"),
    ]
    for ax, (z_lo, z_hi, z_label) in zip(axes[1:], zoom_regions):
        zm = (wl >= z_lo) & (wl <= z_hi)
        ax.plot(wl[zm], flux_orig[zm], alpha=0.35, label="original", lw=0.5)
        ax.plot(wl[zm], flux[zm], alpha=0.85, label="forged", lw=0.5)

        # Mark the flattened interval
        lo, hi = INTERVALS[zoom_regions.index((z_lo, z_hi, z_label))][:2]
        ax.axvline(lo, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.axvline(hi, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.set_ylabel("Flux")
        ax.legend(fontsize=7)
        ax.set_title(f"Zoom {z_label}")

    axes[-1].set_xlabel("Wavelength (Å)")

    plt.tight_layout()
    png_path = os.path.splitext(NPZ_PATH)[0] + "_narrow_forged.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved comparison plot: {png_path}")

except ImportError:
    print("(matplotlib not available — skipping plot)")

# ════════════════════════════════════════════════════════════════════════════
# Write back
# ════════════════════════════════════════════════════════════════════════════

np.savez(
    NPZ_PATH,
    wavelength=data["wavelength"],
    flux=flux,
    snr=snr,
    ivar=ivar,
)
print(f"Done. Saved: {NPZ_PATH}")
