#!/usr/bin/env python3
"""Remove Lyα (4600–5000 Å) from 116_spectrum.npz, replace with interpolated noise.

- Masks flux in [4600, 5000] Å with linear interpolation between boundaries.
- Adds random noise sampled from the 6500–7300 Å quiet region.
- Snr and ivar in the masked region are replaced by linear interpolation.
- Overwrites the NPZ in place (user has a backup).
"""

import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(_SCRIPT_DIR)  # examples/
ORI_NPZ = os.path.join(_EXAMPLES_DIR, "ori_data", "visual_interpreter", "116_spectrum.npz")
NPZ_PATH = os.path.join(_EXAMPLES_DIR, "QSO_116_Lyalpha", "visual_interpreter", "116_spectrum.npz")

MASK_LO = 4600.0
MASK_HI = 5000.0
QUIET_LO = 7900.0
QUIET_HI = 8400.0
RANDOM_SEED = 42

# ── load ───────────────────────────────────────────────────────────────
print(f"Loading original: {ORI_NPZ}")
data = np.load(ORI_NPZ, allow_pickle=False)
wl = data["wavelength"]
flux = data["flux"].copy()
snr = data["snr"].copy()
ivar = data["ivar"].copy()

print(f"  Wavelength: {wl[0]:.1f} – {wl[-1]:.1f} Å, {len(wl)} px")

# ── build masks ────────────────────────────────────────────────────────
mask = (wl >= MASK_LO) & (wl <= MASK_HI)
quiet_mask = (wl >= QUIET_LO) & (wl <= QUIET_HI)

n_masked = mask.sum()
n_quiet = quiet_mask.sum()
print(f"  Masked region [{MASK_LO}, {MASK_HI}]: {n_masked} px")
print(f"  Quiet region  [{QUIET_LO}, {QUIET_HI}]: {n_quiet} px")

if n_masked == 0:
    print("WARNING: mask region contains no pixels — nothing to do")
    exit(0)

# ── uniform noise sigma from quiet-region ivar median ────────────────
quiet_ivar_median = float(np.median(ivar[quiet_mask]))
noise_sigma = 1.0 / np.sqrt(max(quiet_ivar_median, 1e-10))
print(f"  Quiet region median ivar: {quiet_ivar_median:.2f} → σ={noise_sigma:.4f}")

# ── get the boundary flux values for interpolation ────────────────
# Find the last unmasked pixel before MASK_LO and first after MASK_HI
idx_lo = np.where(wl < MASK_LO)[0][-1]
idx_hi = np.where(wl > MASK_HI)[0][0]
wl_lo, flux_lo = wl[idx_lo], flux[idx_lo]
wl_hi, flux_hi = wl[idx_hi], flux[idx_hi]
print(f"  Boundary lo: idx={idx_lo}, λ={wl_lo:.1f}, flux={flux_lo:.4f}")
print(f"  Boundary hi: idx={idx_hi}, λ={wl_hi:.1f}, flux={flux_hi:.4f}")

# ── linear interpolation ──────────────────────────────────────────
wl_masked = wl[mask]
interp = np.interp(wl_masked, [wl_lo, wl_hi], [flux_lo, flux_hi])

# ── uniform Gaussian noise from quiet-region ivar ───────────────────
rng = np.random.RandomState(RANDOM_SEED)
noise = rng.normal(0.0, noise_sigma, size=n_masked)

# ── apply ─────────────────────────────────────────────────────────
flux[mask] = interp + noise
print(f"  Interp range: [{interp.min():.4f}, {interp.max():.4f}]")
print(f"  Noise σ:      {noise_sigma:.4f} (uniform, from ivar @ {QUIET_LO}-{QUIET_HI})")

# ── snr / ivar in masked region: linear interpolation ─────────────────
for arr, name in [(snr, "snr"), (ivar, "ivar")]:
    # Clip negative snr to 0
    arr_lo = max(0.0, arr[idx_lo])
    arr_hi = max(0.0, arr[idx_hi])
    arr[mask] = np.interp(wl_masked, [wl_lo, wl_hi], [arr_lo, arr_hi])

print(f"  Snr interp range:  [{snr[mask].min():.2f}, {snr[mask].max():.2f}]")
print(f"  Ivar interp range: [{ivar[mask].min():.1f}, {ivar[mask].max():.1f}]")

# ── optional: visual check ────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(wl, data["flux"], alpha=0.4, label="original", lw=0.5)
    ax1.plot(wl, flux, alpha=0.8, label="forged", lw=0.5)
    ax1.axvspan(MASK_LO, MASK_HI, color="red", alpha=0.15, label=f"masked {MASK_LO}–{MASK_HI}")
    ax1.axvspan(QUIET_LO, QUIET_HI, color="green", alpha=0.08, label=f"quiet ivar region {QUIET_LO}–{QUIET_HI}")
    ax1.set_ylabel("Flux")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("116 spectrum — Lyα removed & replaced with interpolated noise")

    # Zoom on masked region
    zoom = (wl >= 4400) & (wl <= 5200)
    ax2.plot(wl[zoom], data["flux"][zoom], alpha=0.4, label="original", lw=0.6)
    ax2.plot(wl[zoom], flux[zoom], alpha=0.8, label="forged", lw=0.6)
    ax2.axvline(MASK_LO, color="red", ls="--", lw=1)
    ax2.axvline(MASK_HI, color="red", ls="--", lw=1)
    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel("Flux")
    ax2.legend(fontsize=7)
    ax2.set_title("Zoom 4400–5200 Å")

    png_path = os.path.splitext(NPZ_PATH)[0] + "_lyα_forged.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"\nSaved comparison plot: {png_path}")
except ImportError:
    print("\n(matplotlib not available — skipping plot)")

# ── write back ─────────────────────────────────────────────────────────
np.savez(
    NPZ_PATH,
    wavelength=data["wavelength"],
    flux=flux,
    snr=snr,
    ivar=ivar,
)
print(f"\nDone. Saved: {NPZ_PATH}")
