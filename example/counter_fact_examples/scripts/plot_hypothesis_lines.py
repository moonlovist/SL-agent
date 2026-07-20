#!/usr/bin/env python3
"""Plot a single hypothesis line inventory in the 3-panel ``plot_features`` style.

- Panel 1: spectrum + Chebyshev continuum
- Panel 2: residual + KEEP absorption lines (negative amplitude)
- Panel 3: residual + KEEP emission lines (positive amplitude)

Only features with ``feature_audit == "KEEP"`` are drawn.

Chebyshev continuum follows the VI.py ``run_continuum_fitting_masked`` logic:
feature regions (centre ± 3σ) are masked, then a Chebyshev polynomial is
fit to the remaining data via specutils / astropy.

Usage::

    python .claude/tmp/plot_hypothesis_lines.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import sys

# ── make project imports available ─────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from AstroAgent.agents.multi_agents.utils.feature_finder_precise import (
    SIGMA_TO_FWHM,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev

# ── paths ──────────────────────────────────────────────────────────────────
NPZ_PATH = os.path.join(
    _PROJECT_ROOT, "data/output_/4/visual_interpreter/4_spectrum.npz"
)
CSV_PATH = os.path.join(
    _PROJECT_ROOT, "data/output_/4/single_hypothesis/14_lines_cleaned.csv"
)
OUT_DIR = os.path.join(_PROJECT_ROOT, ".claude/tmp")

# ── plot config ────────────────────────────────────────────────────────────
FIGSIZE = (16, 8)     # 2-panel layout (3 panels in plot_features is (16, 12))
DPI = 150

# Spectrum
SPEC_COLOR = "black"
SPEC_LW = 0.8
SPEC_ALPHA = 0.75

# Continuum
CONT_COLOR = "red"
CONT_LS = "--"
CONT_LW = 1.5
CONT_ALPHA = 0.9

# Residual
RESID_COLOR = "steelblue"
RESID_LW = 0.7
RESID_ALPHA = 0.75

# Emission lines
EM_COLOR = "darkorange"
EM_LW = 1.8
EM_ALPHA = 0.85
EM_VLINE_LW = 0.8
EM_VLINE_ALPHA = 0.4

# Absorption lines
AB_COLOR = "tomato"
AB_LW = 1.8
AB_ALPHA = 0.85
AB_VLINE_LW = 0.8
AB_VLINE_ALPHA = 0.4

GRID_ALPHA = 0.25
LABEL_FONTSIZE = 7
LEGEND_FONTSIZE = 9

# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════

print(f"Spectrum: {NPZ_PATH}")
sp = np.load(NPZ_PATH, allow_pickle=False)
wl = sp["wavelength"]
flux = sp["flux"]
print(f"  λ: {wl[0]:.1f} – {wl[-1]:.1f} Å, {len(wl)} px")

print(f"CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} rows total")
print(f"  feature_audit values: {df['feature_audit'].value_counts().to_dict()}")

# ── filter KEEP ────────────────────────────────────────────────────────────
keep = df[df["feature_audit"] == "KEEP"].copy()
print(f"  KEEP: {len(keep)} features")

# Split into emission / absorption by amplitude sign
em_keep = keep[keep["amplitude"] > 0].copy()
ab_keep = keep[keep["amplitude"] < 0].copy()
print(f"    Emission (amp>0):  {len(em_keep)}")
print(f"    Absorption (amp<0): {len(ab_keep)}")

# ═══════════════════════════════════════════════════════════════════════════
# Chebyshev continuum (following VI.py run_continuum_fitting_masked)
# ═══════════════════════════════════════════════════════════════════════════

# Build feature mask: exclude centre ± 3σ for all KEEP features
fit_mask = np.ones(len(wl), dtype=bool)

for _, row in keep.iterrows():
    centre = row["fitted_center"]
    sigma = row["fitted_sigma"]  # already in Å
    if pd.isna(centre) or pd.isna(sigma) or sigma <= 0:
        continue
    lo = centre - 3 * sigma
    hi = centre + 3 * sigma
    fit_mask &= ~((wl >= lo) & (wl <= hi))

n_masked = int((~fit_mask).sum())
print(f"\nChebyshev fit: {n_masked} px masked, {int(fit_mask.sum())} px used for fit")

if fit_mask.sum() < 20:
    print("  Too few points — falling back to full-spectrum fit")
    fit_mask = np.ones(len(wl), dtype=bool)

# ── Chebyshev fit (numpy, equivalent to VI.py's specutils approach) ────
# Normalise wavelength to [-1, 1] for Chebyshev domain
wl_min, wl_max = float(wl[0]), float(wl[-1])
wl_norm = 2 * (wl - wl_min) / (wl_max - wl_min) - 1  # in [-1, 1]
wl_norm_fit = wl_norm[fit_mask]
flux_fit = flux[fit_mask]

# Auto-select degree via BIC (simplified from select_chebyshev_degree)
chebyshev_degree = 5
best_bic = np.inf
for deg in range(1, 11):
    try:
        c = Chebyshev.fit(wl_norm_fit, flux_fit, deg=deg)
        fitted = c(wl_norm_fit)
        resid_f = flux_fit - fitted
        rss = float(np.sum(resid_f ** 2))
        n_pts = len(flux_fit)
        n_par = deg + 1
        bic = n_pts * np.log(rss / n_pts + 1e-10) + n_par * np.log(n_pts)
        if bic < best_bic:
            best_bic = bic
            chebyshev_degree = deg
    except Exception:
        continue

print(f"  Chebyshev degree: {chebyshev_degree} (BIC auto-selected from 1–10)")

# Final fit
coeffs = Chebyshev.fit(wl_norm_fit, flux_fit, deg=chebyshev_degree)
continuum_flux = coeffs(wl_norm)
residual = flux - continuum_flux

print(f"  Continuum range:  [{continuum_flux.min():.3f}, {continuum_flux.max():.3f}]")
print(f"  Residual range:   [{residual.min():.3f}, {residual.max():.3f}]")

# ═══════════════════════════════════════════════════════════════════════════
# Plot — 3 panels (matching plot_features layout)
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
ax1, ax2 = axes

# ── Panel 1: spectrum + continuum ──────────────────────────────────────
ax1.plot(wl, flux, color=SPEC_COLOR, lw=SPEC_LW, alpha=SPEC_ALPHA, label="Spectrum")
ax1.plot(wl, continuum_flux,
         color=CONT_COLOR, linestyle=CONT_LS, lw=CONT_LW, alpha=CONT_ALPHA,
         label=f"Continuum (Chebyshev deg={chebyshev_degree})")
ax1.set_ylabel("Flux")
ax1.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
ax1.grid(True, alpha=GRID_ALPHA)
ax1.set_title("AGN example, z=0.238")

# ── Panel 2: residual + emission lines ─────────────────────────────────
res_y_min, res_y_max = residual.min(), residual.max()
text_y = res_y_max * 0.92

ax2.plot(wl, residual, color=RESID_COLOR, lw=RESID_LW, alpha=RESID_ALPHA,
         label="Residual")
ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

for _, row in em_keep.iterrows():
    centre = row["fitted_center"]
    sigma = row["fitted_sigma"]
    amp = row["amplitude"]  # positive
    if pd.isna(centre) or pd.isna(sigma) or sigma <= 0:
        continue

    fwhm = sigma * SIGMA_TO_FWHM
    w_local = np.linspace(centre - 3.5 * sigma, centre + 3.5 * sigma, 200)
    gauss = amp * np.exp(-0.5 * ((w_local - centre) / sigma) ** 2)
    ax2.plot(w_local, gauss, color=EM_COLOR, linewidth=EM_LW, alpha=EM_ALPHA)

    ax2.axvline(centre, color=EM_COLOR, linestyle="--",
                linewidth=EM_VLINE_LW, alpha=EM_VLINE_ALPHA)

    name = row.get("name", "")
    label_text = f"{centre:.1f}" if pd.isna(name) or name == "" else f"{name}\n{centre:.1f}"
    ax2.text(centre, text_y, label_text,
             rotation=90, verticalalignment="top", horizontalalignment="center",
             fontsize=LABEL_FONTSIZE, color=EM_COLOR, alpha=0.8)

ax2.set_xlabel("Wavelength (Å)")
ax2.set_ylabel("Flux")
ax2.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
ax2.grid(True, alpha=GRID_ALPHA)

# ── save ───────────────────────────────────────────────────────────────
plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, "hypothesis_14_keep_features.pdf")
fig.savefig(out_pdf, dpi=DPI)
plt.close(fig)
print(f"\nSaved: {out_pdf}")
print("Done.")
