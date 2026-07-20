#!/usr/bin/env python3
"""Draw before/after comparison plots for the two counterfactual spectra.

- forge_lya_comparison.pdf  — Lyα removed (4600-5000 Å)
- forge_narrow_comparison.pdf — broad lines narrowed (3 regions)

Original data in black, forged data in blue (only modified regions).
Style follows plot_features() conventions from plot.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.dirname(SCRIPT_DIR)

ORI_NPZ = os.path.join(
    EXAMPLES_DIR, "ori_data", "visual_interpreter", "116_spectrum.npz"
)

LYA_NPZ = os.path.join(
    EXAMPLES_DIR, "QSO_116_Lyalpha", "visual_interpreter", "116_spectrum.npz"
)

NARROW_NPZ = os.path.join(
    EXAMPLES_DIR, "QSO_116_narrow", "visual_interpreter", "116_spectrum.npz"
)

# ── plot config ────────────────────────────────────────────────────────────
FIGSIZE = (12, 4)         # default FITS figsize from _get_figsize()
DPI = 150

# ── style (matching plot_features) ─────────────────────────────────────────
ORI_COLOR = "#555555"        # medium-dark gray (paper-friendly, less severe than pure black)
ORI_LW = 1.0
ORI_ALPHA = 0.80

FORGED_COLOR = "#e63946"    # vivid red — high contrast against gray, clear in both colour & greyscale
FORGED_LW = 0.9
FORGED_ALPHA = 0.85

SHADE_COLOR = "#d62728"    # red, subtle
SHADE_ALPHA = 0.08

GRID_ALPHA = 0.25

# ── load data ──────────────────────────────────────────────────────────────
print(f"Loading original: {ORI_NPZ}")
ori = np.load(ORI_NPZ, allow_pickle=False)
wl_ori = ori["wavelength"]
flux_ori = ori["flux"]

print(f"Loading Lyα-forged: {LYA_NPZ}")
lya = np.load(LYA_NPZ, allow_pickle=False)
flux_lya = lya["flux"]

print(f"Loading narrow-forged: {NARROW_NPZ}")
narrow = np.load(NARROW_NPZ, allow_pickle=False)
flux_narrow = narrow["flux"]


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison(ax, wl, flux_orig, flux_forged, regions, title):
    """Plot original + forged on a single axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    wl : ndarray  — wavelength
    flux_orig : ndarray  — original flux (black)
    flux_forged : ndarray  — forged flux (blue, only in modified regions)
    regions : list[tuple[float, float, str]]
        Modified regions (lo, hi, label) for shading.
    title : str
    """
    # ── original trace (full range, black) ─────────────────────────────
    ax.plot(wl, flux_orig, color=ORI_COLOR, lw=ORI_LW, alpha=ORI_ALPHA,
            label="original")

    # ── forged trace: draw original everywhere (black, dim), then
    #    overlay forged in blue ONLY in modified regions ─────────────────
    # Build a composite: start with original, then stamp forged onto
    # modified intervals.
    composite = flux_orig.copy()
    for lo, hi, _ in regions:
        mask = (wl >= lo) & (wl <= hi)
        composite[mask] = flux_forged[mask]

    # Plot the modified segments as a separate line (blue)
    for lo, hi, label in regions:
        mask = (wl >= lo) & (wl <= hi)
        ax.plot(wl[mask], flux_forged[mask],
                color=FORGED_COLOR, lw=FORGED_LW, alpha=FORGED_ALPHA,
                label="forged" if regions.index((lo, hi, label)) == 0 else None)

        # Shade the modified region
        shade_label = "modified region" if (regions.index((lo, hi, label)) == 0) else None
        ax.axvspan(lo, hi, color=SHADE_COLOR, alpha=SHADE_ALPHA * 2,
                   label=shade_label)

    # ── labels & style ─────────────────────────────────────────────────
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=8, loc="upper right")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Lyα removed
# ═══════════════════════════════════════════════════════════════════════════

LYA_REGIONS = [(4600.0, 5000.0, "Lyα 4600–5000 Å")]

fig1, ax1 = plt.subplots(1, 1, figsize=FIGSIZE)

plot_comparison(ax1, wl_ori, flux_ori, flux_lya, LYA_REGIONS,
                title="Counterfactual: Lyα removed (4600–5000 Å)")

# # Add a zoom inset
# _inset_bbox = [0.55, 0.40, 0.3, 0.3]  # [left, bottom, width, height] in figure coords
# ax_inset1 = fig1.add_axes(_inset_bbox)
# lo, hi = LYA_REGIONS[0][:2]
# zm = (wl_ori >= lo - 30) & (wl_ori <= hi + 30)
# ax_inset1.plot(wl_ori[zm], flux_ori[zm], color=ORI_COLOR, lw=ORI_LW, alpha=ORI_ALPHA)
# ax_inset1.plot(wl_ori[zm], flux_lya[zm], color=FORGED_COLOR, lw=FORGED_LW, alpha=FORGED_ALPHA)
# ax_inset1.axvspan(lo, hi, color=SHADE_COLOR, alpha=SHADE_ALPHA * 2)
# ax_inset1.set_title(f"Zoom {lo}–{hi} Å", fontsize=8)
# ax_inset1.tick_params(labelsize=7)
# ax_inset1.grid(True, alpha=GRID_ALPHA)

lya_pdf = os.path.join(SCRIPT_DIR, "forge_lya_comparison.pdf")
fig1.tight_layout()
fig1.savefig(lya_pdf, dpi=DPI)
plt.close(fig1)
print(f"Saved: {lya_pdf}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Narrow lines
# ═══════════════════════════════════════════════════════════════════════════

NARROW_REGIONS = [
    (4600.0, 5000.0, "Lyα 4600–5000"),
    (5900.0, 6250.0, "C IV 5900–6250"),
    (7300.0, 7520.0, "Mg II 7300–7520"),
]

fig2, ax2 = plt.subplots(1, 1, figsize=FIGSIZE)

plot_comparison(ax2, wl_ori, flux_ori, flux_narrow, NARROW_REGIONS,
                title="Counterfactual: broad lines narrowed to 900–1000 km/s")

# Add three zoom insets for each region
# _inset_specs = [
#     (0.05, 0.55, 0.28, 0.40, 4600, 5000, "Lyα 4600–5000 Å"),
#     (0.36, 0.55, 0.28, 0.40, 5900, 6250, "C IV 5900–6250 Å"),
#     (0.67, 0.55, 0.28, 0.40, 7300, 7520, "Mg II 7300–7520 Å"),
# ]
# for left, bottom, width, height, lo, hi, label in _inset_specs:
#     ax_in = fig2.add_axes([left, bottom, width, height])
#     zm = (wl_ori >= lo - 20) & (wl_ori <= hi + 20)
#     ax_in.plot(wl_ori[zm], flux_ori[zm], color=ORI_COLOR, lw=0.6, alpha=ORI_ALPHA)
#     ax_in.plot(wl_ori[zm], flux_narrow[zm], color=FORGED_COLOR, lw=0.6, alpha=FORGED_ALPHA)
#     ax_in.axvspan(lo, hi, color=SHADE_COLOR, alpha=SHADE_ALPHA * 2)
#     ax_in.set_title(label, fontsize=7)
#     ax_in.tick_params(labelsize=6)
#     ax_in.grid(True, alpha=GRID_ALPHA)

narrow_pdf = os.path.join(SCRIPT_DIR, "forge_narrow_comparison.pdf")
fig2.tight_layout()
fig2.savefig(narrow_pdf, dpi=DPI)
plt.close(fig2)
print(f"Saved: {narrow_pdf}")

print("\nDone.")
