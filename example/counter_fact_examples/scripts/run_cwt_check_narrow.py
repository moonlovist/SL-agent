#!/usr/bin/env python3
"""Run CWT feature detection on the narrow-forged QSO_116 spectrum.

Uses the VisualInterpreter's ``find_features_cwt`` with custom parameters
and saves emission / absorption catalogs to the QSO_116_narrow directory.
"""

import os, sys
import numpy as np
import pandas as pd

# Ensure the project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.AstroAgent.agents.multi_agents.utils.cwt_feature_finder import (
    find_features_cwt,
    SIGMA_TO_FWHM,
)

# ── paths ──────────────────────────────────────────────────────────────────
NPZ_PATH = os.path.join(
    PROJECT_ROOT,
    "data/fake/fake_data/QSO_116_narrow/visual_interpreter/116_spectrum.npz",
)
OUT_DIR = os.path.dirname(NPZ_PATH)  # .../visual_interpreter/

EMISSION_CSV = os.path.join(OUT_DIR, "emission_narrow_forged.csv")
ABSORPTION_CSV = os.path.join(OUT_DIR, "absorption_narrow_forged.csv")

# ── CWT parameters (user-specified) ────────────────────────────────────────
SNR_THRESH = 8.0
MIN_RIDGE_LENGTH = 4
N_SCALES = 24
MIN_SCALE = 1.0
MAX_SCALE = 80.0

# ── load ───────────────────────────────────────────────────────────────────
print(f"Loading: {NPZ_PATH}")
data = np.load(NPZ_PATH, allow_pickle=False)
wl = data["wavelength"]
flux = data["flux"]
ivar = data["ivar"]
print(f"  λ: {wl[0]:.1f} – {wl[-1]:.1f} Å, {len(wl)} px")
print(f"  flux: [{flux.min():.3f}, {flux.max():.3f}]")
print()

# ── run CWT ────────────────────────────────────────────────────────────────
print(f"CWT parameters:")
print(f"  snr_thresh        = {SNR_THRESH}")
print(f"  min_ridge_length  = {MIN_RIDGE_LENGTH}")
print(f"  n_scales          = {N_SCALES}")
print(f"  min_scale         = {MIN_SCALE}")
print(f"  max_scale         = {MAX_SCALE}")
print()

records_em, records_ab = find_features_cwt(
    wl, flux,
    snr_thresh=SNR_THRESH,
    min_ridge_length=MIN_RIDGE_LENGTH,
    n_scales=N_SCALES,
    min_scale=MIN_SCALE,
    max_scale=MAX_SCALE,
    verbose=True,
)

# ── format and save ────────────────────────────────────────────────────────
OUT_COLS = [
    "index", "amplitude_rank", "wavelength", "wavelength_err",
    "FWHM_A", "FWHM_km_s", "amplitude", "amplitude_err",
    "integrated_flux", "flux_at_center",
    "ridge_length", "snr",
    "width_class", "feature_type",
    "left_neighbor", "right_neighbor",
    "is_pseudo_peak", "pseudo_reason",
    "covered_troughs", "trough_centers",
]


def _save_formatted_csv(records, path, suffix_label):
    """Replicate the CSV format from run_cwt_feature_detection."""
    if not records:
        print(f"  No {suffix_label} features — writing empty CSV")
        pd.DataFrame(columns=OUT_COLS).to_csv(path, index=False)
        return

    df = pd.DataFrame(records)
    df["amplitude_rank"] = df["amplitude"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("wavelength").reset_index(drop=True)

    df_out = pd.DataFrame(index=range(len(df)))
    for col in OUT_COLS:
        if col in df.columns:
            df_out[col] = df[col]
        elif col == "index":
            df_out[col] = range(len(df))
        elif col == "amplitude_rank":
            df_out[col] = df["amplitude"].rank(ascending=False, method="min").astype(int)
        elif col == "integrated_flux":
            df_out[col] = df["amplitude"] * df["FWHM_A"] / SIGMA_TO_FWHM * np.sqrt(2 * np.pi)
        elif col == "flux_at_center":
            df_out[col] = df["amplitude"]
        else:
            df_out[col] = df.get(col, "")

    df_out.to_csv(path, index=False)
    print(f"  Saved {len(df_out)} {suffix_label} features → {path}")


_save_formatted_csv(records_em, EMISSION_CSV, "emission")
_save_formatted_csv(records_ab, ABSORPTION_CSV, "absorption")

# ── summary ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Summary")
print("=" * 60)

for label, recs in [("Emission", records_em), ("Absorption", records_ab)]:
    if not recs:
        print(f"  {label}: 0 features")
        continue
    widths = [r["width_class"] for r in recs]
    fwhms = [r["FWHM_km_s"] for r in recs]
    snrs = [r["snr"] for r in recs]
    print(f"  {label}: {len(recs)} features")
    print(f"    Width classes: {dict((w, widths.count(w)) for w in sorted(set(widths)))}")
    print(f"    FWHM (km/s):   [{min(fwhms):.0f}, {max(fwhms):.0f}]  mean={np.mean(fwhms):.0f}")
    print(f"    SNR:           [{min(snrs):.1f}, {max(snrs):.1f}]  mean={np.mean(snrs):.1f}")
    print()

print("Done.")
