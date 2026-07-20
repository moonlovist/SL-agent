# Counterfactual Examples

Two counterfactual experiments probing the robustness of the **Result Auditor**
(AnalysisAuditor Stage B).  Each example starts from a real QSO spectrum (SDSS
plate 4055, fiber 116) and applies a targeted forgery, then asks the auditor to
re-evaluate the same upstream hypothesis-synthesis verdict on the modified data.

### Important: upstream data were manually aligned with the forgery

These examples test the **Result Auditor in isolation** — they ask whether the
auditor can catch a problem that upstream stages have *already failed to catch*.

The **spectrum NPZ** was forged, and the upstream outputs —
`hypothesis_synthesis/stream.md`, `feature_auditor/verdict.json`, and
`single_hypothesis/*_lines_cleaned.csv` — were **manually edited to be
consistent with the forged spectrum**.  They simulate a scenario where the
upstream pipeline *did* run on the forged data, made mistakes (e.g.
misidentified the narrow peaks as genuine broad lines, or failed to notice
a missing Lyα), and passed a flawed result down to the Result Auditor.

We deliberately do **not** re-run the full upstream pipeline on the forged
spectrum, for two reasons:

1. **The forged spectrum might not survive upstream.**  If VisualInterpreter or
   HypothesisAnalyst catches the forgery first, the hypothesis would never reach
   the Result Auditor — and we would be testing the wrong stage.

2. **We are stress-testing the auditor, not the pipeline.**  The auditor's job
   is to be the last line of defence — to spot errors that earlier stages
   missed.  Manually constructing the upstream data to be consistent with the
   forgery creates the exact scenario the auditor is designed for: a
   convincing-looking but incorrect analysis is on its desk.

## Directory layout

```
counter_fact_examples/
├── README.md
├── ori_data/                        # Original (unmodified) spectrum
│   └── visual_interpreter/
│       └── 116_spectrum.npz         #   wavelength, flux, snr, ivar
├── scripts/                         # Forge & analysis utilities
│   ├── forge_remove_lya.py          #   Remove Lyα (4600-5000 Å)
│   ├── forge_narrow_lines.py        #   Narrow QSO broad lines to ~700 km/s
│   ├── run_cwt_check_narrow.py      #   Re-run CWT on forged spectrum
│   ├── plot_forged_comparison.py    #   Before/after comparison PDFs
│   └── plot_hypothesis_lines.py     #   Plot hypothesis line inventory
├── QSO_116_Lyalpha/                 # Example 1: Lyα removed
│   ├── run_result_auditor.py        #   Single-run auditor script
│   ├── run_result_auditor_batch.py  #   Batch runner (N=100 default)
│   ├── final_report.md              #   Report Writer output
│   ├── visual_interpreter/
│   │   └── 116_spectrum.npz         #   Forged spectrum (wavelength, flux, snr, ivar)
│   ├── hypothesis_synthesis/
│   │   └── stream.md                #   Synthesis verdict JSON
│   ├── feature_auditor/
│   │   └── verdict.json             #   Feature Auditor verdicts
│   ├── single_hypothesis/
│   │   └── *_lines_cleaned.csv      #   Per-hypothesis line catalogs (cleaned)
│   ├── result_auditor/
│   │   └── auditor_verdict.json     #   Single-run auditor output
│   └── result_auditor_batch/
│       └── _summary.json            #   Batch aggregate (verdict counts, confidence, etc.)
└── QSO_116_narrow/                  # Example 2: broad lines narrowed
    └── ... (same structure)
```

## Example 1 — Lyα removed (`QSO_116_Lyalpha`)

**Forgery**: The Lyα emission region (4600–5000 Å) is replaced with linear
interpolation between interval endpoints, plus Gaussian noise drawn from the
spectrum's quiet region (7900–8400 Å ivar median).  The original broad Lyα
line vanishes entirely.

**Expected auditor behaviour**: The synthesis verdict claims a QSO at z≈0.238
with Lyα as a KEEP line.  The auditor should detect that Lyα is absent and
either flag it as NOT_FOUND or downgrade the confidence.

### Quick start

```bash
cd counter_fact_examples/QSO_116_Lyalpha
python run_result_auditor.py          # single run (~30-60 s)
python run_result_auditor_batch.py -n 10   # 10-run batch
```

## Example 2 — Broad lines narrowed (`QSO_116_narrow`)

**Forgery**: Three wavelength intervals covering the key QSO broad lines
(4600–5000 Å Lyα, 5900–6250 Å C IV, 7300–7520 Å Mg II) are flattened to a
linear-interpolated continuum.  Narrow triangular peaks (FWHM ≈ 550–700 km/s)
are then placed at the original line positions.  The resulting emission lines
are far too narrow for a Type 1 QSO (FWHM > 2000 km/s expected).

**Expected auditor behaviour**: The auditor should flag the narrow line widths
as inconsistent with a QSO classification, referencing the `classification.md`
knowledge-base rule that any of Lyα/C IV/C III]/Mg II missing in the observed
range is fatal.

### Quick start

```bash
cd counter_fact_examples/QSO_116_narrow
python run_result_auditor.py          # single run (~30-60 s)
python run_result_auditor_batch.py -n 10   # 10-run batch
```

## Re-forging the spectra

The forge scripts in `counter_fact_examples/scripts/` regenerate the NPZ files from the
original (unmodified) spectrum at `counter_fact_examples/ori_data/visual_interpreter/116_spectrum.npz`.  Run them
from the project root:

```bash
# Set environment variables for your LLM provider
export LLM_MODEL="your-model"
export LLM_API_KEY="your-key"
export LLM_BASE_URL="https://api.example.com"

# Re-forge Lyα-removed spectrum
python counter_fact_examples/scripts/forge_remove_lya.py

# Re-forge narrow-line spectrum
python counter_fact_examples/scripts/forge_narrow_lines.py
```

## Prerequisites

- Python ≥ 3.10
- `numpy`, `pandas`, `matplotlib`, `pywt` (PyWavelets) — for forging & CWT
- An LLM provider with OpenAI-compatible API (set `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`)
- Project source on `PYTHONPATH` (the `run_result_auditor*.py` scripts handle this automatically)
