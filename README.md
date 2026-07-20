中文 README 文件见 [README in Chinese](./README_Chinese.md).

 **A related paper is in preparation.** 

# FORMA

An LLM-powered multi-agent system for human-like analysis of one-dimensional astronomical spectra.

## Overview

FORMA uses large language models (LLMs) to perform human-like astrophysical inference on 1D spectra, specifically:
- **Source classification** (galaxy: LRG / ELG, QSO)
- **Redshift estimation** for QSOs

The system mimics the cognitive workflow of a human astronomer:
1. **Visual interpretation** of the spectrum (feature detection via CWT, continuum fitting)
2. **Redshift hypothesis generation** through Redrock (DESI's official redshift fitter)
3. **Multi-agent evaluation** — each hypothesis is independently analysed, then cross-verified through adversarial LLM review
4. **Synthesis** of a final report with calibrated confidence

The pipeline is currently configured to use the following model via API:
- **Text reasoning**: `deepseek-v4-pro`

> For detailed module documentation, architecture diagrams, and pipeline topology, see [`.repo_info/index.html`](.repo_info/index.html).

---

## Examples

The [`example/`](example/) directory contains two ready-to-run cases:

| Directory | Description |
|-----------|-------------|
| [`basic/`](example/basic/) | Two standard DESI spectra (`4.fits`, `116.fits`) — a quick way to test the pipeline. |
| [`counter_fact_examples/`](example/counter_fact_examples/) | Forged spectra for stress-testing the auditor. Lyα is removed in one, narrow-line QSO impostor in the other. See its [README](example/counter_fact_examples/README.md) for details. |

---

## Quick Start with Docker (Recommended)

The easiest way to run FORMA is via Docker. The image bundles Python 3.12, all dependencies, and the Redrock redshift fitter.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2+

### 1. Configure your `.env` file

```bash
cp .env_example .env
# Edit .env: set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
# Do NOT set INPUT_DIR / OUTPUT_DIR — Docker handles those via volumes
```

### 2. CLI mode (single or batch)

```bash
# Single FITS file
INPUT_DIR_HOST=/path/to/your/fits \
OUTPUT_DIR_HOST=/path/to/results \
docker compose run -e FILE_NAME=QSO_116 -e RUN_MODE=s forma-cli

# Batch (all .fits files in the input directory)
INPUT_DIR_HOST=/path/to/your/fits \
OUTPUT_DIR_HOST=/path/to/results \
docker compose run -e RUN_MODE=b -e FILE_BATCH_HEADER=QSO_ -e FILE_BATCH_START=0 -e FILE_BATCH_END=100 forma-cli
```

### 3. WebUI mode

```bash
docker compose up forma-web
# Open http://localhost:7860 in your browser
```

> **Note:** The first `docker compose build` takes 5–10 minutes (Redrock C extension compilation). Subsequent builds use cached layers.

---

## Manual Installation

If you prefer to run FORMA without Docker:

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env_example .env
```

Key variables (see `.env_example` for full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | *required* | API key for the text LLM |
| `LLM_BASE_URL` | *required* | Base URL for the LLM API |
| `LLM_MODEL` | *required* | Model name (e.g. `deepseek-v4-pro`) |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `LLM_MAX_TOKENS` | API default | Max tokens per LLM response |
| `LLM_THINKING` | `disabled` | Thinking mode (`enabled` / `disabled` / `none`) |
| `REDROCK` | `true` | Enable Redrock redshift fitter |
| `RR_TEMPLATE_DIR` | — | Path to Redrock templates |
| `ARCHETYPE_DIR` | — | Path to archetype files (optional) |
| `USE_ARCHETYPES` | `true` | Use archetypes in fitting |
| `NMINIMA` | `9` | Redshift minima to explore |
| `NNEAREST` | `2` | Nearest archetypes |
| `OMP_NUM_THREADS` | `1` | OpenMP threads for Redrock |
| `RUN_MODE` | `s` | `s` = single, `b` = batch |
| `INPUT_DIR` | — | Directory with input FITS files |
| `OUTPUT_DIR` | — | Directory for results |
| `FILE_NAME` | — | FITS file name (without extension, single mode) |
| `ARM_NAME` | `B,R,Z` | Camera arm names |
| `ARM_WAVELENGTH_RANGE` | `3600-5800,...` | Wavelength range per arm |
| `CWT_SNR_THRESH` | `5.0` | CWT SNR threshold (higher = stricter) |
| `CWT_MIN_RIDGE_LENGTH` | `4` | Min scales for a valid feature |
| `CWT_N_SCALES` | `24` | Number of wavelet scales |
| `CWT_MIN_WIDTH` | `1.0` | Narrowest line to detect |
| `CWT_MAX_WIDTH` | `80.0` | Widest line to detect |
| `HARNESS_CONCURRENCY` | `3` | Parallel hypothesis evaluations |
| `MAX_TRIES` | `3` | Retry attempts on connection error |
| `RETRY_DELAY` | `180` | Retry delay in seconds |

### 3. Redrock Installation

```bash
git clone https://github.com/desihub/redrock
cd redrock
git clone https://github.com/desihub/redrock-templates py/redrock/templates
pip install -e .
pip install desiutil
pip install desispec
```

Set `RR_TEMPLATE_DIR` in `.env` to the template path (default: `redrock/py/redrock/templates`).

#### (Optional) Archetypes Mode

```bash
git clone https://github.com/desihub/redrock-archetypes.git
```

Set `ARCHETYPE_DIR` in `.env` to the cloned directory. Verify with `rrdesi --help`.

### 4. Run

```bash
python scripts/main.py
```

Results are saved to `OUTPUT_DIR/{file_name}/`.

---

## Output Files

For an input FITS file `{your_file_name}.fits`, results are saved to `{OUTPUT_DIR}/{your_file_name}/`:

```
{file_name}/
├── {name}_in_brief.json                 # Machine-readable summary (type, z, confidence, lines)
├── final_report.md                      # Human-readable final report
├── {name}_redshift_hypotheses.txt       # Redrock hypothesis listing
├── {name}_hypothesis_analysis.txt       # Synthesis verdict (JSON)
├── {name}_redrock/                      # Redrock external fitting results
│   ├── {name}_redrock.fits
│   └── {name}_rrdetails.h5
├── visual_interpreter/                  # Feature detection outputs
│   ├── {name}_continuum.png             # Fitted continuum
│   ├── {name}_features.png              # Detected features visualization
│   ├── {name}_residual_spectrum.png     # Residual (data − continuum)
│   ├── {name}_emission.csv              # Emission line table
│   ├── {name}_absorption.csv            # Absorption line table
│   └── {name}_spectrum.npz              # Extracted spectrum (NumPy)
├── single_hypothesis/                   # Per-hypothesis analysis
│   ├── {N}_report.md
│   ├── {N}_features.png
│   ├── {N}_lines.csv
│   ├── {N}_lines_cleaned.csv
│   └── {N}_stream.md
├── hypothesis_synthesis/                # Cross-hypothesis comparison
│   ├── report.md
│   ├── catalog.csv
│   └── stream.md
├── feature_auditor/                     # Feature-level audit
├── result_auditor/                      # Result-level audit
└── report_writer/                       # Report writer log
```

### Key Outputs

| File | Description |
|------|-------------|
| `{name}_in_brief.json` | Machine-readable summary: type, redshift, confidence, identified lines |
| `final_report.md` | Human-readable report with full analysis |
| `hypothesis_synthesis/report.md` | Summary of all hypotheses and final verdict |
| `hypothesis_synthesis/catalog.csv` | All hypotheses with line measurements |
| `visual_interpreter/{name}_emission.csv` | Detected emission features |
| `visual_interpreter/{name}_absorption.csv` | Detected absorption features |

---

## License

This project is for research and educational purposes.
