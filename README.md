# Lens Expert Agent

An evidence-first expert reasoning system for strong gravitational lens candidate
inspection. The system separates astronomical measurement from interpretation:

```text
FITS/PNG -> VisionPipeline -> Observation JSON -> ExpertWorkflow -> audited report
```

The vision layer reports measured facts and provenance. It never emits a lens
classification. The expert layer compares strong lensing against intrinsic galaxy
morphology, merger/interaction, and observation/subtraction artifacts. When the
evidence is weak or tied, it abstains.

## Install

```powershell
python -m pip install -e ".[dev]"
```

## Run a complete analysis

```powershell
lens-agent analyze "example/102018667_NEG587746535508057234" --output outputs
```

The target directory contains:

- `observation.json`: versioned measurements only;
- `vision_artifacts.npz`: pixel-level provenance;
- `vision_diagnostic.png`: input/model/residual/mask QA;
- `vision_field_context.png`: full FITS field with crop, search annulus, WCS N/E, and structure IDs;
- `vision_rgb_context.png`: validated RGB-to-FITS orientation mapping and the same overlays;
- `vision_snr_threshold_process.png`: every S/N threshold mask and survival layer;
- `vision_sep_consensus.png`: stable detections across the SEP parameter grid;
- `vision_arc_extraction.png`: the exact pixel-intersection path used for arc measurements;
- `vision_manifest.json`: runtime parameters, processing steps, output paths, and input hash;
- `observation_summary_cn.md`: deterministic Chinese explanation of the measured Observation only;
- `reasoning.json`: hypotheses, evidence, contradictions, decision, confidence, and layered audits;
- `summary.json` and `report.md`: final user-facing results.

## Run reasoning without image access

```powershell
lens-agent reason outputs/<object-id>/observation.json --output outputs/reasoning_only
```

This command demonstrates the architectural boundary: the expert workflow consumes
only the Observation document and does not read FITS, PNG, or NPZ pixels.

## Test

```powershell
python -m pytest
```

The current MWA uses transparent domain rules for reproducible scoring. Scores are
comparative reasoning aids, not calibrated lens probabilities. Multi-band color,
PSF-aware modeling, and dataset-level calibration are explicit future extensions.

Version 0.5 adds WCS-aware sky coordinates, angular scales, sky position angles,
and full-field/RGB location audits. See the [current-version note](src/docs/CURRENT_VERSION_CN.md).

PNG input runs in a conservative reduced-measurement mode because rendered images
do not carry calibrated flux or RMS maps. A high-S/N elongated PNG residual cannot
trigger a lens decision unless a lens-specific geometry measurement also passes.

See [the architecture note](src/docs/ARCHITECTURE.md) for schemas, reasoning
constraints, and extension points.

For a comprehensive Chinese introduction, beginner-friendly operating guide,
output interpretation, and troubleshooting reference, see
[the Chinese user manual](src/docs/USER_MANUAL_CN.md).
