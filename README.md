# Lens Expert Agent

An evidence-first expert reasoning system for strong gravitational lens candidate
inspection. The system separates astronomical measurement from interpretation:

```text
FITS/PNG -> VisionPipeline -> Observation JSON -> ExpertWorkflow -> grading readiness report
```

The vision layer reports measured facts and provenance. It never emits a lens
classification. The expert layer compares strong lensing against intrinsic galaxy
morphology, merger/interaction, and observation/subtraction artifacts. The former
provisional score/confidence chain has been retired. Until calibration is complete,
the system reports grading readiness rather than manufacturing A/B/C.

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
- `vision_sep_consensus.png`: exact SEP pixel unions across the deblending-parameter grid, retained as trace-only detections;
- `vision_model_chain.png`: SEP-only, SEP-plus-geometry, and route-recovered pixels shown without an implied confidence ordering;
- `vision_geometric_path_search.png`: geometric search routes and the actual pixels detected inside them;
- `vision_expert_views.png`: clean Euclid-style VIS arcsinh and MTF views at full-field and central 100×100 scales;
- `vision_manifest.json`: runtime parameters, processing steps, output paths, and input hash;
- `observation_summary_cn.md`: deterministic Chinese explanation of the measured Observation only;
- `reasoning.json`: hypotheses, evidence, contradictions, audits, and grading readiness;
- `summary.json` and `report.md`: user-facing evidence and grading-status results.

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

The current MWA uses transparent rules to construct cited evidence, but it does not
turn those provisional rule values into a lens score. `expert_score` remains null
until a labelled Euclid Q1 calibration set and an independently tested estimator
are available. The only implemented grade mapping is the Q1 paper scheme:
`A > 2.0`, `B: 1.5 < score <= 2.0`, and `C: score <= 1.5`.

Version 0.14.0 routes every visual candidate through candidate-level connected
S/N persistence, ellipse-route support, and foreground-contamination checks.
Only primary candidates enter the current expert rules; recovery candidates,
foreground-contaminated residuals, and weak trace candidates remain fully
recorded for later review. Version 0.15.0 freezes that visual path, removes the
uncalibrated decision/confidence formulas, adds an explicit Q1 grading contract,
and distinguishes `insufficient_data` from a scientifically assessed low score. Version 0.5
added WCS-aware sky coordinates, angular
scales, sky position angles, and full-field/RGB location audits. See the
[current-version note](src/docs/CURRENT_VERSION_CN.md).

PNG input runs in a conservative reduced-measurement mode because rendered images
do not carry calibrated flux or RMS maps. It cannot receive a scientific Q1 grade
before the future calibrated grading stage supports that evidence profile.

See [the architecture note](src/docs/ARCHITECTURE.md) for schemas, reasoning
constraints, and extension points.

For a comprehensive Chinese introduction, beginner-friendly operating guide,
output interpretation, and troubleshooting reference, see
[the Chinese user manual](src/docs/USER_MANUAL_CN.md).

For a copy-ready prompt that hands batch data running to a separate Codex task
without changing the frozen vision or grading rules, see
[the Chinese data-run handoff prompt](src/docs/NEW_TASK_DATA_RUN_PROMPT_CN.md).
