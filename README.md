# Lens Expert Agent

An evidence-first expert reasoning system for strong gravitational lens candidate
inspection. The system separates astronomical measurement from interpretation:

```text
FITS/PNG -> frozen VisionPipeline -> Observation JSON -> ExpertAssessmentAgent -> DecisionAudit -> report
```

The vision layer reports measured facts and provenance. It never emits a lens
classification. `ExpertAssessmentAgent` is the only active scientific interpretation
layer: it builds one traceable evidence dossier, compares competing explanations, and
issues an explicitly uncalibrated expert-vote anchor or interval. `DecisionAudit` only
checks the assessment for integrity mistakes; it does not repeat the hypothesis analysis.

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
- `reasoning.json`: evidence dossier, hypothesis comparison, provisional assessment, and decision audit;
- `summary.json` and `report.md`: user-facing evidence and provisional grading results.

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

The current MWA uses transparent holistic pattern gates rather than additive
sub-scores. It issues a categorical single-agent vote or a contiguous set of plausible
votes. Numeric anchor fields are retained only as deprecated compatibility encodings;
`official_expert_score` remains null until a labelled Euclid Q1 calibration set and
independently tested estimator are available. The formal catalogue layer implements
only the Q1 paper cuts explicitly defined for `A > 2.0` and
`B: 1.5 < score <= 2.0`; lower aggregate scores are `below_B_cut`, not an invented
continuous X/C boundary.

Version 0.14.0 routes every visual candidate through candidate-level connected
S/N persistence, ellipse-route support, and foreground-contamination checks.
Only primary candidates can establish a point vote. A strong audited held candidate
may widen the plausible vote set without becoming equal-weight primary evidence;
other recovery, contaminated, and trace candidates remain recorded for review.
Version 0.15.0 freezes that visual path, removes the
uncalibrated decision/confidence formulas, adds an explicit Q1 grading contract,
and distinguishes `insufficient_data` from a scientifically assessed low score.
Version 0.16.0 freezes that same visual layer, consolidates all primary reasoning in
`ExpertAssessmentAgent`, adds a lightweight independent `DecisionAudit`, and emits
provisional vote anchors without interpreting them as probabilities.
Version 0.17.0 makes the expert decision category-first, permits audited contiguous
vote sets such as B-to-A, relaxes overly strict double-image/common-radius gates, and
prevents tidal asymmetry alone from acting as an equally strong non-lens explanation.
Version 0.5 added WCS-aware sky coordinates, angular
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
