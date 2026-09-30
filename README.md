# Lens Expert Agent

Current version: **0.30.0 delivery snapshot** (Observation schema **1.18**, additive optional fields).
See [the delivery and reproducibility note](src/docs/DELIVERY_V0300_CN.md).
See [the desktop workflow revision](src/docs/SCIENTIFIC_REVISION_V0290_UI_CN.md).
This trial uses quality-controlled calibrated colour relationships in the A/B confidence boundary and adds measured arc unions; it is not Q1-catalogue calibrated.
See [the deterministic revision contract](src/docs/DETERMINISTIC_V0210_CN.md).
No multimodal model or generative restoration is used. New measurements are merged
before one rule-based decision; the former max-of-two-grades pass has been removed.
The following older centre-selection guide remains useful: see the
[Chinese change and validation guide](src/docs/V0190_CENTER_STRUCTURE_CHANGE_CN.md)
for target-centred foreground selection, local curved-shape measurements, and
structure-specific comparison evidence. Default inputs should be target-centred;
off-centre targets can specify both `--center-x` and `--center-y`.

An evidence-first expert reasoning system for strong gravitational lens candidate
inspection. The system separates astronomical measurement from interpretation:

```text
FITS/PNG -> VisionPipeline -> signal-reality handoff -> Observation JSON -> configuration-aware ExpertAssessmentAgent -> DecisionAudit -> report
```

The vision layer reports measured facts and provenance. It never emits a lens
classification. `ExpertAssessmentAgent` is the only active scientific interpretation
layer: it builds one traceable evidence dossier, compares competing explanations, and
issues one explicitly uncalibrated expert vote category. `DecisionAudit` only
checks integrity and replays the deterministic decision from source measurements.
Replay is not a second vote or independent scientific validation.

## Install

```powershell
python -m pip install -e ".[dev,ui]"
```

## Desktop workflow

```powershell
lens-agent ui
```

The desktop application imports one FITS file or a folder, runs a persistent
background queue, groups completed objects by A/B/C/X or explicit abstention,
shows diagnostic artifacts and decision evidence, and launches the native-pixel
foreground editor for a versioned full rerun. Dark and light themes are available.
Scientific outputs remain the source of truth; the UI index is rebuildable metadata.

To launch the standalone colour/geometry structure annotation UI:

```powershell
lens-agent-structure-review "<dataset-root>" --output outputs_colour_structure_annotations
```

## Run a complete analysis

```powershell
lens-agent analyze "example/102018667_NEG587746535508057234" --output outputs
```

To choose the foreground centre and masks manually before fitting:

```powershell
lens-agent analyze "example/target" --interactive-foreground --output outputs
```

To reopen a completed object, edit its foreground, and write a complete rerun:

```powershell
lens-agent foreground-review "outputs/target" --output outputs/human_foreground_rerun
```

The accepted `foreground_override.json` can also be replayed without opening the UI:

```powershell
lens-agent analyze "example/target" --foreground-override "outputs/target/foreground_override.json" --output outputs/replay
```

The target directory contains:

- `observation.json`: versioned measurements only;
- `vision_artifacts.npz`: pixel-level provenance;
- `vision_segmentation.fits`: layered native-grid segmentation with promoted residual-structure labels, the original field SEP segmentation, candidate and handoff maps, foreground models, residuals, RMS, WCS, and a structure catalogue;
- `vision_diagnostic.png`: input/model/residual/mask QA;
- `vision_field_context.png`: full FITS field with crop, search annulus, WCS N/E, and structure IDs;
- `vision_target_selection.png`: the 2--4 object pre-fit routing audit; its scores never enter grading;
- `vision_source_boundary.png`: fitted closed radial-minimum boundary, binary fit inclusion map, selected foreground model, and residual S/N;
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
- `source_observation.json` and `reasoning_observation.json`: source and canonical decision inputs;
- `run_manifest.json` and `code_snapshot/`: input/code hashes, archived source, parameters and dependency versions.

Use `summary.json.published_grade` (A/B/C/unknown) as the public result.
`expert_assessment.vote_category` is the internal provisional vote; audit failure
blocks publication as ABC. Internal X is preserved as `publication_reason=non_lens_vote_X`,
distinct from insufficient data. Q1 catalogue calibration is still pending.

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

The current MWA uses transparent configuration hypotheses rather than additive
sub-scores. It issues one categorical single-agent vote. Deprecated numeric anchor and
score fields remain null; they are not a substitute for multi-expert aggregation.
`official_expert_score` remains null until a labelled Euclid Q1 calibration set and
independently tested estimator are available. The formal catalogue layer implements
the Q1 Engine A cuts `A > 2.0`, `B: 1.5 < score <= 2.0`, and
`C: 1.0 < score <= 1.5` (paper section 8.3); scores <= 1 are `below_C_cut`,
not an inferred official aggregate X class.

Version 0.14.0 introduced candidate-level connected S/N persistence, ellipse-route
provenance, and foreground-contamination checks.
Version 0.15.0 freezes that visual path, removes the
uncalibrated decision/confidence formulas, adds an explicit Q1 grading contract,
and distinguishes `insufficient_data` from a scientifically assessed low score.
Version 0.16.0 freezes that same visual layer, consolidates all primary reasoning in
`ExpertAssessmentAgent`, adds a lightweight independent `DecisionAudit`, and emits
provisional vote anchors without interpreting them as probabilities.
Version 0.17.0 makes the expert decision category-first, permits audited contiguous
vote sets such as B-to-A, relaxes overly strict double-image/common-radius gates, and
prevents tidal asymmetry alone from acting as an equally strong non-lens explanation.
Version 0.18.0 repairs the evidence handoff exposed by real A-sample runs: stable
off-route candidates now enter reasoning with complete measurements, near-core or
fallback-model cautions no longer erase independently surviving signals, and only
affirmative comparison-model erasure isolates foreground contamination. The expert
layer evaluates annular/partial rings, tangential arcs, fold and cusp arcs, radial arcs,
compact multiple images, and asymmetric configurations before casting one A/B/C/X vote.
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
