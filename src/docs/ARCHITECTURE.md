# LensAgent MWA Architecture

## Non-negotiable boundary

The vision layer may measure morphology, geometry, signal-to-noise, and detector
stability. It must not output `lens`, `not lens`, or a lens probability. Its only
semantic output is a versioned `ObservationDocument`.

The expert workflow receives that document and never opens image pixels. Pixel
arrays remain available as provenance artifacts for a separate scientific audit.

## Vision pipeline

One preprocessing pass produces a foreground-galaxy model, residual, residual S/N,
source mask, and segmentation. Three related measurements then share that residual:

1. connected structures surviving a grid of S/N thresholds;
2. SEP detections surviving a deblending-parameter grid;
3. arc pixels formed from a residual SEP footprint and an S/N mask.

Because all three derive from one subtraction, they carry a shared correlation
group. The decision stage aggregates a correlation group once.

FITS input uses supplied flux and RMS extensions when present. PNG input estimates
noise from pixels and uses a reduced parameter grid and radial foreground model.
This limitation is recorded in the Observation document and lowers confidence.

## Observation contract

`schema_version=1.1` records:

- input identity, format, band, shape, and nucleus;
- measured central foreground-light properties, without assigning a lens label;
- data and foreground-model quality;
- per-structure centroid, radius, PA, S/N, angular extent, axis ratio, tangential
  misalignment, and detector stability;
- atomic measurements with IDs, reliability, methods, units, and correlation groups;
- artifact paths, limitations, parameters, and pipeline version.

Version 1.1 also records scene-level tangential geometry and explicitly labelled
screening proxies for inner connectivity, secondary central concentrations,
180-degree residual asymmetry, core residual concentration, invalid pixels, and
PSF availability. Proxy measurements remain observations: they are not calibrated
morphology classifiers and cannot independently decide a hypothesis.

## Expert workflow

The deterministic v0.2 MWA runs these explicit stages:

1. normalize observed facts;
2. maintain four competing hypotheses;
3. plan discriminating evidence;
4. map measurements to signed, cited evidence for every relevant hypothesis;
5. build an explicit cross-hypothesis matrix showing how one fact affects H1-H4;
6. expose opposing evidence as contradictions;
7. create a separate case file for each hypothesis with support, opposition,
   missing evidence, uncertainties, explained observations, and unexplained facts;
8. synthesize and rank hypotheses with correlation-aware aggregation and abstention;
9. calibrate confidence from score, margin, completeness, and data quality;
10. audit references, comparison coverage, per-hypothesis analysis, ties, and stages;
11. write JSON and Markdown reports without creating new facts.

Unavailable measurements are not negative evidence. A strong-lens decision also
requires at least one sufficiently specific geometry item; signal significance or
visual extent alone cannot pass this gate.

## Intended extensions

- PSF-aware and multi-band foreground modeling;
- source-color consistency and repeated-image comparison;
- WCS/pixel-scale validation;
- merger/core and tidal-asymmetry measurements;
- calibration on labelled positive and hard-negative datasets;
- optional schema-constrained LLM explanations that may summarize cited evidence
  but cannot alter measurements, scores, gates, or audit results.
