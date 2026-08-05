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

`schema_version=1.0` records:

- input identity, format, band, shape, and nucleus;
- data and foreground-model quality;
- per-structure centroid, radius, PA, S/N, angular extent, axis ratio, tangential
  misalignment, and detector stability;
- atomic measurements with IDs, reliability, methods, units, and correlation groups;
- artifact paths, limitations, parameters, and pipeline version.

## Expert workflow

The deterministic MWA runs these explicit stages:

1. normalize observed facts;
2. maintain four competing hypotheses;
3. plan discriminating evidence;
4. map measurements to signed, cited evidence;
5. expose opposing evidence as contradictions;
6. rank hypotheses with correlation-aware aggregation and abstention;
7. calibrate confidence from score, margin, completeness, and data quality;
8. audit references, ties, competition, and required stages;
9. write JSON and Markdown reports without creating new facts.

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
