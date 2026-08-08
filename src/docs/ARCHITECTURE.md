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

The deterministic v0.3 MWA runs these explicit stages:

1. normalize observed facts;
2. audit data usability, measurement reliability, and observation provenance;
3. maintain four competing hypotheses;
4. plan discriminating evidence;
5. map measurements to signed, cited evidence for every relevant hypothesis;
6. build an explicit cross-hypothesis matrix showing how one fact affects H1-H4;
7. expose opposing evidence as contradictions;
8. create a separate case file for each hypothesis with support, opposition,
   missing evidence, uncertainties, explained observations, and unexplained facts;
9. audit evidence provenance, dependence, specificity, and over-interpretation;
10. synthesize and rank hypotheses with correlation-aware aggregation and abstention;
11. calibrate confidence from score, margin, completeness, and data quality;
12. adversarially audit the decision and confirm, downgrade, or abstain;
13. audit schema/reference integrity and enforcement of scientific audit failures;
14. write JSON and Markdown reports without creating new facts.

Unavailable measurements are not negative evidence. A strong-lens decision also
requires at least one sufficiently specific geometry item; signal significance or
visual extent alone cannot pass this gate.

## Knowledge and skill separation

Files under `knowledge/` state domain interpretation, confounders, missing evidence,
and forbidden inferences for lens geometry, arc morphology, intrinsic galaxy
morphology, mergers, artifacts, and colour. Files under `skills/` define repeatable
agent procedures. Python code remains responsible for measurement, validation,
threshold enforcement, and structured outputs. Loading knowledge never creates an
observation and never overrides measurement provenance.

The three scientific audits have distinct meanings. Observation audit asks whether
the measurements are usable. Evidence audit asks whether claims are traceable,
independent enough, and proportionate to specificity. Decision audit challenges the
proposed conclusion and can only preserve, reduce, or abstain. `ReasoningAuditor`
remains a structural integrity check; a scientifically unusable case can therefore
have a valid reasoning trace when the decision audit correctly abstains.

## Intended extensions

- PSF-aware and multi-band foreground modeling;
- source-color consistency and repeated-image comparison;
- WCS/pixel-scale validation;
- merger/core and tidal-asymmetry measurements;
- calibration on labelled positive and hard-negative datasets;
- optional schema-constrained LLM explanations that may summarize cited evidence
  but cannot alter measurements, scores, gates, or audit results.
