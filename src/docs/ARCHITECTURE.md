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

`schema_version=1.4` records:

- input identity, format, band, shape, and nucleus;
- measured central foreground-light properties, without assigning a lens label;
- data and foreground-model quality;
- per-structure centroid, radius, PA, S/N, angular extent, axis ratio, tangential
  misalignment, and detector stability;
- atomic measurements with IDs, reliability, methods, units, and correlation groups;
- artifact paths, limitations, parameters, and pipeline version.

Version 1.2 adds celestial WCS provenance, pixel scale, sky coordinates for the
measured center and residual structures, sky position angles, and explicit N/E
display vectors. FITS diagnostics use an astronomical lower-origin display, and
the full-field/RGB context figures map every reported structure back to its source
location. These fields improve traceability; they do not decide whether a structure
is a gravitational lens.

Version 1.4 adds an audit-only, PSF-convolved fixed-geometry multi-Gaussian
foreground model. It compares the primary subtraction with an independent model,
records residual correlation and the survival fraction of already measured
structure pixels, and writes a six-panel comparison figure. The comparison model
is explicitly excluded from detection and decision; it exposes model-sensitive
residuals without silently changing the official measurement path.

Version 1.3 originally recorded a fixed Euclid Q1 SLDE cutout contract, and renames the scientific extraction
product to an unclassified residual structure, and adds clean VIS arcsinh/MTF
expert views. The legacy `arc_mask` NPZ key remains only as a compatibility alias.
Pipeline 0.7.1 removes that fixed-size/12-extension validation from new runs.
The legacy schema field remains optional only so older Observation files still load.

Version 1.1 introduced scene-level tangential geometry and explicitly labelled
screening proxies for inner connectivity, secondary central concentrations,
180-degree residual asymmetry, core residual concentration, invalid pixels, and
PSF availability. Proxy measurements remain observations: they are not calibrated
morphology classifiers and cannot independently decide a hypothesis.

## Expert workflow

The deterministic v0.4 MWA runs these explicit stages:

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

## Visual trace bundle

Version 0.5 renders the actual intermediates used by the vision pipeline rather
than reconstructing illustrative approximations. The threshold detector and its
diagnostic share `threshold_masks`; the SEP figure consumes the returned consensus
detections; and the extraction figure consumes the final SEP footprint and arc
mask. A per-run manifest records the input SHA-256 digest, runtime parameters,
processing order, numerical-array meanings, and output paths.

The active scientific channel is the raw foreground-subtracted residual divided by
RMS. No wavelet denoising is applied. PNG diagnostics are human-readable audit
views, while `vision_artifacts.npz` remains the numerical provenance record.

FITS celestial WCS is retained in memory and validated at the measured center.
Observation schema 1.2 records ICRS center coordinates, pixel scale, native N/E
vectors, per-structure sky coordinates, angular separation, and sky PA. SEP major
axes are transformed through the local WCS before tangential misalignment is
calculated. FITS diagnostics use the native pixel grid with `origin=lower` and
explicit N/E arrows; no unrecorded reprojection is performed.

Two sample-dependent location audits connect measurements back to the source data.
`vision_field_context.png` shows the full scientific field, crop, annulus, and IDs.
When a sibling RGB preview passes orientation-correlation validation,
`vision_rgb_context.png` records the chosen mapping and overlays the same geometry.

The fixed architecture diagram is maintained once in `VISION_PROCESS_FLOW_CN.md`.
Per-object output directories contain only sample-dependent diagnostic images; they
do not duplicate a static flowchart.

## Intended extensions

- PSF-aware and multi-band foreground modeling;
- source-color consistency and repeated-image comparison;
- WCS/pixel-scale validation;
- merger/core and tidal-asymmetry measurements;
- calibration on labelled positive and hard-negative datasets;
- optional schema-constrained LLM explanations that may summarize cited evidence
  but cannot alter measurements, scores, gates, or audit results.
