# LensAgent MWA Architecture

## Experimental executable verification branch

`feature/deepseek-judgment` adds an opt-in, observation-only DeepSeek workflow
inspired by [FORMA](https://arxiv.org/abs/2607.06128). Pixel extraction and the
audited rule vote remain the baseline. The model generates competing physical
hypotheses, deterministic tools validate references and fill original measured values,
and a second model role adjudicates the dossier. One revision in total is allowed
after citation/schema failure. Failure or unusable data leads to abstention or an
explicit unavailable model vote, never an invented official score.

```text
FITS → numerical vision → canonical Observation
                           ├─ rule assessment → DecisionAudit → published_grade
                           └─ anonymized measurements → hypothesis analyst
                                → citation/value checks → model adjudicator
                                → citation/quality gates → experimental llm_grade
                                      ↓
                            report + desktop comparison
Reference catalogue → post-inference evaluation only
```

Responsibilities are separated across `core/llm_client.py` (transport/settings),
`schemas/llm_reasoning.py` (typed hypotheses/votes),
`tools/evidence_verification.py` (executable checks),
`agents/llm_assessment.py` (bounded verification workflow), and
`core/evaluation.py` (label comparison). The model does not see file paths,
target identity, sky coordinates, rule votes or catalogue labels. The final report
retains the baseline and experimental vote separately. Two calls to one model are
two roles, not independent experts. Tool checks verify references, not scientific
accuracy; lens-model closure is explicitly marked unperformed.
Every canonical structure must receive a cited review. Current scene statistics
and multiband measurements enter the handoff; obsolete extraction-era statistics
are excluded. Explicit model-supplied numbers are checked with relative/absolute
tolerance `1e-12`. These checks address integrity and evidence coverage, rather
than turning model self-confidence into a scientific probability. Both model roles
receive the versioned local lens/arc/colour/artifact knowledge bundle. Its hash is
archived alongside input and prompt hashes; model citation choices are enumerated
from the actual non-missing measurements.

The sections below describe the deterministic baseline and historical revisions.

## v0.21 experimental superseding contract

Read [DETERMINISTIC_V0210_CN.md](DETERMINISTIC_V0210_CN.md). FITS pixels are
measured by deterministic numerical algorithms, not a multimodal model. Complete
candidate measurements are canonicalized before ONE expert-rule decision.
DecisionAudit replays that decision from source measurements; it is not a second
vote or independent physical test. Failed audits cannot publish ABC. Sections
below describing v0.19 or earlier are historical, not alternative runtime modes.

## v0.19 current delta

Foreground selection now uses a target-position prior (cutout centre by default,
or explicit pixel coordinates) and records ambiguous selections for abstention.
Observation 1.15 adds per-structure comparison survival, occupied angular bins,
radial fractional width and local tangent measurements. The expert evaluates
independent survival on the structures supporting a configuration, not on an
unrelated pooled mask. Low pooled survival remains a processing caution, not an
artifact verdict by itself. Weak held recovery evidence may justify C but cannot
be promoted to a formal structure or justify A by that fact alone.
See [the Chinese v0.19 guide](V0190_CENTER_STRUCTURE_CHANGE_CN.md) for thresholds,
compatibility boundaries and validation limitations.

## Non-negotiable boundary

The vision layer may measure morphology, geometry, signal-to-noise, and detector
stability. It must not output `lens`, `not lens`, or a lens probability. Its only
semantic output is a versioned `ObservationDocument`.

The expert workflow receives that document and never opens image pixels. Pixel
arrays remain available as provenance artifacts for a separate scientific audit.

## Vision pipeline

One preprocessing pass produces a foreground-galaxy model, residual, residual S/N,
source mask, and segmentation. The v0.10 detection path is deliberately chained:

1. S/N threshold layers remain a trace and stability diagnostic;
2. SEP runs across the deblending grid and preserves the exact union of segmented
   pixels as trace-only detections;
3. stable SEP pixels can seed constrained circular/elliptical search routes, while
   centre-based fallback routes remain available even when a seed exists;
4. pixels outside each route are masked through SEP's extraction interface and SEP
   runs again inside the permitted band;
5. route detections become unified model-chain candidates;
6. a candidate-structure audit combines candidate-level connected S/N
   persistence, route support, and foreground-contamination checks to route each
   candidate to primary, recovery, foreground-contaminated, or trace handoff;
7. primary candidates and promoted stable recovery candidates create `structures`
   for the current expert rules; held recovery, foreground-contaminated, and trace
   candidates remain available for review and provenance.

The search route is never filled with synthetic signal. Raw SEP detections, route
statistics, and route-masked SEP candidates share one residual and are correlated.
None is a lens label.

FITS input uses supplied flux and RMS extensions when present. PNG input estimates
noise from pixels and uses a reduced parameter grid and radial foreground model.
This limitation is recorded in the Observation document for the future grading audit.

## Observation contract

`schema_version=1.14` records:

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

Version 1.5 added a posterior circular/elliptical annulus measurement channel.
It runs after SEP extraction but does not require SEP to see every ring pixel:
foreground-centred circular templates with small centre offsets, plus
foreground-centred elliptical templates, sample residual S/N along an annular
band.  It records angular positive-S/N coverage, longest contiguous coverage,
on-band S/N, and an adjacent-band contrast. Version 1.6 removes the initially
implemented 0.5-S/N/80-percent hard screen because those numbers were meeting
examples, not calibrated rules. It records descriptive local-background coverage
instead, with no pass/fail field and no direct evidence weight. A retained ring
template is not an Einstein-ring or lens label. This channel
shares the primary foreground-subtracted residual and is therefore correlated
with SEP measurements; it is assigned a separate correlation group for audit.

Version 1.7 replaced the formal v1.5/v1.6 posterior-ring output with a unified
geometry-guided path search. Detected residual structures seed constrained
circular/elliptical routes. The route, its seed/linked structures, and
newly recovered pixel segments are stored separately. Unfilled route pixels are
not observations, and path generation does not contribute direct lens evidence.

Version 1.8 chains these formerly parallel outputs. Raw SEP detections no longer
become scored structures. Centre fallbacks remain alongside seeded routes so that
an unrelated SEP island cannot suppress a weak-ring search. Route-masked SEP must
also obey the configured global inner/outer search boundary. A seed-fitted route
cannot validate its own seed: geometry support additionally requires positive
on-route versus adjacent-route contrast and independently recovered pixels. The
new `model_candidates` records origin, status, path links, geometry, S/N, foreground
comparison survival, and explicit `not_scoring_input` flags. Until candidate
promotion is calibrated, `structures` remains empty for new v0.10 vision runs.

Version 1.9 removes any implied ordering between original SEP components and
route-masked SEP recoveries. Raw SEP components are labelled `raw_sep_only` when
they lack qualifying geometric support and `sep_plus_geometry` when their median
route deviation lies inside the route's recorded half-width. Route-masked SEP
discoveries are labelled `route_recovered`. These are structural-support sources,
not confidence levels. A seed-generated route cannot support its own seed without
an additional component, while a positive-contrast centre-fallback route can supply
a separately constructed but still residual-correlated geometric relation.
Components associated through one route are stored
as a `candidate_system`, with foreground-model and near-core audit flags kept on a
separate axis. Candidate systems remain excluded from the reasoning score.

Version 1.10 changes the classification unit from a whole raw SEP connected
component to local pixel partitions. Pixels inside an admissible route band are
split into `sep_plus_geometry` components, while off-route remainders are retained
as downgraded SEP trace. Route-supported original pixels and route-recovered pixels
share a `priority_keep` review priority but retain distinct provenance. Sub-minimum
fragments created by partitioning remain in pixel trace arrays rather than becoming
new structural candidates. No partition is promoted to `structures` or scoring
without a later calibrated promotion audit.

Version 1.11 adds the explicit candidate-to-structure audit bridge. It records
one of `promoted`, `hold_for_review`, or `trace_only` for every model candidate,
along with individual checks, supporting facts, cautions, rationale, and the
resulting structure ID. Only promoted candidates create measured `structures`
for downstream hypothesis comparison. Promotion is deliberately limited to
measurement usability; it is not an arc, ring, multiple-image, or lens label.
Fallback foreground modeling remains a visible caution rather than silently
discarding every candidate, while unusable input and complete erasure under the
comparison foreground model prevent automatic promotion.

Version 1.12 separates geometric discovery from signal-reality audit. Every
recorded model candidate, including an off-route global-SEP remainder, receives
the same detector-level audit. Route membership remains recorded as discovery
context and a later reasoning feature, but it is never an audit admission
condition. The audit is split into signal-quality, processing-robustness, and
discovery-provenance sections. A candidate may be promoted only as a measurable
residual signal; this does not identify an arc, ring, multiple image, or lens.
The initial implementation does not automatically refit a route or rerun the
visual pipeline, preventing an audit-recovery loop from tuning geometry toward
the signal it is trying to validate.

Version 1.13 introduces an explicit four-way candidate handoff. Candidate-level
S/N persistence is measured by connected survival across the configured eight
threshold masks. Geometry still does not decide whether a signal is real, but it
may route a stable candidate to the primary or recovery channel. Candidates that
overlap the subtraction-sensitive core under a failed foreground fit, or disappear
under the comparison foreground model, are isolated as foreground-contaminated.
Only primary candidates create current-rule `structures`; recovery,
foreground-contaminated, and trace candidates remain machine-readable and
pixel-traceable for later hypothesis-specific analysis.

Version 1.14 corrects that handoff boundary. A stable off-route candidate is now
promoted with `recovery_candidate` provenance and complete structure measurements;
route membership cannot block a real measurable signal. Near-core overlap and a
fallback primary foreground model remain cautions. They isolate a candidate only when
the independent comparison foreground model affirmatively erases it. Both primary and
promoted-recovery structures enter scene geometry and expert configuration analysis.

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

Version 0.18.0 has one primary interpretation pass:

```text
ObservationDocument -> ExpertAssessmentAgent -> DecisionAudit -> ReportWriter
```

`ExpertAssessmentAgent` builds one evidence dossier containing cited observed facts,
signal reality, lens specificity, supporting and contradicting evidence, five
competing explanations, missing information, processing cautions, correlation groups,
unused fields, null tests, future tool requests, and eight traceable lens-configuration
hypotheses. It then answers the holistic expert questions and emits one categorical
X/C/B/A vote, or `insufficient_for_grading`. Numeric compatibility fields remain null.
No component scores are added or averaged. High S/N, SEP persistence, threshold
survival, and route agreement cannot individually satisfy the lens-specificity gate.

The v0.18.0 interpretation recognizes annular/partial-ring, tangential-arc, fold-arc,
cusp-arc, radial-arc, compact-multiple-image, asymmetric-multiple-image, and single-image
hypotheses. A robust partial arc may support A without mandatory multiplicity; a short
arc may support A through a fold/cusp relationship; radial arcs are not rejected by a
tangential-only gate; compact quads need not contain a long arc. These hypotheses are
alternative morphology routes, not additive votes or universal hard cuts.

`DecisionAudit` is deliberately lightweight. It checks references, missing-as-negative
errors, correlated double counting, measurement-as-proof errors, automatic recovery
penalties, ignored alternatives, external-answer leakage, score/rationale mismatch,
and required abstention. It never reconstructs the competing-hypothesis analysis and
never generates an alternative score.

The old `HypothesisAgent`, `EvidenceAgent`, `CounterEvidenceAgent`,
`HypothesisAnalysisAgent`, `EvidenceAuditor`, `ReasoningAuditor`, and
`GradingPreparationAgent` modules remain in the repository for history and safe Git
rollback, but they are not imported by the active workflow.

Unavailable measurements are uncertainty, not negative evidence. Route recovery is
provenance, not an automatic penalty. Search-route and candidate-system records remain
available to explain discovery context but cannot convert the number of residual
components into a multiple-image claim.

The active grading scheme is `euclid_q1_paper`. Provisional point outputs may display
the paper boundaries (A > 2.0, B for 1.5 < score <= 2.0, C <= 1.5), but are explicitly
`provisional_uncalibrated`. `official_expert_score` remains null until labelled Q1
aggregate scores and an independently evaluated calibration procedure are supplied.
A+ is represented by separate scientific-value flags. Scientifically unusable data
abstain as `insufficient_for_grading`, never as X.

## Knowledge and skill separation

Files under `knowledge/` state domain interpretation, confounders, missing evidence,
and forbidden inferences for lens geometry, arc morphology, intrinsic galaxy
morphology, mergers, artifacts, and colour. Files under `skills/` define repeatable
agent procedures. Python code remains responsible for measurement, validation,
threshold enforcement, and structured outputs. Loading knowledge never creates an
observation and never overrides measurement provenance.

The active `DecisionAudit` is a decision-integrity check, not a second expert. Data
usability, measurement reliability, independence, specificity, missing information,
and competing explanations are all assembled once inside the evidence dossier. Any
future quantitative calibration audit must be added as a post-assessment calibration
stage and validated on labelled data rather than guessed from individual examples.

## Visual trace bundle

Version 0.5 renders the actual intermediates used by the vision pipeline rather
than reconstructing illustrative approximations. The threshold detector and its
diagnostic share `threshold_masks`; the SEP figure plots exact consensus
segmentation pixels rather than fixed-size symbols; and the model-chain figure
compares trace-only raw SEP pixels with route-masked SEP candidates. A per-run
manifest records the input SHA-256 digest, runtime parameters,
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

`vision_field_context.png` connects measurements back to the source data by showing
the full scientific FITS field, crop, annulus, and IDs. RGB previews are not emitted
as a separate context diagnostic. Target-routing values remain in the manifest and
Observation; `vision_target_selection.png` is emitted only for a close score
competition, a centre-prior conflict, or a human override.

The fixed architecture diagram is maintained once in `VISION_PROCESS_FLOW_CN.md`.
Per-object output directories contain only sample-dependent diagnostic images; they
do not duplicate a static flowchart.

## Intended extensions

- PSF-aware and multi-band foreground modeling;
- source-color consistency and repeated-image comparison;
- WCS/pixel-scale validation;
- merger/core and tidal-asymmetry measurements;
- calibration on labelled positive and hard-negative datasets;
- posterior multi-image grouping and cross-band consistency measurements;
- optional schema-constrained LLM explanations that may summarize cited evidence
  but cannot alter measurements, scores, gates, or audit results.
