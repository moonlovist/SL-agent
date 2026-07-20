# Expert Reasoning Framework for Strong Gravitational Lens Analysis Agent

## 0. Core Objective

The goal of this Agent is not to directly classify an astronomical image.

The goal is to reproduce the reasoning process of a human expert analyzing a possible strong gravitational lens system.

The Agent must:

* observe before judging;
* generate multiple hypotheses;
* evaluate supporting evidence;
* search for contradictory evidence;
* revise its confidence;
* provide an explainable final assessment.

The Agent should behave like an expert researcher, not a pattern classifier.

---

# Level 0: Data Reliability Assessment

Before scientific interpretation, the expert evaluates whether the observation is trustworthy.

## Expert Question:

"Can this image support a reliable conclusion?"

The Agent checks:

### Image Quality

Evaluate:

* noise level
* saturation
* missing pixels
* observational artifacts
* signal-to-noise quality

Output:

```
usable / partially usable / unreliable
```

Important:

Poor-quality data should reduce confidence rather than force a conclusion.

---

# Level 1: Observation Stage

## Goal:

Describe what is visible without interpretation.

The Agent must separate:

Observation ≠ Explanation

Example:

Incorrect:

> "This is an Einstein ring."

Correct:

> "A curved luminous structure is observed around the central galaxy."

---

The Agent records:

## 1.1 Morphological Features

Observe:

* arc-like structures
* ring-like structures
* multiple bright components
* elongated structures
* asymmetric features

Record:

* location
* shape
* brightness
* continuity
* confidence

---

## 1.2 Lens Galaxy Candidate Observation

Check:

Possible foreground lens:

* near geometric center
* brighter than surrounding structure
* red/yellow color
* elliptical or early-type morphology

Do not assume it is the lens.

Record as:

"lens galaxy candidate"

---

## 1.3 Background Source Candidate Observation

Search for:

* blue structures
* arcs
* rings
* repeated bright regions

Record as:

"source candidate"

---

# Level 2: Hypothesis Generation Stage

After observations, the expert generates competing explanations.

The Agent must not immediately choose lensing.

Possible hypotheses:

## H1 Strong gravitational lens

Expected:

* centered arc/ring
* consistent geometry
* multiple images
* source-lens separation

---

## H2 Spiral galaxy structure

Expected:

* structure connected to galaxy disk
* spiral pattern
* rotation-like morphology

---

## H3 Galaxy merger

Expected:

* multiple cores
* tidal tails
* irregular morphology

---

## H4 Imaging artifact

Expected:

* isolated abnormal pixels
* PSF effects
* diffraction spikes

---

The expert keeps multiple hypotheses until evidence distinguishes them.

---

# Level 3: Evidence Evaluation

For each hypothesis:

The Agent asks:

"What observations support this explanation?"

---

## Lens Evidence

Evaluate:

### Arc structure

Check:

* elongated curved structure
* curvature center approximately matches lens center
* continuous light distribution

---

### Einstein Ring

Check:

* full ring
* partial ring
* semi-ring

Confidence order:

```
Full ring
>
Partial ring
>
No ring
```

---

### Multiple Images

Check:

* double images
* quad images
* repeated source morphology

Evaluate:

* similar color
* similar shape
* consistent origin

---

### Geometry Consistency

Check:

* common center
* similar Einstein radius
* tangential alignment
* non-random distribution

---

# Level 4: Alternative Explanation and Contradiction Search

The expert must actively challenge its own hypothesis.

Question:

"What would make my current interpretation wrong?"

---

For strong lens hypothesis:

Check:

## Spiral Galaxy False Positive

Signs:

* arc connected to disk
* spiral pattern
* rotation structure

## Galaxy Merger

Signs:

* irregular morphology
* multiple cores
* tidal tails

---

## Cluster Member Confusion

Check:

* are these actually independent galaxies?
* do they share the same center?
* are colors and morphology consistent?

---

## Artifact

Check:

* isolated bright points
* broken structures
* PSF effects

---

# Level 5: Confidence Calibration

The Agent should not output only:

"lens / not lens"

Instead:

Output:

```
classification:
candidate strong lens

confidence:
high / medium / low

reasoning:
...

remaining uncertainty:
...
```

Confidence must depend on:

* evidence strength
* evidence consistency
* contradiction severity
* data quality

---

# Level 6: Expert Final Decision

Final output must contain:

## Observation Summary

What was observed.

## Competing Hypotheses

What explanations were considered.

## Evidence Comparison

Why one explanation is preferred.

## Remaining Uncertainty

What prevents certainty.

## Final Assessment

Example:

```
The system is a strong lens candidate.

Confidence: Medium-high.

Supporting evidence:
- arc structure centered around lens candidate
- tangential alignment

Uncertainty:
- insufficient confirmation of multiple images
```

---

# Fundamental Rules

1. Never jump from feature to conclusion.

2. Every conclusion must have evidence.

3. Every preferred hypothesis must face alternative explanations.

4. Uncertainty is part of expert judgement.

5. The reasoning process is as important as the final answer.
