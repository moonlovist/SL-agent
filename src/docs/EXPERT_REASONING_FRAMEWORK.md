# Strong Gravitational Lens Expert Reasoning Framework v1.1

## Expert Reasoning Framework for Strong Gravitational Lens Analysis Agent


---

# 0. Core Objective


The goal of this project is not to build a simple image classifier, nor to let the model directly answer:

> "Is this a gravitational lens?"


Instead, this project aims to build an intelligent agent system that simulates the reasoning process of human astronomical experts when analyzing unknown targets.


The agent should reproduce the expert workflow:


```

Data Reliability Assessment
↓
Observation Extraction
↓
Candidate Hypothesis Generation
↓
Evidence Planning
↓
Evidence Evaluation
↓
Contradiction Search and Alternative Explanation Analysis
↓
Confidence Calibration
↓
Final Expert Judgement

```


The final output should not only include the conclusion, but also:


* Why this judgement was made;
* Which evidence supports it;
* What uncertainties remain;
* Why alternative explanations are weakened.


---

# 1. Data Reliability Assessment Stage


## Expert Question:


> "Is this image reliable enough to support scientific judgement?"


Before analyzing gravitational lensing, experts first evaluate data quality.


---

## Image Quality


Check:


* Noise level;
* Overexposure;
* Bad pixels;
* Observational artifacts;
* Signal-to-noise ratio.


Output:


* Suitable for analysis;
* Partially suitable;
* Unreliable.


Principle:


When data quality is poor:

The system should not force a conclusion.

Instead, confidence should be reduced.


---

## Image Scale


Check:


* Whether the complete galaxy system is included;
* Whether the image is too small for structural analysis.


---

## Environmental Information


If the system contains:


* Multiple galaxies;
* Dense galaxy environments;


Consider:


* Group-scale lensing;
* Cluster-scale lensing.


---

# 2. Observation Stage


## Core Principle:


Observation ≠ Interpretation


Experts first answer:


> "What do I observe?"


Not:


> "What is this object?"


---

# 2.1 Morphological Observation


Record:


* Arc-like structures;
* Ring-like structures;
* Multiple bright components;
* Elongated structures;
* Irregular structures.


Record:


* Position;
* Shape;
* Brightness;
* Continuity;
* Confidence.


---

# 2.2 Foreground Lens Candidate


Search for:


Possible Lens Galaxy:


* Located near the geometric center;
* Relatively bright;
* Red/yellow color;
* Elliptical galaxy morphology.


Important:


Do not directly call it a lens.


Use:


> lens candidate


---

# 2.3 Background Source Candidate


Search for:


* Blue structures;
* Arc structures;
* Ring structures;
* Multiple bright components.


Record:


> source candidate


---

# 3. Hypothesis Generation Stage


## Core Principle


Experts do not immediately conclude:


> "This is a lens."


when they observe an arc-like structure.


Instead, they construct competing explanations.


However, experts also do not enumerate every possible explanation.


In real analysis, experts select a small number of the most relevant competing hypotheses based on current observations.


---

## Hypothesis Generation Objective


Generate:


* A small number of competing hypotheses;
* Initial confidence estimates;
* Questions that require further verification.


The goal is not to select the answer immediately.


The goal is to determine:


> "Which explanations are most worth testing?"


---

# H1: Strong Gravitational Lens


Possible forms:


## Giant Arc


Expected:


* Long curved structure;
* Located around lens candidate;
* Tangential alignment.


---

## Einstein Ring


Expected:


* Complete ring;
* Partial ring;
* Ring-like structure.


---

## Multiple Images


Expected:


* Double images;
* Quad images;
* Multiple similar structures.


Compare:


* Color;
* Morphology;
* Source consistency.


---

## Galaxy-Galaxy Lens


Expected:


* Foreground lens galaxy;
* Background source distortion.


---

## Cluster / Group Scale Lens


Expected:


* Multiple galaxy environment;
* Large-scale lensing effects.


---

# H2: Intrinsic Galaxy Morphology


Possible forms:


## Spiral Arm


Expected:


* Structure connected to galaxy center;
* Rotational pattern.


---

## Disk Structure


Expected:


* Structure belongs to galaxy disk.


---

## Irregular Morphology


Expected:


* Complex internal structure;
* Asymmetric brightness distribution.


---

# H3: Galaxy Interaction


Possible forms:


## Merger


Expected:


* Multiple cores;
* Disturbed morphology.


---

## Tidal Feature


Expected:


* Tidal tails;
* Elongated structures;
* Asymmetric distribution.


---

# H4: Observation Artifact


Possible forms:


## Noise


Expected:


* Random bright points;
* Lack of continuous structure.


---

## PSF Artifact


Expected:


* Structure related to point spread function effects.


---

## Bad Pixel / Detector Issue


Expected:


* Local anomalies;
* Non-astronomical structures.


---

# Hypothesis Generation Principles


1. Do not lock onto a conclusion too early.

2. Maintain competing explanations.

3. Do not infer conclusions from a single feature.

4. Prioritize hypotheses that can be distinguished through observations.


---

# 4. Evidence Planning Stage


## Core Question:


> "What should we check next?"


Experts do not perform unlimited feature searches.


When facing multiple hypotheses, experts first identify:


> The evidence that provides the strongest discrimination between competing explanations.


This is called:


> Discriminative Evidence


---

## Objective


Based on current hypotheses, determine:


* Which observations are most valuable;
* Which checks can eliminate alternatives;
* Which analysis should be prioritized.


---

## Example 1: Lens vs Spiral Arm


Current hypotheses:


```

H1: Strong gravitational lens

H2: Spiral arm structure

````


Important questions:


### Question 1:

Does the arc-like structure share the same center as the lens candidate?


Purpose:


Distinguish:


* Lensing geometry;
* Intrinsic galaxy structure.


---

### Question 2:

Is the structure connected to the galaxy disk?


Purpose:


Evaluate spiral arm explanation.


---

## Example 2: Lens vs Merger


Important questions:


Check:


* Multiple cores;
* Tidal features;
* Irregular brightness distribution.


If these features are significant:


Increase merger possibility.


If absent:


Decrease merger possibility.


---

## Evidence Planning Output


Each verification question should contain:


```json
{
    "question": "",
    "purpose": "",
    "related_hypotheses": []
}
````

Example:

```json
{
    "question":
    "Does the arc share the same center as the lens candidate?",

    "purpose":
    "Distinguish gravitational lensing geometry from intrinsic galaxy structure",

    "related_hypotheses":
    [
        "strong_lens",
        "spiral_arm"
    ]
}
```

---

# 5. Evidence Evaluation Stage

## Core Question:

> "Which observations support or weaken each hypothesis?"

Evidence evaluation is not simply collecting reasons that support one answer.

It compares competing explanations based on planned questions.

---

## Each Evidence Item Should Record:

* Observation;
* Related hypothesis;
* Reliability;
* Specificity;
* Reasoning.

Format:

```json
{
    "feature": "",
    "related_hypothesis": "",
    "reliability": "",
    "specificity": "",
    "reasoning": ""
}
```

---

# 5.1 Strong Gravitational Lens Evidence

## Arc Structure

Check:

* Long curved structure;
* Central alignment;
* Curvature center;
* Continuity.

Important:

Arc structure alone is not absolute evidence.

Spiral arms and merger structures may create similar appearances.

---

## Einstein Ring

Check:

* Complete ring;
* Partial ring;
* Half ring.

Confidence:

```
Complete ring
 >
Partial ring
 >
No ring
```

---

## Multiple Images

Check:

* Double;
* Quad;
* Multiple similar structures.

Compare:

* Color;
* Morphology;
* Brightness;
* Spatial position.

Important:

Multiple bright components are not automatically multiple images.

They must be consistent with the same background source.

---

## Geometric Consistency

Check:

* Shared center;
* Einstein radius consistency;
* Tangential alignment;
* Random distribution.

---

# 5.2 Alternative Explanation Evidence

Evidence evaluation must also test alternatives.

## Intrinsic Galaxy Structure

Check:

* Connection to galaxy center;
* Spiral pattern;
* Disk structure.

---

## Galaxy Merger

Check:

* Multiple cores;
* Tidal tails;
* Disturbed morphology.

---

## Observation Artifact

Check:

* Isolated bright points;
* Lack of continuous structure;
* Possible instrumental origin.

---

# 6. Contradiction Search and Alternative Explanation Stage

## Core Question:

> "If my current judgement is wrong, what is the most likely alternative?"

This is the major difference between experts and ordinary classifiers.

A classifier:

```
Find supporting evidence
        ↓
Increase confidence
```

An expert:

```
Find supporting evidence
        ↓
Search for contradictions
        ↓
Check whether the conclusion still holds
```

---

# False Positive Analysis

## 6.1 Lens vs Spiral Arm

If it is lensing:

* Arc exists independently;
* Structure surrounds center;
* Not connected to galaxy disk.

If it is spiral arm:

* Connected to center;
* Rotational pattern;
* Part of galaxy morphology.

---

## 6.2 Lens vs Galaxy Merger

Merger indicators:

* Multiple cores;
* Tidal tails;
* Asymmetric structures.

If strong:

Decrease lensing confidence.

---

## 6.3 Multiple Images vs Independent Galaxies

Check whether components have:

* Similar colors;
* Similar morphology;
* Consistent geometry.

Otherwise:

They may simply be separate objects.

---

## 6.4 Real Structure vs Observation Artifact

Check:

* Continuity;
* Repeatability;
* Astronomical consistency.

---

# 7. Confidence Calibration Stage

## Core Question:

> "How strongly should this conclusion be trusted?"

Experts do not output only:

```
Yes
No
```

They output:

```
Candidate: Strong gravitational lens

Confidence:
High / Medium / Low

Reasons:
...

Supporting evidence:
...

Remaining uncertainty:
...
```

---

# Factors Increasing Confidence

Including:

* Multiple independent supporting evidence;
* Consistent geometry;
* Weak alternatives;
* High data quality.

---

# Factors Reducing Confidence

Including:

* Poor data quality;
* Ambiguous structure;
* Strong alternative explanations;
* Missing key evidence.

---

# Confidence Updating Principle

Confidence should not be fixed.

It should update with:

* New evidence;
* New contradictions;
* Data quality assessment.

---

# 8. Final Expert Report Stage

The final report must include:

---

## 1. Data Reliability

Whether the data supports analysis.

---

## 2. Observation Summary

What was observed.

---

## 3. Candidate Explanation Comparison

Which explanations were considered.

---

## 4. Evidence Analysis

Why certain explanations are supported.

---

## 5. Contradiction Analysis

Why alternatives are weakened.

---

## 6. Final Judgement

Example:

> This system is a strong gravitational lens candidate.
>
> Confidence: Medium-high.
>
> Supporting evidence:
>
> * Arc structure around the center;
> * Geometry consistent with lensing.
>
> Remaining uncertainty:
>
> * Lack of confirmed multiple images.

---

# Project Principles

1. Conclusions must not be directly inferred from single features.

2. Every conclusion must have evidence.

3. Every judgement must consider alternative explanations.

4. Uncertainty must always be recorded.

5. Reasoning process is as important as the final result.

6. The agent should not search every possible explanation. It should simulate expert prioritization of the most valuable information.

7. High-quality reasoning does not mean generating more hypotheses. It means finding the most important evidence under limited information.

---
