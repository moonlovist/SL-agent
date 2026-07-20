# Blind Review — Failure Analysis Round 1

## Role

You are reviewing a redshift synthesis pipeline's output. Critically examine the synthesis reasoning and identify any potential issues — you do NOT know the correct answer yet.

## Pipeline Architecture

1. **Harness**: Per-hypothesis verification. CWT pre-detects features in the spectrum, then the harness assigns rest-frame line identities and runs Gaussian fits where possible. A line may be CWT-adopted (no independent fit attempted) when the feature is clearly present but too shallow/noisy for a free Gaussian fit to converge.
2. **Middleware**: Structured markdown summaries from CSV + LLM text extraction.
3. **Synthesis**: Cross-compares all hypotheses (Phase 1 blind review → Phase 2 targeted spectrum reads → Phase 3 JSON verdict).

## Trust Model

- **CWT-detected features are real.** The CWT algorithm reliably finds genuine spectral features. A line labelled LIKELY means the feature is unambiguously present at the predicted wavelength. Do NOT second-guess whether a CWT-adopted feature "really exists" — it does.
- **The question is assignment, not detection.** The risk is that a real feature at wavelength λ is assigned to the wrong rest-frame line identity. Two competing hypotheses may both claim the SAME observed feature as different rest-frame lines. This is the contradiction you should hunt for.
- **Absent lines matter more than present ones.** A NOT_FOUND line that SHOULD be strong at a given redshift/classification is a fatal problem. Focus on missing mandatory diagnostics (Ca K without Ca H, [O II] absent in an ELG, Hα missing when [N II] is present, etc.).

## Harness Summaries

{harness_summaries}

## Synthesis Reasoning

{synthesis_reasoning}

## Task

Critically review the synthesis above. Focus on:

1. **Contradictions**: Are there inconsistencies in the contradiction matrix or between hypotheses? Does one hypothesis claim a feature as line X while another claims the same feature as line Y — and which assignment is physically justified?
2. **Evidence quality**: Does the conclusion follow from the line data? Are mandatory diagnostics (doublet pairs, Balmer series, key absorption/emission lines) present when the classification demands them? Are there weak or missing lines being glossed over?
3. **Methodology gaps**: What should the synthesis have checked but didn't? Did it apply physics rules inconsistently across hypotheses?
4. **Alternative hypotheses**: Is there a stronger candidate being overlooked? Look for hypotheses with internally consistent line inventories that were dismissed for weak reasons.

Return ONLY a valid JSON object (no markdown fences):

{{
    "overall_assessment": "<1 paragraph: is the synthesis conclusion well-supported or questionable?>",
    "issues": [
        {{
            "severity": "critical | major | minor",
            "category": "contradiction | evidence_quality | methodology_gap | overlooked_hypothesis | classification_error | other",
            "description": "<specific observation, cite wavelengths and line names>",
            "affected_hypothesis": "<which hypothesis index, or 'overall'>"
        }}
    ],
    "strongest_alternative": "<which hypothesis (if any) looks stronger than the chosen one, and why; or null>",
    "should_have_abstained": true or false
}}
