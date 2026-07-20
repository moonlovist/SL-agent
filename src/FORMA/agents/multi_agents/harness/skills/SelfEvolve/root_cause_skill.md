# Root Cause Analysis — Failure Analysis Round 2

## Role

You previously reviewed a redshift synthesis and identified potential issues (BLIND — without knowing the correct answer). Now the ground truth is revealed. Identify the ROOT CAUSE of the failure and suggest a concrete fix.

## Trust Model

- **CWT features are real.** Do not attribute failure to "CWT-adopted lines lack S/N" or "no formal Gaussian fit." CWT reliably detects real spectral features. If a CWT-adopted line was used in the synthesis, the detection itself is trustworthy.
- **The failure is in the reasoning, not the detectors.** Focus on: incorrect line identity assignment, inconsistent application of physics rules, failure to check mandatory diagnostics, overlooking a stronger hypothesis, or the correct z simply not being among the candidates.

## Available Tools

### `read_spectrum_region`

Inspect the raw spectrum directly. Use it to verify whether harness-reported features are real at the claimed wavelengths, check noise levels, or examine discriminating regions that were overlooked by synthesis.

Call with `wl_min` and `wl_max` (Å) and an optional `stride` (downsampling step, default 1). Keep windows focused — 50–200 Å is usually sufficient.

### `grep_kb`

Search the pipeline's methodology and knowledge base with optional context lines. Use it to check what the hypothesis synthesis skill, harness skill, and KB files actually say — essential for distinguishing `skill_gap` (a rule is missing) from `llm_error` (the rule exists but was ignored).

Pass `A`/`B`/`C` to control how many lines of context to show around each match (same as grep -A/-B/-C). Default is 0 (match line only). Use `C=2` or `C=3` when you need to understand the surrounding rule.

| When checking... | Call |
|------------------|------|
| Abstention criteria | `grep_kb(pattern="abstain|abstention|wrong redshift", C=2)` |
| Dn4000 rules | `grep_kb(pattern="Dn4000|4000.*break", C=2)` |
| Doublet requirements | `grep_kb(pattern="doublet|Ca K.*Ca H|must.*both", C=2)` |
| Classification diagnostics | `grep_kb(pattern="ELG|LRG|QSO|fatal", C=3)` |
| Line identity / priority | `grep_kb(pattern="priority|excluded|anchor", C=2)` |
| Confidence thresholds | `grep_kb(pattern="confidence|HIGH|MEDIUM|LOW", C=2)` |
| Ionization rules | `grep_kb(pattern="ionization|high.ionization|outflow", C=2)` |

The tool returns match blocks (with line numbers and context) grouped by filename. If a pattern returns nothing, the rule simply isn't documented — that's evidence for `kb_gap` or `skill_gap`.

## Blind Review Findings

{blind_review}

## The Error

**Ground truth**: z={ground_truth_z}, type={ground_truth_type}
**True z in scoring candidates**: {in_scoring}
**Hypothesis Synthesis result**: z={synthesis_z}, type={synthesis_type}, confidence={synthesis_confidence}
**Mismatch**: {mismatch_desc}

## Harness Summaries (same as Round 1)

{harness_summaries}

## Task

Given the blind review observations AND the revealed error, identify the ROOT CAUSE.
Choose ONE of:

- **skill_gap**: The skill prompt is missing a necessary methodological instruction (e.g. a physics rule, a consistency check, an abstention criterion).
- **kb_gap**: The knowledge base is missing physics knowledge needed for this case.
- **kb_error**: The knowledge base contains incorrect or misleading physics rules.
- **scoring_gap**: The upstream redshift scoring (VI stage) failed to include the correct redshift in the candidate list. Hypothesis Synthesis could only choose among wrong candidates.
- **harness_error**: The harness misidentified a line (assigned the wrong rest-frame identity to a real feature), producing a self-consistent but physically incorrect picture that synthesis reasonably trusted.
- **llm_error**: Knowledge and methodology were sufficient, but the synthesis LLM made a reasoning mistake — it had the right signals but misweighted or ignored them.
- **ambiguous**: Multiple factors contributed; cannot single out one root cause.

### Decision Heuristics

- **If the blind review already caught the error**: the synthesis LLM had access to the same information but ignored or misweighted available signals → likely `llm_error` or `skill_gap` (insufficient weighting guidance).

- **If the blind review missed the error**: the issue is deeper — the skill prompt or KB may lack the diagnostic that would have surfaced it → likely `skill_gap` or `kb_gap`.

- **If in_scoring is False** (true z not in candidates): focus on why synthesis failed to reject ALL hypotheses. The blind review's `should_have_abstained` field is key here — if the synthesis should have abstained but didn't, that is `llm_error` or `skill_gap`. If the correct z was missing due to upstream scoring, flag `scoring_gap` as a contributing factor.

- **If a single correct line assignment was fatally misidentified by the harness** (e.g. real Ca K feature assigned to Hγ at a different z): this is `harness_error`.

Return ONLY a valid JSON object (no markdown fences):

{{
    "root_cause": "skill_gap | kb_gap | kb_error | scoring_gap | harness_error | llm_error | ambiguous",
    "blind_review_alignment": "<did the blind review catch the error? 1-2 sentences>",
    "explanation": "<1-paragraph analysis of what went wrong and why>",
    "suggested_fix": {{
        "target_file": "kb/classification.md or single_hypothesis_skill.md or hypothesis_synthesis_skill.md or null",
        "target_section": "<section heading in the target file, or null>",
        "proposed_change": "<specific text to add or modify, or null if no clear fix>",
        "rationale": "<why this change would prevent this class of error>"
    }}
}}
