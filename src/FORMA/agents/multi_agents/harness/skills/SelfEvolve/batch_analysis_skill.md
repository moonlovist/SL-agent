# Batch Failure Analysis

## Role

You are reviewing a batch of {n_failures} redshift pipeline failures to identify systematic patterns and suggest concrete improvements.

## Trust Model

- **CWT features are real.** Do not attribute failures to "CWT-adopted lines lack S/N." CWT reliably detects real spectral features. If a line was detected by CWT, the physical feature exists in the spectrum.
- **Focus on reasoning failures.** Look for patterns in: incorrect line identity assignment, inconsistent application of physics rules, missing mandatory diagnostic checks, overlooked alternative hypotheses, or correct z missing from candidate list.

## Failure Summaries

{failure_summaries}

## Task

1. **Group** these failures by their likely root cause category:
   - `skill_gap`: methodology / prompt instructions are missing
   - `kb_gap`: knowledge base is missing physics knowledge
   - `kb_error`: knowledge base has incorrect physics rules
   - `scoring_gap`: upstream redshift scoring failed to include correct z in candidates
   - `harness_error`: per-hypothesis line identification produced a misleading but self-consistent picture
   - `llm_error`: the synthesis LLM made a reasoning mistake despite adequate knowledge
   - `ambiguous`: multiple factors, or not enough information

2. **For each group**, identify the common pattern and write a concise diagnosis.

3. **For skill_gap / kb_gap / kb_error / scoring_gap groups**, propose ONE concrete fix per group:
   - `target_file`: which file to modify
   - `target_section`: which section
   - `proposed_change`: what text to add or change
   - `rationale`: why this addresses the pattern

4. **For harness_error / llm_error / ambiguous groups**, note that manual review is needed.

5. **Prioritise**: which fix would prevent the most failures?

Return ONLY a valid JSON object (no markdown fences):

{{
    "batch_diagnosis": "<2-3 sentence summary of the dominant failure mode across this batch>",
    "groups": [
        {{
            "root_cause": "skill_gap|kb_gap|kb_error|scoring_gap|harness_error|llm_error|ambiguous",
            "count": <number of failures in this group>,
            "spectrum_ids": ["id1", "id2", ...],
            "pattern": "<1-sentence description of the common pattern>",
            "diagnosis": "<1-paragraph analysis>",
            "suggested_fix": {{
                "target_file": "<path or null>",
                "target_section": "<section or null>",
                "proposed_change": "<specific text or null>",
                "rationale": "<why this fix addresses the pattern or null>"
            }}
        }}
    ],
    "priority_fix": {{
        "group_index": <index into groups array, 0-based>,
        "reasoning": "<why this fix should be applied first>"
    }}
}}
