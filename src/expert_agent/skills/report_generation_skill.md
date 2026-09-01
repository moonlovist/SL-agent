# Report Generation Skill

## Role

Transform the measured and audited reasoning trace into a readable report without
creating new observations, a provisional score, or an unsupported grade.

## Required content

The report must preserve:

- data quality and observation-audit findings;
- visual candidate provenance and handoff categories;
- all four competing hypothesis case files;
- supporting, opposing, missing, and uncertain evidence;
- cross-hypothesis comparisons and correlation groups;
- evidence and reasoning audit status;
- the versioned Q1 grading status.

## Forbidden inference

- Do not call a measured residual an arc, ring, or multiple image unless an
  explicit interpretation stage has established that claim.
- Do not turn `strength`, `reliability`, or `specificity` metadata into an
  `expert_score`.
- Do not report a confidence number or lens probability.
- Do not convert missing or unusable data to a low expert score.
- Do not issue A/B/C while grading status is `calibration_pending`,
  `insufficient_data`, or `blocked_by_audit`.
- Do not infer A+ from morphology alone; it requires an explicit scientific-value
  review and a recorded reason.

## Grading language

Use the exact fields supplied by `GradeAssessment`:

- `grading_scheme`;
- `status`;
- `expert_score`;
- `grade`;
- `a_plus` and `a_plus_reason`;
- `rationale` and `missing_requirements`.

When a field is null, state that it was not issued or not evaluated. Never rewrite
null as zero, C, false, low confidence, or a negative scientific conclusion.

## Core principle

The report reconstructs what was measured, how each explanation was examined, and
why grading is or is not currently allowed. It does not fill scientific gaps with
plausible prose.
