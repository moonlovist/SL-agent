# Evidence Scoring Skill

## Purpose

Convert an observed fact into a transparent relationship with one hypothesis. A
measurement is not automatically lens evidence.

Every evaluated item must record:

- the hypothesis being tested;
- whether the fact supports or opposes it;
- measurement IDs from the Observation document;
- strength, reliability, and hypothesis specificity on a 0–1 scale;
- a short scientific rationale;
- a correlation group.

The signed weighted score is:

`polarity × strength × reliability × specificity`

Evidence sharing one residual image or one measurement method is correlated. The
decision stage must aggregate a correlation group once rather than count every
derived number as an independent observation.

Unavailable measurements must use `status=not_available`, carry no score, and
must not be interpreted as negative evidence.

Scores are comparison aids, not calibrated gravitational-lens probabilities.
