# Decision Audit Skill

## Purpose
Act as an adversarial reviewer after decision and confidence calibration.

## Procedure
1. Confirm that alternative hypotheses remain represented.
2. Enforce upstream observation/evidence audit failures by abstaining.
3. Verify declared selection score and margin thresholds.
4. Search for unresolved high-priority evidence and explicit counter-evidence.
5. Downgrade confidence when coverage is limited; abstain on blocking failures.

## Guardrail
The audit may preserve, downgrade, or abstain. It must never increase confidence.
