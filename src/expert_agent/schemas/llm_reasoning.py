"""Structured hypotheses and votes, independent of the rule-based grading schema."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CitedMeasurement(StrictModel):
    observation_ref: str = Field(min_length=2)
    # Filled by deterministic tools when absent from a model response.
    observed_value: Any = None


def model_output_schema(schema_class, allowed_refs=None) -> dict:
    schema = schema_class.model_json_schema()
    schema['$defs']['CitedMeasurement']['properties'].pop('observed_value', None)
    if allowed_refs is not None:
        schema['$defs']['CitedMeasurement']['properties']['observation_ref']['enum'] = allowed_refs
    return schema


class CitedEvidence(StrictModel):
    statement: str = Field(min_length=1, max_length=2000)
    measurements: list[CitedMeasurement] = Field(min_length=1, max_length=12)


class PhysicalHypothesis(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    family: Literal["lens", "non_lens"]
    configuration: str = Field(min_length=1, max_length=200)
    status: Literal["supported", "possible", "disfavored", "unresolved"]
    supporting_evidence: list[CitedEvidence] = Field(max_length=12)
    counter_evidence: list[CitedEvidence] = Field(max_length=12)
    expected_but_unmeasured: list[str] = Field(max_length=12)


class StructureReview(StrictModel):
    structure_id: str = Field(min_length=1, max_length=80)
    interpretation: Literal["lens_compatible", "foreground_compatible", "ambiguous", "artifact", "unresolved"]
    evidence: list[CitedEvidence] = Field(min_length=1, max_length=6)


class HypothesisDossier(StrictModel):
    structure_reviews: list[StructureReview] = Field(max_length=100)
    hypotheses: list[PhysicalHypothesis] = Field(min_length=2, max_length=8)
    missing_information: list[str] = Field(max_length=20)

    @model_validator(mode="after")
    def competing_families(self):
        if {item.family for item in self.hypotheses} != {"lens", "non_lens"}:
            raise ValueError("Both lens and non-lens alternatives are required")
        if len({item.id for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("Hypothesis IDs must be unique")
        return self


class LLMVote(StrictModel):
    vote_category: Literal["A", "B", "C", "X", "insufficient_data"]
    disposition: Literal["accept", "reject", "defer"]
    rationale: str = Field(min_length=1, max_length=6000)
    supporting_evidence: list[CitedEvidence] = Field(max_length=20)
    limitations: list[str] = Field(min_length=1, max_length=20)
    competing_explanations: list[str] = Field(min_length=1, max_length=20)
    preferred_hypothesis_ids: list[str] = Field(max_length=8)

    @model_validator(mode="after")
    def evidence_and_disposition(self):
        if self.vote_category != "insufficient_data" and not self.supporting_evidence:
            raise ValueError("A categorical vote requires cited measurements")
        expected = {"A": "accept", "B": "accept", "C": "defer",
                    "X": "reject", "insufficient_data": "defer"}
        if self.disposition != expected[self.vote_category]:
            raise ValueError("Disposition contradicts the vote category")
        if any(not item.strip() for item in self.limitations + self.competing_explanations):
            raise ValueError("Limitations and competing explanations cannot be blank")
        return self
