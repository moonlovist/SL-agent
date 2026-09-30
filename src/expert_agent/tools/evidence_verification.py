"""Executable citation checks; these do not establish scientific correctness."""
from __future__ import annotations
import math
import re


class EvidenceVerificationError(ValueError):
    def __init__(self, diagnostics: dict):
        super().__init__(diagnostics['error'])
        self.diagnostics = diagnostics


def same_measurement(actual, cited):
    if isinstance(actual, bool) or isinstance(cited, bool):
        return type(actual) is type(cited) and actual == cited
    if isinstance(actual, (int, float)) and isinstance(cited, (int, float)):
        if isinstance(actual, int) and isinstance(cited, int):
            return actual == cited
        return math.isclose(actual, cited, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(actual, list) and isinstance(cited, list):
        return len(actual) == len(cited) and all(same_measurement(a, b) for a, b in zip(actual, cited))
    if isinstance(actual, dict) and isinstance(cited, dict):
        return actual.keys() == cited.keys() and all(same_measurement(actual[k], cited[k]) for k in actual)
    return type(actual) is type(cited) and actual == cited


def available_references(document):
    """Enumerate non-missing scalar references for constrained model output."""
    references = []
    def visit(value, pointer):
        if isinstance(value, dict):
            for key, item in value.items():
                escaped = key.replace('~', '~0').replace('/', '~1')
                visit(item, pointer + '/' + escaped)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, pointer + '/' + str(index))
        elif value is not None and value != '':
            references.append(pointer)
    visit(document, '')
    return references


def resolve_pointer(document: dict, reference: str):
    if not reference.startswith("/") or re.search(r"~(?![01])", reference):
        raise ValueError("Invalid JSON Pointer")
    value = document
    for token in reference[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not token.isdecimal() or (len(token) > 1 and token.startswith("0")):
                raise ValueError("Invalid array index")
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise ValueError("Reference does not point to a measurement")
    if value is None or value == "" or value == [] or value == {}:
        raise ValueError("Reference points to missing evidence")
    return value


def verify_evidence(evidence, observation: dict) -> dict:
    count = 0
    for item in evidence:
        for citation in item.measurements:
            try:
                actual = resolve_pointer(observation, citation.observation_ref)
            except (ValueError, KeyError, IndexError):
                raise EvidenceVerificationError({"error": "invalid_or_missing_reference",
                                                 "reference": citation.observation_ref}) from None
            # Distinguish booleans from numbers, while allowing 2 and 2.0.
            if 'observed_value' not in citation.model_fields_set:
                citation.observed_value = actual
            elif not same_measurement(actual, citation.observed_value):
                raise EvidenceVerificationError({"error": "cited_value_mismatch",
                                                 "reference": citation.observation_ref,
                                                 "actual_value": actual,
                                                 "cited_value": citation.observed_value})
            count += 1
    return {"passed": True, "citations_checked": count,
            "checks": ["reference_exists", "measurement_not_missing", "value_match"],
            "float_tolerance": {"relative": 1e-12, "absolute": 1e-12},
            "values_materialized_from_observation": True,
            "scope": "data_integrity_only_not_physical_confirmation"}
