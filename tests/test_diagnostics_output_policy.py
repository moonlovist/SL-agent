import numpy as np

from expert_agent.vision.diagnostics import (
    grading_evidence_mask,
    target_selection_requires_diagnostic,
)


def test_grading_evidence_mask_uses_only_included_candidates():
    arrays = {
        "data": np.zeros((3, 4)),
        "model_candidate_mask_C1": np.array(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        ),
        "model_candidate_mask_C2": np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8
        ),
    }
    trace = {
        "items": [
            {"included": True, "mask_key": "model_candidate_mask_C1"},
            {"included": False, "mask_key": "model_candidate_mask_C2"},
        ]
    }

    mask = grading_evidence_mask(arrays, trace)

    assert mask.dtype == np.bool_
    assert np.argwhere(mask).tolist() == [[0, 1]]


def test_target_selection_image_is_skipped_for_clear_winner():
    selection = {
        "candidates": [
            {"routing_score": 0.80},
            {"routing_score": 0.60},
        ]
    }

    assert target_selection_requires_diagnostic(selection) is False


def test_target_selection_image_is_written_for_close_competition():
    selection = {
        "candidates": [
            {"routing_score": 0.80},
            {"routing_score": 0.76},
        ]
    }

    assert target_selection_requires_diagnostic(selection) is True


def test_target_selection_image_is_written_for_prior_conflict_or_override():
    assert target_selection_requires_diagnostic(
        {"central_prior_gate": {"offcenter_override_accepted": False}}
    )
    assert target_selection_requires_diagnostic(
        {"human_override": {"source": "review.json"}}
    )
