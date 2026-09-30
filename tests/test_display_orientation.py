import numpy as np

from expert_agent.vision.display_orientation import match_rgb_to_native


def test_rgb_orientation_matching_recovers_vertical_flip() -> None:
    yy, xx = np.indices((31, 29))
    science = (
        5.0 * np.exp(-((xx - 7.0) ** 2 + (yy - 9.0) ** 2) / 5.0)
        + 2.0 * np.exp(-((xx - 21.0) ** 2 + (yy - 24.0) ** 2) / 9.0)
    )
    gray = np.flipud(science)
    rgb = np.repeat(gray[..., None], 3, axis=2)

    matched = match_rgb_to_native(rgb, science)

    assert matched is not None
    aligned, mapping, score = matched
    assert mapping == "flip_y"
    assert score > 0.99
    assert np.allclose(aligned[..., 0], science)
