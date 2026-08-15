from pathlib import Path
from unittest.mock import patch

import numpy as np

from expert_agent.vision.preprocess import load_image


class _PrimaryHdu:
    data = np.ones((48, 64), dtype=np.float32)
    header: dict[str, object] = {}


class _HduList(list):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_plain_non_300_fits_does_not_require_q1_cutout_contract() -> None:
    path = Path("plain.fits")
    with (
        patch("expert_agent.vision.preprocess.resolve_input", return_value=path),
        patch("expert_agent.vision.preprocess.fits.open", return_value=_HduList([_PrimaryHdu()])),
    ):
        image = load_image(path, band="VIS")

    assert image.data.shape == (48, 64)
    assert image.input_format == "fits"
    assert not image.supplied_rms
