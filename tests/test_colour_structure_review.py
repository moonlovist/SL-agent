from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from astropy.io import fits

from expert_agent.tools.colour_structure_review import (
    Sample,
    display_stretch,
    discover_samples,
    load_annotation,
    load_agent_review_context,
    save_annotation,
    write_batch_index,
)
from expert_agent.vision.preprocess import load_image
from scripts.prepare_unresolved_colour_pair_review import compare_rows


def _sample_tree(root: Path) -> Sample:
    directory = root / "A" / "object_1"
    directory.mkdir(parents=True)
    path = directory / "object_1.fits"
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data, name="VIS_FLUX"),
            fits.ImageHDU(np.ones_like(data), name="VIS_RMS"),
        ]
    ).writeto(path)
    return Sample("A", "object_1", directory.resolve(), path.resolve())


@pytest.fixture
def review_path() -> Path:
    path = Path(".tmp") / ("colour-review-" + uuid4().hex)
    path.mkdir(parents=True)
    return path


def test_discovery_and_annotation_roundtrip(review_path: Path) -> None:
    sample = _sample_tree(review_path / "dataset")
    discovered = discover_samples(review_path / "dataset")
    assert discovered == [sample]
    image = load_image(sample.fits_path, "VIS")
    foreground = np.zeros((10, 10), dtype=bool)
    foreground[4:6, 4:6] = True
    structures = np.zeros((10, 10), dtype=np.int16)
    structures[2:4, 1:4] = 1
    structures[6:8, 6:9] = 2
    records = {
        1: {"role": "independent_image_candidate", "source_group": "SRC1", "note": "arc"},
        2: {"role": "independent_image_candidate", "source_group": "SRC1", "note": "counterimage"},
    }
    json_path = save_annotation(
        review_path / "annotations",
        review_path / "dataset",
        sample,
        image,
        foreground,
        structures,
        records,
        "accepted",
        "two-image system",
    )
    assert json_path.exists()
    restored = load_annotation(review_path / "annotations", sample, image)
    assert np.array_equal(restored["foreground"], foreground)
    assert np.array_equal(restored["structures"], structures)
    assert restored["records"][2]["source_group"] == "SRC1"
    assert restored["sample_status"] == "accepted"
    assert restored["pair_reviews"] == {}
    index = write_batch_index(review_path / "annotations", review_path / "dataset", discovered)
    assert index.exists()


def test_display_stretches_are_finite_monotonic_and_bounded() -> None:
    data = np.linspace(-5.0, 20.0, 400).reshape(20, 20)
    for method in ("linear", "asinh_weak", "asinh_strong", "log", "sqrt", "mtf"):
        shown = display_stretch(data, method, 1.0, 99.0)
        assert np.isfinite(shown).all()
        assert shown.min() >= 0.0 and shown.max() <= 1.0
        assert np.all(np.diff(shown.ravel()) >= -1e-12)


def test_pair_review_roundtrip_and_colour_comparison(review_path: Path) -> None:
    sample = _sample_tree(review_path / "dataset")
    image = load_image(sample.fits_path, "VIS")
    first = {"role": "unresolved"}
    second = {"role": "independent_image_candidate"}
    for band, first_mag, second_mag in zip(
        ("VIS", "NIR_Y", "NIR_J", "NIR_H"),
        (22.0, 22.4, 22.8, 23.2),
        (23.0, 23.4, 23.8, 24.2),
    ):
        first[f"candidate_{band}_magnitude"] = str(first_mag)
        second[f"candidate_{band}_magnitude"] = str(second_mag)
        first[f"candidate_{band}_magnitude_error"] = "0.05"
        second[f"candidate_{band}_magnitude_error"] = "0.05"
    diagnostic = compare_rows(first, second)
    assert diagnostic["current_status"] == "consistent_with_achromatic_scaling"
    masks = np.zeros((10, 10), dtype=np.int16)
    masks[2:4, 2:4] = 1
    masks[6:8, 6:8] = 2
    review = {
        "S1-S2": {
            "first_label": 1,
            "second_label": 2,
            "geometry_relation": "plausible_same_source",
            "visual_colour_relation": "visually_different",
            "contamination": "possible",
            "mask_correspondence": "corresponding",
            "note": "review",
        }
    }
    save_annotation(
        review_path / "annotations",
        review_path / "dataset",
        sample,
        image,
        np.zeros((10, 10), dtype=bool),
        masks,
        {},
        "accepted",
        "",
        review,
    )
    restored = load_annotation(review_path / "annotations", sample, image)
    assert restored["pair_reviews"]["S1-S2"]["geometry_relation"] == "plausible_same_source"


def test_geometry_boundary_review_roundtrip(review_path: Path) -> None:
    sample = _sample_tree(review_path / "dataset")
    image = load_image(sample.fits_path, "VIS")
    review = {
        "vis_morphology_grade": "B",
        "all_imaging_grade": "B",
        "geometry_features": ["only_short_or_fragmented"],
        "non_A_reasons": ["curvature_not_reliable"],
        "note": "probable but not clear",
        "agent_context_at_review": {"agent_grade": "A"},
    }
    save_annotation(
        review_path / "annotations",
        review_path / "dataset",
        sample,
        image,
        np.zeros((10, 10), dtype=bool),
        np.zeros((10, 10), dtype=np.int16),
        {},
        "accepted",
        "",
        geometry_boundary_review=review,
    )
    restored = load_annotation(review_path / "annotations", sample, image)
    assert restored["geometry_boundary_review"] == review


def test_load_agent_review_context_missing_project_is_empty(review_path: Path) -> None:
    assert load_agent_review_context(review_path / "missing", "object_1") == {}
