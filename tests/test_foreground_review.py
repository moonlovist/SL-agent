from pathlib import Path

import numpy as np

from expert_agent.vision.foreground_review import (
    _create_napari_viewer,
    _display_y_to_native,
    _native_to_fits_display,
    _native_y_to_display,
    apply_foreground_override,
    load_foreground_override,
    save_foreground_override,
)
from expert_agent.vision.preprocess import ImageData, PreprocessedImage


def _image(path: Path) -> ImageData:
    path.write_bytes(b"native-input-for-hash")
    data = np.zeros((21, 21), dtype=float)
    data[12, 15] = 8.0
    return ImageData(
        path=path, object_id="manual", input_format="fits", band="VIS",
        data=data, rms=np.ones_like(data), supplied_rms=True,
        finite_fraction=1.0, wcs=None, pixel_scale_arcsec=None, psf=None,
    )


def _prep(image: ImageData) -> PreprocessedImage:
    zeros = np.zeros_like(image.data)
    false = np.zeros_like(image.data, dtype=bool)
    segmentation = np.zeros_like(image.data, dtype=np.int32)
    segmentation[8:13, 8:13] = 1
    return PreprocessedImage(
        image=image, model=zeros, residual=image.data.copy(), residual_snr=zeros,
        source_mask=false.copy(), fit_mask=false.copy(), segmentation=segmentation,
        nucleus_yx=(10, 10), background_median=0.0, background_rms=1.0,
        residual_median=0.0, residual_robust_rms=1.0,
        model_method="synthetic", model_success=True, foreground_detected=True,
        foreground_peak_value=0.0, foreground_integrated_flux=0.0,
        foreground_ellipticity=0.0, foreground_position_angle_deg=0.0,
        foreground_axis_ratio=1.0, foreground_reliability=1.0,
        multiple_core_count=1, tidal_asymmetry=0.0,
        core_residual_fraction=0.0, flags=[],
        center_selection={"selected_sep_label": 1, "selected_centroid_xy": [10.0, 10.0]},
    )


def _test_dir() -> Path:
    path = Path.cwd() / ".codex_tmp" / "foreground-review-tests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_napari_viewer_theme_is_set_after_construction() -> None:
    calls = []

    class FakeViewer:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.theme = "dark"

    class FakeNapari:
        Viewer = FakeViewer

    viewer = _create_napari_viewer(
        FakeNapari, title="foreground review", ui_theme="light"
    )

    assert calls == [{"title": "foreground review", "show": False}]
    assert viewer.theme == "light"


def test_fits_review_display_roundtrip_preserves_native_annotation() -> None:
    native = np.arange(20).reshape(4, 5)
    shown = _native_to_fits_display(native)

    assert shown[0, 0] == native[-1, 0]
    assert np.array_equal(_native_to_fits_display(shown), native)
    for native_y in (0.0, 1.25, 3.0):
        display_y = _native_y_to_display(native_y, native.shape[0])
        assert _display_y_to_native(display_y, native.shape[0]) == native_y


def test_foreground_override_roundtrip_and_application() -> None:
    test_dir = _test_dir()
    image = _image(test_dir / "roundtrip-manual.fits")
    seed = np.zeros_like(image.data, dtype=bool)
    seed[10:15, 13:18] = True
    excluded = np.zeros_like(seed)
    excluded[11, 14] = True
    analysis_excluded = np.zeros_like(seed)
    analysis_excluded[1:3, 1:3] = True
    path = test_dir / "roundtrip-foreground_override.json"

    saved = save_foreground_override(
        path, image, (15.0, 12.0), seed, excluded, analysis_excluded
    )
    loaded = load_foreground_override(path, image)
    applied = apply_foreground_override(_prep(image), loaded)

    assert saved.input_sha256 == loaded.input_sha256
    assert loaded.center_xy == (15.0, 12.0)
    assert np.array_equal(loaded.foreground_seed, seed)
    assert np.array_equal(loaded.fit_exclude, excluded)
    assert np.array_equal(loaded.analysis_exclude, analysis_excluded)
    assert applied.nucleus_yx == (12, 15)
    assert applied.fit_mask[11, 14]
    assert applied.fit_mask[1, 1]
    assert applied.source_mask[1, 1]
    assert applied.center_selection["selection_basis"] == "human_in_the_loop_native_pixel_annotation"
    assert applied.center_selection["human_override"]["foreground_seed_pixel_count"] == 25
    assert applied.center_selection["human_override"]["analysis_exclude_pixel_count"] == 4
    assert "human_foreground_override" in applied.flags


def test_foreground_override_rejects_different_input() -> None:
    test_dir = _test_dir()
    image = _image(test_dir / "mismatch-manual.fits")
    seed = np.zeros_like(image.data, dtype=bool)
    seed[10, 10] = True
    path = test_dir / "mismatch-foreground_override.json"
    save_foreground_override(path, image, (10.0, 10.0), seed, np.zeros_like(seed))
    image.path.write_bytes(b"changed-input")

    try:
        load_foreground_override(path, image)
    except ValueError as exc:
        assert "different input" in str(exc)
    else:
        raise AssertionError("Input hash mismatch was not rejected")


def test_foreground_override_rebuilds_masks_around_human_target() -> None:
    test_dir = _test_dir()
    image = _image(test_dir / "mask-rebuild-manual.fits")
    prep = _prep(image)
    prep.source_mask[:] = True
    prep.fit_mask[:] = True
    prep.segmentation[1:3, 1:3] = 2
    seed = np.zeros_like(image.data, dtype=bool)
    seed[10:15, 13:18] = True
    override = save_foreground_override(
        test_dir / "mask-rebuild-foreground_override.json",
        image,
        (15.0, 12.0),
        seed,
        np.zeros_like(seed),
    )

    applied = apply_foreground_override(prep, override, ring_outer_px=6.0)

    assert not np.any(applied.source_mask[seed])
    assert not np.any(applied.fit_mask[seed])
    assert applied.fit_mask[10, 10]
    assert not applied.source_mask[10, 10]
    assert applied.source_mask[1, 1]
