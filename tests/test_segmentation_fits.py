from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from expert_agent.vision.foreground_models import ForegroundModelComparison
from expert_agent.vision.segmentation import write_layered_segmentation_fits


def test_layered_segmentation_preserves_wcs_and_separates_products() -> None:
    shape = (12, 12)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [6.0, 6.0]
    wcs.wcs.cdelt = [-0.1 / 3600.0, 0.1 / 3600.0]
    wcs.wcs.crval = [63.9, -48.9]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    field_segmentation = np.zeros(shape, dtype=np.int32)
    field_segmentation[4:9, 4:9] = 7
    structure_mask = np.zeros(shape, dtype=bool)
    structure_mask[3, 5:8] = True
    recovery_mask = np.zeros(shape, dtype=bool)
    recovery_mask[8, 3:6] = True

    image = SimpleNamespace(
        data=np.zeros(shape),
        rms=np.ones(shape),
        object_id="synthetic",
        band="VIS",
        wcs=wcs,
    )
    prep = SimpleNamespace(
        image=image,
        segmentation=field_segmentation,
        model=np.full(shape, 2.0),
        residual=np.full(shape, 0.5),
        residual_snr=np.full(shape, 3.0),
        source_mask=np.zeros(shape, dtype=bool),
    )
    structures = [
        SimpleNamespace(
            id="S1",
            source_candidate_ids=["C1"],
            pixel_count=3,
            centroid_xy=(6.0, 3.0),
            integrated_snr=8.0,
            radius_px=3.0,
            near_core=False,
        )
    ]
    audits = [
        SimpleNamespace(
            id="CSA1",
            candidate_id="C1",
            handoff_class="primary_candidate",
        )
    ]
    output_dir = Path.cwd() / ".tmp"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"test_segmentation_{uuid4().hex}.fits"
    write_layered_segmentation_fits(
        prep=prep,
        structures=structures,
        candidate_audits=audits,
        model_candidate_masks={"C1": structure_mask, "C2": recovery_mask},
        primary_mask=structure_mask,
        recovery_mask=recovery_mask,
        foreground_contaminated_mask=np.zeros(shape, dtype=bool),
        trace_mask=np.zeros(shape, dtype=bool),
        comparison=ForegroundModelComparison(method="none", success=False),
        path=output,
    )

    with fits.open(output, checksum=True) as hdul:
        hdul.verify("exception")
        assert hdul[0].name == "STRUCT"
        assert np.count_nonzero(hdul[0].data == 1) == 3
        assert np.array_equal(hdul["FIELDSEG"].data, field_segmentation)
        assert set(np.unique(hdul["CANDIDATE"].data)) == {0, 1, 2}
        assert set(np.unique(hdul["HANDOFF"].data)) == {0, 1, 2}
        assert hdul["STRUCTCAT"].data[0]["SID"].strip() == "S1"
        output_wcs = WCS(hdul[0].header)
        point = np.asarray([[5.5, 5.5]])
        assert np.allclose(
            output_wcs.all_pix2world(point, 0),
            wcs.all_pix2world(point, 0),
        )
    output.unlink()
