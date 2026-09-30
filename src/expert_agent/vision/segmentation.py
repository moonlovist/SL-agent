from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from astropy.io import fits

from ..schemas.observation import (
    CandidateStructureAuditObservation,
    StructureObservation,
)
from .foreground_models import ForegroundModelComparison
from .preprocess import PreprocessedImage


HANDOFF_CODES = {
    "primary_candidate": 1,
    "recovery_candidate": 2,
    "foreground_contaminated": 3,
    "trace_candidate": 4,
}


def _numeric_id(value: str, prefix: str) -> int:
    if value.startswith(prefix):
        try:
            return int(value[len(prefix) :])
        except ValueError:
            pass
    raise ValueError(f"Expected {prefix}<integer> identifier, received {value!r}")


def _base_header(prep: PreprocessedImage) -> fits.Header:
    header = fits.Header()
    if prep.image.wcs is not None:
        header.update(prep.image.wcs.to_header(relax=True))
    header["OBJECT"] = prep.image.object_id
    header["FILTER"] = prep.image.band
    header["SEGMETH"] = "LensAgent+SEP"
    header["SEGNOTE"] = "Measured residual structures; not confirmed lens labels"
    return header


def _image_hdu(
    prep: PreprocessedImage,
    data: np.ndarray,
    name: str,
    bunit: str,
) -> fits.ImageHDU:
    header = _base_header(prep)
    header["BUNIT"] = bunit
    return fits.ImageHDU(data=data, header=header, name=name)


def write_layered_segmentation_fits(
    prep: PreprocessedImage,
    structures: Sequence[StructureObservation],
    candidate_audits: Sequence[CandidateStructureAuditObservation],
    model_candidate_masks: Mapping[str, np.ndarray],
    primary_mask: np.ndarray,
    recovery_mask: np.ndarray,
    foreground_contaminated_mask: np.ndarray,
    trace_mask: np.ndarray,
    comparison: ForegroundModelComparison,
    path: str | Path,
) -> Path:
    """Write a layered segmentation product on the native science-image grid.

    A single exclusive label image cannot represent foreground and lensed light
    that contribute to the same detector pixels.  The primary HDU therefore
    labels only promoted residual structures, while separate extensions retain
    the original scene segmentation, all model candidates, handoff classes,
    foreground model, residual, and residual S/N.
    """

    output = Path(path)
    shape = prep.image.data.shape
    structure_labels = np.zeros(shape, dtype=np.int32)
    candidate_labels = np.zeros(shape, dtype=np.int32)
    structure_overlap_pixels = 0
    candidate_overlap_pixels = 0

    for candidate_id in sorted(
        model_candidate_masks, key=lambda value: _numeric_id(value, "C")
    ):
        mask = np.asarray(model_candidate_masks[candidate_id], dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"Candidate mask {candidate_id} has shape {mask.shape}")
        overlap = mask & (candidate_labels != 0)
        candidate_overlap_pixels += int(overlap.sum())
        candidate_labels[mask & ~overlap] = _numeric_id(candidate_id, "C")

    audit_by_candidate = {item.candidate_id: item for item in candidate_audits}
    structure_masks: dict[str, np.ndarray] = {}
    for structure in sorted(structures, key=lambda value: _numeric_id(value.id, "S")):
        if not structure.source_candidate_ids:
            continue
        candidate_id = structure.source_candidate_ids[0]
        if candidate_id not in model_candidate_masks:
            continue
        mask = np.asarray(model_candidate_masks[candidate_id], dtype=bool)
        structure_masks[structure.id] = mask
        overlap = mask & (structure_labels != 0)
        structure_overlap_pixels += int(overlap.sum())
        structure_labels[mask & ~overlap] = _numeric_id(structure.id, "S")

    handoff = np.zeros(shape, dtype=np.uint8)
    # Later assignments have higher display/inspection priority.
    for mask, value in (
        (trace_mask, HANDOFF_CODES["trace_candidate"]),
        (foreground_contaminated_mask, HANDOFF_CODES["foreground_contaminated"]),
        (recovery_mask, HANDOFF_CODES["recovery_candidate"]),
        (primary_mask, HANDOFF_CODES["primary_candidate"]),
    ):
        values = np.asarray(mask, dtype=bool)
        if values.shape != shape:
            raise ValueError(f"Handoff mask has shape {values.shape}, expected {shape}")
        handoff[values] = value

    header = _base_header(prep)
    header["EXTNAME"] = "STRUCT"
    header["BUNIT"] = "label"
    header["NSOURCE"] = (len(structure_masks), "Promoted residual structures")
    header["NPIX"] = (int(np.count_nonzero(structure_labels)), "Labeled pixels")
    header["OVRLPX"] = (structure_overlap_pixels, "Overlapping structure pixels")
    header.add_history("0 is background; positive values are S<n> structure labels.")
    header.add_history("These labels are measured residual structures, not lens truth.")
    primary = fits.PrimaryHDU(data=structure_labels, header=header)

    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [primary]
    hdus.extend(
        [
            _image_hdu(
                prep,
                (structure_labels > 0).astype(np.uint8),
                "STRUCTMASK",
                "boolean",
            ),
            _image_hdu(
                prep,
                np.asarray(prep.segmentation, dtype=np.int32),
                "FIELDSEG",
                "label",
            ),
            _image_hdu(prep, candidate_labels, "CANDIDATE", "label"),
            _image_hdu(prep, handoff, "HANDOFF", "class"),
            _image_hdu(prep, np.asarray(prep.model, dtype=np.float32), "FGMODEL", "flux"),
            _image_hdu(
                prep, np.asarray(prep.residual, dtype=np.float32), "RESIDUAL", "flux"
            ),
            _image_hdu(
                prep,
                np.asarray(prep.residual_snr, dtype=np.float32),
                "RESIDSNR",
                "sigma",
            ),
            _image_hdu(
                prep, np.asarray(prep.image.rms, dtype=np.float32), "RMS", "flux"
            ),
            _image_hdu(
                prep,
                np.asarray(prep.source_mask, dtype=np.uint8),
                "SOURCEMSK",
                "boolean",
            ),
        ]
    )
    hdus[3].header["OVRLPX"] = (
        candidate_overlap_pixels,
        "Overlapping candidate pixels",
    )
    hdus[4].header["CLASS0"] = "none"
    hdus[4].header["CLASS1"] = "primary"
    hdus[4].header["CLASS2"] = "recovery"
    hdus[4].header["CLASS3"] = "foreground_contaminated"
    hdus[4].header["CLASS4"] = "trace"

    if comparison.success and comparison.model is not None:
        hdus.append(
            _image_hdu(
                prep,
                np.asarray(comparison.model, dtype=np.float32),
                "CMPMODEL",
                "flux",
            )
        )
    if comparison.success and comparison.residual is not None:
        hdus.append(
            _image_hdu(
                prep,
                np.asarray(comparison.residual, dtype=np.float32),
                "CMPRESID",
                "flux",
            )
        )

    for structure_id, mask in structure_masks.items():
        hdus.append(
            _image_hdu(
                prep,
                np.asarray(mask, dtype=np.uint8),
                f"{structure_id}MASK",
                "boolean",
            )
        )

    rows = []
    for structure in sorted(structures, key=lambda value: _numeric_id(value.id, "S")):
        candidate_id = (
            structure.source_candidate_ids[0]
            if structure.source_candidate_ids
            else ""
        )
        audit = audit_by_candidate.get(candidate_id)
        rows.append(
            (
                structure.id,
                candidate_id,
                "" if audit is None else audit.id,
                "" if audit is None else audit.handoff_class,
                int(structure.pixel_count),
                float(structure.centroid_xy[0]),
                float(structure.centroid_xy[1]),
                float(structure.integrated_snr),
                float(structure.radius_px),
                bool(structure.near_core),
            )
        )
    columns = [
        fits.Column(name="SID", format="16A", array=[row[0] for row in rows]),
        fits.Column(name="CID", format="16A", array=[row[1] for row in rows]),
        fits.Column(name="AID", format="16A", array=[row[2] for row in rows]),
        fits.Column(name="CLASS", format="32A", array=[row[3] for row in rows]),
        fits.Column(name="NPIX", format="J", array=[row[4] for row in rows]),
        fits.Column(name="XCEN", format="D", array=[row[5] for row in rows]),
        fits.Column(name="YCEN", format="D", array=[row[6] for row in rows]),
        fits.Column(name="SNR", format="D", array=[row[7] for row in rows]),
        fits.Column(name="RADIUS", format="D", array=[row[8] for row in rows]),
        fits.Column(name="NEARCORE", format="L", array=[row[9] for row in rows]),
    ]
    hdus.append(fits.BinTableHDU.from_columns(columns, name="STRUCTCAT"))

    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    return output
