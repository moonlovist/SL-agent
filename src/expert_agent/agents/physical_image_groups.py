"""Conservative, non-destructive grouping of measured structures.

The vision layer records pixel islands.  A physical lensed image can be split
into more than one island, especially near the PSF scale.  This module records
a grouping hypothesis without unioning masks, filling gaps, or deleting any
of the original structures.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, pi

import numpy as np

from ..core.config import ReasoningConfig
from ..schemas.observation import ObservationDocument, StructureObservation


@dataclass(frozen=True)
class PhysicalImageGroup:
    id: str
    members: tuple[StructureObservation, ...]
    centroid_xy: tuple[float, float]
    radius_px: float
    position_angle_deg: float
    pixel_count: int
    psf_scale_eligible: bool

    @property
    def structure_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.members)

    def to_trace(self) -> dict[str, object]:
        return {
            "id": self.id,
            "structure_ids": list(self.structure_ids),
            "centroid_xy": list(self.centroid_xy),
            "radius_px": self.radius_px,
            "position_angle_deg": self.position_angle_deg,
            "pixel_count": self.pixel_count,
            "psf_scale_eligible": self.psf_scale_eligible,
            "pixel_masks_modified": False,
        }


def _angle_distance(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _psf_fwhm(document: ObservationDocument) -> float | None:
    value = document.supplementary_diagnostics.get("psf_fwhm_px")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0 else None


def _is_psf_scale(
    item: StructureObservation,
    *,
    psf_fwhm: float | None,
    config: ReasoningConfig,
) -> bool:
    if psf_fwhm is None:
        return False
    return bool(
        item.angular_extent_deg <= config.compact_image_max_extent_deg
        and item.pixel_count <= pi * psf_fwhm**2
        and not item.near_core
    )


def _can_group_pair(
    first: StructureObservation,
    second: StructureObservation,
    *,
    psf_fwhm: float | None,
    config: ReasoningConfig,
) -> bool:
    if psf_fwhm is None:
        return False
    if not (
        _is_psf_scale(first, psf_fwhm=psf_fwhm, config=config)
        and _is_psf_scale(second, psf_fwhm=psf_fwhm, config=config)
    ):
        return False
    distance = hypot(
        first.centroid_xy[0] - second.centroid_xy[0],
        first.centroid_xy[1] - second.centroid_xy[1],
    )
    return bool(
        distance <= config.physical_group_max_separation_psf * psf_fwhm
        and abs(first.median_radius_px - second.median_radius_px)
        <= config.physical_group_max_separation_psf * psf_fwhm
        and _angle_distance(first.position_angle_deg, second.position_angle_deg)
        <= config.physical_group_max_polar_separation_deg
    )


def _group_distance(
    first: tuple[StructureObservation, ...],
    second: tuple[StructureObservation, ...],
) -> float:
    return min(
        hypot(a.centroid_xy[0] - b.centroid_xy[0], a.centroid_xy[1] - b.centroid_xy[1])
        for a in first
        for b in second
    )


def build_physical_image_groups(
    document: ObservationDocument,
    config: ReasoningConfig,
) -> list[PhysicalImageGroup]:
    """Return complete-link PSF-fragment groups without changing structures."""
    psf_fwhm = _psf_fwhm(document)
    clusters: list[tuple[StructureObservation, ...]] = [
        (item,) for item in sorted(document.structures, key=lambda item: item.id)
    ]
    while True:
        eligible: list[tuple[float, int, int]] = []
        for first_index in range(len(clusters)):
            for second_index in range(first_index + 1, len(clusters)):
                first = clusters[first_index]
                second = clusters[second_index]
                # Complete-link: every cross-pair must independently satisfy
                # the conservative fragment relation.  No proximity chains.
                if all(
                    _can_group_pair(a, b, psf_fwhm=psf_fwhm, config=config)
                    for a in first
                    for b in second
                ):
                    eligible.append(
                        (_group_distance(first, second), first_index, second_index)
                    )
        if not eligible:
            break
        _, first_index, second_index = min(eligible)
        merged = tuple(
            sorted(
                (*clusters[first_index], *clusters[second_index]),
                key=lambda item: item.id,
            )
        )
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {first_index, second_index}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda cluster: tuple(item.id for item in cluster))

    groups: list[PhysicalImageGroup] = []
    for index, members in enumerate(clusters, 1):
        weights = np.asarray([item.pixel_count for item in members], dtype=float)
        xs = np.asarray([item.centroid_xy[0] for item in members], dtype=float)
        ys = np.asarray([item.centroid_xy[1] for item in members], dtype=float)
        radii = np.asarray([item.median_radius_px for item in members], dtype=float)
        angles = np.radians([item.position_angle_deg for item in members])
        angle = degrees(
            atan2(
                float(np.sum(weights * np.sin(angles))),
                float(np.sum(weights * np.cos(angles))),
            )
        ) % 360.0
        groups.append(
            PhysicalImageGroup(
                id=f"IMG{index}",
                members=members,
                centroid_xy=(
                    float(np.average(xs, weights=weights)),
                    float(np.average(ys, weights=weights)),
                ),
                radius_px=float(np.average(radii, weights=weights)),
                position_angle_deg=angle,
                pixel_count=int(weights.sum()),
                psf_scale_eligible=all(
                    _is_psf_scale(item, psf_fwhm=psf_fwhm, config=config)
                    for item in members
                ),
            )
        )
    return groups


def cross_quad_is_supported(
    groups: list[PhysicalImageGroup],
    config: ReasoningConfig,
) -> bool:
    """Recognize a conservative four-sector cross, not fold/cusp quads."""
    if len(groups) != 4 or not all(group.psf_scale_eligible for group in groups):
        return False
    radii = [group.radius_px for group in groups]
    if min(radii) <= 0 or max(radii) / min(radii) > config.configuration_max_radius_ratio:
        return False
    angles = sorted(group.position_angle_deg % 360.0 for group in groups)
    gaps = [second - first for first, second in zip(angles, angles[1:])]
    gaps.append(angles[0] + 360.0 - angles[-1])
    return bool(
        min(gaps) >= config.cross_quad_min_gap_deg
        and max(gaps) <= config.cross_quad_max_gap_deg
    )
