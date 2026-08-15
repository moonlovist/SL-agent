from __future__ import annotations

import numpy as np

from .preprocess import ImageData


def pixel_to_sky(
    image: ImageData, x: float, y: float
) -> tuple[float | None, float | None]:
    if image.wcs is None:
        return None, None
    try:
        sky = image.wcs.pixel_to_world(float(x), float(y))
        ra, dec = float(sky.ra.deg), float(sky.dec.deg)
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    except Exception:
        pass
    return None, None


def sky_separation_and_pa(
    image: ImageData,
    center_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> tuple[float | None, float | None]:
    """Return angular separation in arcsec and PA east of north."""
    if image.wcs is None:
        return None, None
    try:
        center = image.wcs.pixel_to_world(*center_xy)
        target = image.wcs.pixel_to_world(*target_xy)
        separation = float(center.separation(target).arcsec)
        pa = float(center.position_angle(target).deg % 360.0)
        if np.isfinite(separation) and np.isfinite(pa):
            return separation, pa
    except Exception:
        pass
    return None, None


def sky_axis_pa(
    image: ImageData, x: float, y: float, theta_rad: float
) -> float | None:
    """Transform a SEP pixel-axis direction to sky PA east of north."""
    if image.wcs is None:
        return None
    try:
        start = image.wcs.pixel_to_world(float(x), float(y))
        end = image.wcs.pixel_to_world(
            float(x + np.cos(theta_rad)), float(y + np.sin(theta_rad))
        )
        pa = float(start.position_angle(end).deg % 180.0)
        return pa if np.isfinite(pa) else None
    except Exception:
        return None


def axial_misalignment_deg(axis_pa: float, target_pa: float) -> float:
    """Smallest angle between an unoriented axis and a target sky PA."""
    return float(abs((axis_pa - target_pa + 90.0) % 180.0 - 90.0))


def orientation_vectors(
    image: ImageData, center_xy: tuple[float, float]
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Return normalized pixel-space vectors pointing north and east."""
    if image.wcs is None:
        return None, None
    try:
        import astropy.units as u

        center = image.wcs.pixel_to_world(*center_xy)
        north = center.directional_offset_by(0.0 * u.deg, 1.0 * u.arcsec)
        east = center.directional_offset_by(90.0 * u.deg, 1.0 * u.arcsec)
        cx, cy = image.wcs.world_to_pixel(center)
        nx, ny = image.wcs.world_to_pixel(north)
        ex, ey = image.wcs.world_to_pixel(east)

        def normalized(dx: float, dy: float) -> tuple[float, float]:
            length = float(np.hypot(dx, dy))
            if not np.isfinite(length) or length <= 0:
                raise ValueError("invalid orientation vector")
            return float(dx / length), float(dy / length)

        return normalized(nx - cx, ny - cy), normalized(ex - cx, ey - cy)
    except Exception:
        return None, None
