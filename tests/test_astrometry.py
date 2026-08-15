import unittest
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from expert_agent.vision.astrometry import (
    orientation_vectors,
    pixel_to_sky,
    sky_separation_and_pa,
)
from expert_agent.vision.preprocess import ImageData


class AstrometryTests(unittest.TestCase):
    def setUp(self):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1.0, 1.0]
        wcs.wcs.cd = np.asarray(
            [[-0.1 / 3600.0, 0.0], [0.0, 0.1 / 3600.0]]
        )
        wcs.wcs.crval = [10.0, -20.0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        self.image = ImageData(
            path=Path("demo.fits"),
            object_id="demo",
            input_format="fits",
            band="VIS",
            data=np.zeros((20, 20)),
            rms=np.ones((20, 20)),
            supplied_rms=True,
            finite_fraction=1.0,
            wcs=wcs,
            pixel_scale_arcsec=0.1,
            psf=None,
        )

    def test_pixel_to_sky_and_scale_distance(self):
        ra, dec = pixel_to_sky(self.image, 0.0, 0.0)
        self.assertAlmostEqual(ra, 10.0, places=6)
        self.assertAlmostEqual(dec, -20.0, places=6)
        separation, pa = sky_separation_and_pa(
            self.image, (0.0, 0.0), (10.0, 0.0)
        )
        self.assertAlmostEqual(separation, 1.0, places=4)
        self.assertAlmostEqual(pa, 270.0, places=3)

    def test_orientation_is_north_up_east_left(self):
        north, east = orientation_vectors(self.image, (10.0, 10.0))
        self.assertIsNotNone(north)
        self.assertIsNotNone(east)
        self.assertAlmostEqual(north[1], 1.0, places=4)
        self.assertAlmostEqual(east[0], -1.0, places=4)


if __name__ == "__main__":
    unittest.main()
