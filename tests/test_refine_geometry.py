"""Stage-2 crop geometry: the long axis ends up horizontal and the warp is invertible."""
import unittest

import cv2
import numpy as np

from solarfil.refine import CROP_H, CROP_W, crop_affine, unwarp_window, warp


def ellipse(cx, cy, a, b, angle):
    t = np.zeros((2048, 2048), np.uint8)
    cv2.ellipse(t, (cx, cy), (a, b), angle, 0, 360, 1, -1)
    return t


class RefineGeometryTest(unittest.TestCase):
    def test_long_axis_becomes_horizontal(self):
        seed = ellipse(900, 700, 150, 12, 33)
        crop = warp(seed.astype(np.float32), crop_affine(seed)) > 0.5
        ys, xs = np.nonzero(crop)
        self.assertGreater(xs.max() - xs.min(), 5 * (ys.max() - ys.min()))
        self.assertGreater(xs.min(), 0)
        self.assertLess(xs.max(), CROP_W - 1)

    def test_round_trip_preserves_the_seed(self):
        seed = ellipse(1200, 800, 120, 14, -50)
        m = crop_affine(seed)
        y0, x0, patch = unwarp_window(warp(seed.astype(np.float32), m), m)
        back = np.zeros_like(seed, dtype=bool)
        back[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch > 0.5
        inter, union = (back & (seed > 0)).sum(), (back | (seed > 0)).sum()
        self.assertGreater(inter / union, 0.93)

    def test_very_long_seed_is_scaled_down_never_up(self):
        long_seed = ellipse(1000, 1000, 600, 10, 10)
        self.assertLess(np.hypot(*crop_affine(long_seed)[0, :2]), 1.0)
        short_seed = ellipse(1000, 1000, 40, 6, 10)
        self.assertAlmostEqual(float(np.hypot(*crop_affine(short_seed)[0, :2])), 1.0, places=5)
        self.assertEqual(warp(short_seed.astype(np.float32), crop_affine(short_seed)).shape, (CROP_H, CROP_W))


if __name__ == "__main__":
    unittest.main()
