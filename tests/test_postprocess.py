"""Instance extraction: linking of fragments, disjoint masks, area filter."""
import unittest

import numpy as np

from solarfil.postprocess import PostCfg, instances


def two_bars(gap):
    fg = np.zeros((2048, 2048), np.float32)
    fg[1000:1016, 400:600] = 0.9
    fg[1000:1016, 600 + gap:800 + gap] = 0.9
    return fg


CFG = PostCfg(thr=0.5, min_cc=40, link_px=14, min_area=100, min_score=0.0, fill_holes=True)


class PostprocessTest(unittest.TestCase):
    def test_close_fragments_are_linked(self):
        self.assertEqual(len(instances(two_bars(8), None, CFG)), 1)

    def test_distant_filaments_stay_separate(self):
        self.assertEqual(len(instances(two_bars(40), None, CFG)), 2)

    def test_masks_are_disjoint_binary_full_size(self):
        ms = instances(two_bars(40), None, CFG)
        self.assertTrue(all(m.shape == (2048, 2048) and m.dtype == np.uint8 for m in ms))
        self.assertLessEqual(int(sum(m.astype(int) for m in ms).max()), 1)

    def test_area_filter_and_speck_removal(self):
        fg = np.zeros((2048, 2048), np.float32)
        fg[100:105, 100:105] = 0.9            # 25 px: below min_cc (speck)
        fg[500:520, 500:560] = 0.9            # 1200 px
        self.assertEqual(len(instances(fg, None, CFG)), 1)
        self.assertEqual(len(instances(fg, None, PostCfg(**{**CFG.as_dict(), "min_area": 2000}))), 0)

    def test_resolution_independent(self):
        small = np.zeros((512, 512), np.float32)
        small[250:254, 100:200] = 0.9
        self.assertEqual(len(instances(small, None, CFG)), 1)


if __name__ == "__main__":
    unittest.main()
