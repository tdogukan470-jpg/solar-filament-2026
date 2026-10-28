"""Rubric evaluation: Dice / IoU of a perfect prediction and the annotator-prefixed ground-truth format."""
import unittest

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from solarfil.evaluate import bootstrap, evaluate


def rect(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_util.encode(np.asfortranarray(m))["counts"].decode()


class EvaluateTest(unittest.TestCase):
    def test_perfect_and_fragmented_prediction(self):
        gt = pd.DataFrame({"filament_id": ["010101-p_1", "010101-p_2"], "segmentation_rle": [rect(0, 50, 0, 200), rect(400, 450, 0, 200)]})
        res = evaluate(gt, pd.DataFrame({"filament_id": ["p_1", "p_2"], "segmentation_rle": gt["segmentation_rle"].tolist()}))
        self.assertAlmostEqual(res["PQ"], 1.0)
        self.assertAlmostEqual(res["mean_dice_semantic"], 1.0)
        self.assertEqual(res["gt_split_into_n_predictions"], {1: 2})
        frag = pd.DataFrame({"filament_id": ["p_1", "p_2", "p_3"], "segmentation_rle": [rect(0, 50, 0, 60), rect(0, 50, 60, 200), rect(400, 450, 0, 200)]})
        res = evaluate(gt, frag)                                                   # first filament split into 2 pieces
        self.assertEqual(res["gt_split_into_n_predictions"], {1: 1, 2: 1})
        self.assertEqual(res["TP"], 2)                                             # piece 2 has IoU 0.7 > 0.5 with filament 1

    def test_bootstrap_interval_of_a_perfect_prediction(self):
        gt = pd.DataFrame({"filament_id": ["010101-20110101000000Ch_1", "010101-20110202000000Ch_1"],
                           "segmentation_rle": [rect(0, 50, 0, 200), rect(400, 450, 0, 200)]})
        pr = pd.DataFrame({"filament_id": ["20110101000000Ch_1", "20110202000000Ch_1"], "segmentation_rle": gt["segmentation_rle"].tolist()})
        b = bootstrap(evaluate(gt, pr), n_boot=50)
        self.assertEqual(b["PQ"], (1.0, 1.0, 1.0))
        self.assertEqual(b["mean_dice_semantic"], (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
