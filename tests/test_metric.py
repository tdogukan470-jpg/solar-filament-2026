"""The metric re-implementation must follow the organisers' rules (strict IoU > 0.5, pooling over annotator copies)."""
import unittest

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from solarfil.metric import image_stats, score_frames


def rect(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_util.encode(np.asfortranarray(m))


def frame(rows):
    return pd.DataFrame([{"filament_id": k, "segmentation_rle": r["counts"].decode()} for k, r in rows])


class MetricTest(unittest.TestCase):
    def test_perfect_prediction(self):
        g = [rect(100, 200, 100, 300)]
        s = image_stats(g, g)
        self.assertAlmostEqual(s.pq, 1.0)
        self.assertEqual((s.tp, s.fp, s.fn), (1, 0, 0))

    def test_iou_threshold_is_strict(self):
        s = image_stats([rect(0, 100, 0, 100)], [rect(0, 100, 0, 50)])           # IoU is exactly 0.5 -> no match
        self.assertEqual((s.tp, s.fp, s.fn), (0, 1, 1))

    def test_pq_formula(self):
        gt = [rect(0, 100, 0, 100), rect(300, 400, 300, 400)]
        pr = [rect(0, 100, 0, 80), rect(900, 1000, 900, 1000)]                    # one TP (IoU 0.8), one FP, one FN
        s = image_stats(gt, pr)
        self.assertAlmostEqual(s.pq, 0.8 / (1 + 0.5 + 0.5))
        self.assertAlmostEqual(s.sq, 0.8)
        self.assertAlmostEqual(s.dq, 1 / 2.0)

    def test_empty_inputs(self):
        self.assertEqual(image_stats([], [rect(0, 10, 0, 10)]).fp, 1)
        self.assertEqual(image_stats([rect(0, 10, 0, 10)], []).fn, 1)

    def test_annotator_copies_are_pooled(self):
        a, b = rect(0, 100, 0, 100), rect(500, 600, 500, 600)
        gt = frame([("010101-photo_1", a), ("010102-photo_1", b)])                 # two annotators, different filaments
        pr = frame([("photo_1", a)])
        s = score_frames(gt, pr)
        self.assertEqual((s.tp, s.fp, s.fn), (1, 1, 1))                           # copy 1: TP; copy 2: FP and FN
        self.assertAlmostEqual(s.pq, 1.0 / 2.0)


if __name__ == "__main__":
    unittest.main()
