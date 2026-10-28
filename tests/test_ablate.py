"""Bookkeeping of the ablation table: statistics pooled over annotator copies, PQ / SQ / DQ."""
import unittest

import numpy as np

from solarfil.ablate import pq, stats


class AblateTest(unittest.TestCase):
    def test_stats_pool_over_annotator_copies(self):
        # 3 candidates, two annotator copies with 2 and 1 filaments
        copy_a = np.array([[0.8, 0.0], [0.0, 0.0], [0.0, 0.2]])      # 1 match (0.8), candidate 1 matches nothing, GT 1 missed
        copy_b = np.array([[0.0], [0.6], [0.0]])                      # candidate 1 matches the single filament (0.6)
        s = stats([copy_a, copy_b], np.array([True, True, True]))
        # IoU sum 1.4, TP 2, FP: (cand 1, cand 2) in copy a + (cand 0, cand 2) in copy b = 4, FN: GT 1 of copy a = 1
        np.testing.assert_allclose(s, [1.4, 2, 4, 1])
        s = stats([copy_a, copy_b], np.array([True, False, False]))   # keep only candidate 0
        np.testing.assert_allclose(s, [0.8, 1, 1, 2])       # copy a: TP, FN; copy b: FP, FN

    def test_pq_sq_dq(self):
        p, sq, dq = pq(np.array([1.4, 2, 4, 1]))
        self.assertAlmostEqual(dq, 2 / (2 + 2 + 0.5))
        self.assertAlmostEqual(sq, 0.7)
        self.assertAlmostEqual(p, sq * dq)


if __name__ == "__main__":
    unittest.main()
