"""Regression target and PQ bookkeeping of the learned re-scorer."""
import unittest

import numpy as np

from solarfil.rescore import expected_gain, pq_of, select_classes


class RescoreTest(unittest.TestCase):
    def test_expected_gain_is_mean_over_copies_of_iou_times_match(self):
        copy_a = np.array([[0.8, 0.0], [0.0, 0.3]])        # candidate 0 matches (0.8), candidate 1 does not (0.3 <= 0.5)
        copy_b = np.array([[0.6, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(expected_gain([copy_a, copy_b], 2), [0.7, 0.0])

    def test_pq_of_matches_hand_computation(self):
        ious = [np.array([[0.8, 0.0], [0.0, 0.0], [0.0, 0.2]])]     # 3 candidates, 2 GT: 1 TP, 2 FP, 1 FN
        s = pq_of([(ious, np.array([True, True, True]))])
        self.assertEqual((s.tp, s.fp, s.fn), (1, 2, 1))
        self.assertAlmostEqual(s.pq, 0.8 / (1 + 1 + 0.5))
        s = pq_of([(ious, np.array([True, False, False]))])          # dropping the false positives
        self.assertAlmostEqual(s.pq, 0.8 / (1 + 0 + 0.5))

    def test_class_selection_drops_empty_copies(self):
        m = [np.ones((2, 3)), np.ones((2, 2))]
        out = select_classes(m, [np.array([1, 2, 3]), np.array([3, 4])], {1, 2})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
