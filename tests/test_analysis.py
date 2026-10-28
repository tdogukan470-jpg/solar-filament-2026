"""Unit tests of the pure helpers of solarfil.analysis (no data needed)."""
import numpy as np
from pycocotools import mask as mask_util

from solarfil.analysis import RADII, _primary_scores, extent_failures, fates, geometry_op, iou_matrix, shrink_grow, vertex


def _rle(m):
    return mask_util.encode(np.asfortranarray(m.astype(np.uint8)))


def test_fates_four_cases():
    # candidates x annotations; candidate 1 is dropped by the filter
    iou = np.array([[0.9, 0.0, 0.3, 0.0],
                    [0.0, 0.8, 0.0, 0.05]])
    c = fates(iou, np.array([True, False]))
    assert c == {"matched": 1, "dropped": 1, "extent": 1, "missed": 1}


def test_fates_without_candidates():
    assert fates(np.zeros((0, 3)), np.zeros(0, bool)) == {"missed": 3}


def test_extent_failures_recall_and_precision():
    # 2 candidates x 3 annotations: only annotation 0 is overlapped (IoU in [0.1, 0.5]) without being matched
    iou = np.array([[0.30, 0.0, 0.05],
                    [0.00, 0.8, 0.00]])
    d = extent_failures(iou, np.array([1000.0, 400.0]), np.array([500.0, 380.0, 700.0]))
    assert d.shape == (1, 4)
    inter = 0.3 * 1500 / 1.3                               # |A and B| from IoU = I / (a + b - I)
    assert np.allclose(d[0], [500, 1000, inter / 500, inter / 1000])
    assert abs(inter / (1000 + 500 - inter) - 0.3) < 1e-12
    assert extent_failures(np.zeros((0, 2)), np.zeros(0), np.array([5.0, 6.0])).shape == (0, 4)


def test_iou_matrix_empty_and_values():
    a = np.zeros((64, 64), np.uint8); a[10:30, 10:30] = 1
    b = np.zeros((64, 64), np.uint8); b[10:30, 20:40] = 1
    iou = iou_matrix([_rle(a)], [_rle(b)])
    assert iou.shape == (1, 1) and abs(iou[0, 0] - 200 / 600) < 1e-9
    assert iou_matrix([], [_rle(b)]).shape == (0, 1) and iou_matrix([_rle(a)], []).shape == (1, 0)


def test_shrink_grow_changes_area_and_keeps_vanishing_masks():
    m = np.zeros((100, 100), np.uint8); m[20:60, 20:60] = 1
    assert shrink_grow(m, 0) is m
    assert shrink_grow(m, -2).sum() < m.sum() < shrink_grow(m, 2).sum()
    tiny = np.zeros((100, 100), np.uint8); tiny[50, 50] = 1
    out = shrink_grow(tiny, -1)
    assert out.sum() == 1                                  # stays a one-pixel mask (false positive), never empty


def test_vertex_of_a_parabola():
    radii = np.array(RADII, float)
    p = -(radii - 0.4) ** 2                                # maximum at r = 0.4
    assert abs(vertex(p) - 0.4) < 1e-9
    assert abs(vertex(-(radii + 1.3) ** 2) + 1.3) < 1e-9
    assert vertex(np.array([0.1, 0.2, 0.3, 0.4, 0.5])) == 1.0     # monotone curve: end point clamped, no extrapolation


def test_primary_scores_skip_extra_level_candidates():
    c = {"primary": 0.5, "photos": {
        "a": {"level": np.array([0.5, 0.4, 0.5], np.float32), "score": np.array([0.1, 0.9, 0.3], np.float32)},
        "b": {"level": np.array([0.5], np.float32), "score": np.array([0.7], np.float32)}}}
    assert np.allclose(_primary_scores(c), [0.1, 0.3, 0.7])


def test_geometry_ops_square_cross_and_disjointness():
    a = np.zeros((40, 40), np.uint8); a[10:20, 10:20] = 1          # 10 x 10 block
    b = np.zeros((40, 40), np.uint8); b[10:20, 21:31] = 1          # right neighbour, one empty column in between
    grown = geometry_op([a, b], "dil_sq")
    assert grown[0].sum() == 12 * 11                                           # grows by one pixel, except at the shared column
    assert (grown[0] & grown[1]).sum() == 0                                    # the shared column goes to neither
    assert (grown[0] >= a).all() and (grown[1] >= b).all()                     # every mask keeps its own pixels
    sq, cross = geometry_op([a], "dil_sq")[0], geometry_op([a], "dil_x")[0]
    assert sq.sum() == 12 * 12 and cross.sum() == 12 * 12 - 4                  # the cross kernel leaves out the four corners
    assert geometry_op([a], "ero_sq")[0].sum() == 8 * 8 and geometry_op([a], "base")[0] is a
    tiny = np.zeros((40, 40), np.uint8); tiny[5, 5] = 1
    assert geometry_op([tiny], "ero_x") == []                                  # vanished masks are dropped
