"""Exact re-implementation of the official Panoptic Quality metric (self-evaluation notebook).

Official behaviour reproduced here:
  * GT rows are keyed "<annotator>-<stem>_<k>", prediction rows "<stem>_<k>".
  * For every annotator copy of a photo, ALL predictions of that photo are compared with
    that annotator's filaments; TP iff IoU > 0.5 (strict), unmatched preds = FP,
    unmatched GT = FN.  PQ = sum(IoU_TP) / (TP + FP/2 + FN/2), pooled over everything.
  * Photos without GT (e.g. the other half of the test set) are ignored.
IoUs are computed on RLEs with pycocotools (identical numbers, ~100x faster than dense
2048x2048 matmuls).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_util

SIZE = [2048, 2048]


def _as_rle(r):
    if isinstance(r, dict):
        return r
    return {"size": SIZE, "counts": r.encode() if isinstance(r, str) else r}


@dataclass
class PQStats:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, o: "PQStats"):
        self.iou_sum += o.iou_sum
        self.tp += o.tp
        self.fp += o.fp
        self.fn += o.fn
        return self

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / d if d > 0 else 0.0

    @property
    def sq(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def dq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else 0.0

    def __repr__(self):
        return (f"PQ={self.pq:.4f} (SQ={self.sq:.4f} DQ={self.dq:.4f}) "
                f"TP={self.tp} FP={self.fp} FN={self.fn}")


def image_stats(gt_rles: list, pred_rles: list) -> PQStats:
    """Stats for one annotator copy vs. the predictions of that photo."""
    n_gt, n_pred = len(gt_rles), len(pred_rles)
    if n_gt == 0:
        return PQStats(fp=n_pred)
    if n_pred == 0:
        return PQStats(fn=n_gt)
    g = [_as_rle(r) for r in gt_rles]
    p = [_as_rle(r) for r in pred_rles]
    iou = np.asarray(mask_util.iou(p, g, [0] * n_gt))  # (n_pred, n_gt)
    hit = iou > 0.5
    return PQStats(
        iou_sum=float(iou[hit].sum()),
        tp=int(hit.sum()),
        fp=int((hit.sum(axis=1) == 0).sum()),
        fn=int((hit.sum(axis=0) == 0).sum()),
    )


def photo_stats(gt_by_annotator: dict, pred_rles: list) -> PQStats:
    s = PQStats()
    for rles in gt_by_annotator.values():
        if len(rles):
            s.add(image_stats(rles, pred_rles))
    return s


def score_frames(gt_df, pred_df) -> PQStats:
    """Score two DataFrames in the official submission format."""
    gt = defaultdict(lambda: defaultdict(list))
    for fid, rle in zip(gt_df["filament_id"], gt_df["segmentation_rle"]):
        ann_img = fid.split("_", 1)[0]
        stem = ann_img.split("-", 1)[1]
        gt[stem][ann_img].append(rle)
    pred = defaultdict(list)
    for fid, rle in zip(pred_df["filament_id"], pred_df["segmentation_rle"]):
        pred[fid.split("_", 1)[0]].append(rle)
    total = PQStats()
    for stem, anns in gt.items():
        total.add(photo_stats(anns, pred.get(stem, [])))
    return total
