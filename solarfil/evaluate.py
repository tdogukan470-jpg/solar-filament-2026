"""Rubric-aligned evaluation: PQ, Dice/IoU distributions and one-to-many / many-to-one relations.

Mirrors the organisers' self-evaluation notebook (IoU > 0.5 matching, FP = unmatched prediction,
FN = unmatched ground truth) but works on COCO-RLE with numpy + pycocotools only (no GPU / torch needed).

CLI:  python -m solarfil.evaluate --gt gt.csv --pred pred.csv [--plots out_dir]
Both CSVs have the submission format: filament_id (<image>_<k>), segmentation_rle (COCO counts, 2048x2048).
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

SIZE = [2048, 2048]


def _rles(df: pd.DataFrame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for fid, c in zip(df["filament_id"], df["segmentation_rle"]):
        out[fid.rsplit("_", 1)[0]].append({"size": SIZE, "counts": c.encode() if isinstance(c, str) else c})
    return out


def overlap_matrices(gt: list[dict], pred: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(n_gt, n_pred) IoU and Dice matrices for one image."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred))), np.zeros((len(gt), len(pred)))
    iou = np.asarray(mask_util.iou(gt, pred, [0] * len(pred)))
    a_g = np.asarray(mask_util.area(gt), dtype=float)[:, None]
    a_p = np.asarray(mask_util.area(pred), dtype=float)[None, :]
    inter = iou * (a_g + a_p) / (1.0 + iou)            # I = IoU (A+B) / (1+IoU)
    dice = np.where(a_g + a_p > 0, 2 * inter / np.maximum(a_g + a_p, 1), 0.0)
    return iou, dice


def evaluate(gt_df: pd.DataFrame, pred_df: pd.DataFrame, thr: float = 0.5) -> dict:
    g, p = _rles(gt_df), _rles(pred_df)
    stems = sorted(g)                                   # images without ground truth are ignored
    tp_iou, tp_dice, fp, fn = [], [], 0, 0
    all_iou, all_dice, gt_deg, pred_deg, img_dice = [], [], [], [], []
    for s in stems:
        iou, dice = overlap_matrices(g[s], p.get(s, []))
        n_gt, n_pred = iou.shape
        hit = iou > thr
        tp_iou += iou[hit].tolist(); tp_dice += dice[hit].tolist()
        fp += int((hit.sum(0) == 0).sum()) if n_gt else n_pred
        fn += int((hit.sum(1) == 0).sum()) if n_pred else n_gt
        all_iou += iou[iou > 0].tolist(); all_dice += dice[dice > 0].tolist()
        gt_deg += (iou > 0).sum(1).tolist() if n_pred else [0] * n_gt
        pred_deg += (iou > 0).sum(0).tolist() if n_gt else [0] * n_pred
        if g[s] and p.get(s):                           # semantic Dice of the union masks of the image
            ug, up = mask_util.merge(g[s]), mask_util.merge(p[s])
            i = mask_util.area(mask_util.merge([ug, up], intersect=True))
            img_dice.append(2.0 * i / max(mask_util.area(ug) + mask_util.area(up), 1))
        else:
            img_dice.append(0.0)
    tp = len(tp_iou)
    den = tp + 0.5 * fp + 0.5 * fn
    return {
        "PQ": sum(tp_iou) / den if den else 0.0, "SQ": float(np.mean(tp_iou)) if tp else 0.0, "DQ": tp / den if den else 0.0,
        "TP": tp, "FP": fp, "FN": fn, "n_images": len(stems),
        "mean_dice_semantic": float(np.mean(img_dice)) if img_dice else 0.0,
        "mean_dice_overlapping_pairs": float(np.mean(all_dice)) if all_dice else 0.0,
        "mean_iou_overlapping_pairs": float(np.mean(all_iou)) if all_iou else 0.0,
        "gt_split_into_n_predictions": dict(sorted(Counter(gt_deg).items())),      # one-to-many (fragmentation)
        "pred_covering_n_gt": dict(sorted(Counter(pred_deg).items())),             # many-to-one (over-merging)
        "_iou": all_iou, "_dice": all_dice,
    }


def plots(res: dict, out_dir: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for key, name in (("_iou", "IoU"), ("_dice", "Dice")):
        plt.figure(figsize=(5, 3.2)); v = np.asarray(res[key])
        plt.hist(v, bins=50, range=(0, 1), weights=np.ones_like(v) / max(len(v), 1), color="navy", alpha=.7)
        plt.axvline(v.mean(), color="crimson", ls="--", label=f"mean={v.mean():.3f}"); plt.legend()
        plt.xlabel(name); plt.ylabel("probability"); plt.title(f"{name} of overlapping GT/pred pairs"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name.lower()}_distribution.png"), dpi=150); plt.close()
    plt.figure(figsize=(5, 3.2))
    for d, lab, c in ((res["gt_split_into_n_predictions"], "GT -> n predictions", "tab:blue"), (res["pred_covering_n_gt"], "prediction -> n GT", "tab:orange")):
        ks = sorted(d); plt.plot(ks, [d[k] for k in ks], "o-", label=lab, color=c)
    plt.yscale("log"); plt.xlabel("number of overlapping partners"); plt.ylabel("count"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "one_to_many_many_to_one.png"), dpi=150); plt.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--plots", default="")
    a = ap.parse_args(argv)
    res = evaluate(pd.read_csv(a.gt), pd.read_csv(a.pred))
    for k, v in res.items():
        if not k.startswith("_"): print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    if a.plots: plots(res, a.plots)


if __name__ == "__main__":
    main()
