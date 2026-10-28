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
    records = []                                        # per annotator copy: (photo, IoU sum, TP, FP, FN, semantic Dice)
    for s in stems:
        # ground-truth rows of the official format are keyed "<annotator>-<photo>": every annotator copy is scored
        # separately against the prediction of the photo (see metric.py)
        iou, dice = overlap_matrices(g[s], p.get(s.split("-", 1)[-1], []))
        n_gt, n_pred = iou.shape
        hit = iou > thr
        tp_iou += iou[hit].tolist(); tp_dice += dice[hit].tolist()
        c_fp = int((hit.sum(0) == 0).sum()) if n_gt else n_pred
        c_fn = int((hit.sum(1) == 0).sum()) if n_pred else n_gt
        fp += c_fp; fn += c_fn
        all_iou += iou[iou > 0].tolist(); all_dice += dice[dice > 0].tolist()
        gt_deg += (iou > 0).sum(1).tolist() if n_pred else [0] * n_gt
        pred_deg += (iou > 0).sum(0).tolist() if n_gt else [0] * n_pred
        pr = p.get(s.split("-", 1)[-1], [])
        if g[s] and pr:                                 # semantic Dice of the union masks of the image
            ug, up = mask_util.merge(g[s]), mask_util.merge(pr)
            i = mask_util.area(mask_util.merge([ug, up], intersect=True))
            img_dice.append(2.0 * i / max(mask_util.area(ug) + mask_util.area(up), 1))
        else:
            img_dice.append(0.0)
        records.append((s.split("-", 1)[-1], float(iou[hit].sum()), int(hit.sum()), c_fp, c_fn, img_dice[-1]))
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
        "_iou": all_iou, "_dice": all_dice, "_records": records,
    }


def bootstrap(res: dict, n_boot: int = 1000, seed: int = 0, group_chars: int = 8) -> dict:
    """Percentile 95 % intervals of PQ / SQ / DQ / mean semantic Dice by resampling observation days.

    Photos of the same day (the first `group_chars` characters of the photo id are the date) are strongly correlated,
    so the days, not the photos, are the resampling unit.  `res` is the output of `evaluate`."""
    rec = res["_records"]
    days = sorted({r[0][:group_chars] for r in rec})
    idx = {d: [i for i, r in enumerate(rec) if r[0][:group_chars] == d] for d in days}
    a = np.array([[r[1], r[2], r[3], r[4], r[5], 1.0] for r in rec])           # iou_sum, tp, fp, fn, dice, count
    per_day = np.stack([a[idx[d]].sum(0) for d in days])
    rng = np.random.default_rng(seed)
    out = {"PQ": [], "SQ": [], "DQ": [], "mean_dice_semantic": []}
    for _ in range(n_boot):
        s = per_day[rng.integers(0, len(days), len(days))].sum(0)
        den = s[1] + 0.5 * s[2] + 0.5 * s[3]
        out["PQ"].append(s[0] / den); out["SQ"].append(s[0] / max(s[1], 1)); out["DQ"].append(s[1] / den)
        out["mean_dice_semantic"].append(s[4] / s[5])
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


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
    ap.add_argument("--bootstrap", type=int, default=0, help="number of day-level bootstrap resamples for 95 %% intervals")
    a = ap.parse_args(argv)
    res = evaluate(pd.read_csv(a.gt), pd.read_csv(a.pred))
    for k, v in res.items():
        if not k.startswith("_"): print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    if a.bootstrap:
        for k, (m, lo, hi) in bootstrap(res, a.bootstrap).items():
            print(f"{k}: {m:.4f}  95% interval [{lo:.4f}, {hi:.4f}]  (bootstrap over days)")
    if a.plots: plots(res, a.plots)


if __name__ == "__main__":
    main()
