"""Figures of the report: example overlays (annotations vs. predictions) and summary plots.

    # out-of-fold examples: image | annotator copies | prediction coloured by outcome (needs the annotated photos)
    python -m solarfil.figures oof --data_root ROOT --cands cands_oof.pkl --thr 0.2 --stems A B --out examples_oof.png
    # test examples: image | prediction (no ground truth exists for the test photos)
    python -m solarfil.figures test --sub submission.csv --img_dir test_images --stems A B C --out examples_test.png

Colours (all views): lime = prediction matched to an annotation at IoU > 0.5 (TP), red = unmatched prediction (FP),
yellow = annotation without a matching prediction (FN); annotator copies of one photo get cyan / orange / magenta.
"""
from __future__ import annotations

import argparse
import os
import pickle

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .annotations import load_photos, photo_gt_rles
from .evaluate import overlap_matrices
from .preprocess import make_input, read_gray

SIZE = 2048
LIME, RED, YELLOW = (60, 230, 60), (255, 50, 50), (255, 225, 0)
COPY_COLORS = [(0, 215, 255), (255, 150, 0), (235, 70, 235)]


def rle_dict(r) -> dict:
    return {"size": [SIZE, SIZE], "counts": r.encode() if isinstance(r, str) else r}


def display_image(path: str) -> np.ndarray:
    """Flat-fielded intensity (limb darkening removed) as RGB: dark ribbons = filaments, as in the network input."""
    ch = make_input(read_gray(path))[..., 1]
    return np.repeat(ch[..., None], 3, axis=2)


def best_roi(rle_lists: list[list[dict]], size: int = 1024, stride: int = 64) -> tuple[int, int]:
    """Top-left corner of the size x size window that contains most annotated / predicted filament area."""
    acc = np.zeros((SIZE, SIZE), np.float32)
    for rles in rle_lists:
        for r in rles:
            acc += mask_util.decode(r)
    ii = cv2.integral(acc)
    best, arg = -1.0, (0, 0)
    for y in range(0, SIZE - size + 1, stride):
        for x in range(0, SIZE - size + 1, stride):
            v = ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x]
            if v > best:
                best, arg = v, (y, x)
    return arg


def draw(img: np.ndarray, groups: list[tuple[list[dict], tuple, int]], roi: tuple[int, int, int] | None = None) -> np.ndarray:
    """Draw contours of RLE groups [(rles, rgb colour, thickness)] on an RGB image; roi = (y0, x0, size) crops first."""
    y0, x0, s = roi if roi else (0, 0, img.shape[0])
    out = np.ascontiguousarray(img[y0:y0 + s, x0:x0 + s]).copy()
    for rles, color, thick in groups:
        for r in rles:
            bx, by, bw, bh = mask_util.toBbox(r)
            if bx > x0 + s or bx + bw < x0 or by > y0 + s or by + bh < y0:
                continue
            m = np.ascontiguousarray(mask_util.decode(r)[y0:y0 + s, x0:x0 + s])
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cnts, -1, color, thick)
    return out


def classify(gt: dict[str, list[dict]], pred: list[dict], thr: float = 0.5):
    """Split predictions into TP / FP and every annotator copy's filaments into matched / missed (IoU > thr)."""
    tp = np.zeros(len(pred), bool)
    miss = {}
    for aid, g in gt.items():
        iou, _ = overlap_matrices(g, pred)
        hit = iou > thr
        if hit.size:
            tp |= hit.any(0)
        miss[aid] = [g[i] for i in range(len(g)) if not hit[i].any()] if hit.size else list(g)
    return tp, miss


def photo_scores(gt: dict[str, list[dict]], pred: list[dict]) -> dict:
    tp_iou, tp, fp, fn = 0.0, 0, 0, 0
    for g in gt.values():
        iou, _ = overlap_matrices(g, pred)
        hit = iou > 0.5
        tp_iou += float(iou[hit].sum()); tp += int(hit.sum())
        fp += int((hit.sum(0) == 0).sum()) if g else len(pred)
        fn += int((hit.sum(1) == 0).sum()) if pred else len(g)
    den = tp + 0.5 * fp + 0.5 * fn
    return {"PQ": tp_iou / den if den else 0.0, "TP": tp, "FP": fp, "FN": fn}


def examples_oof(data_root: str, cands: dict, stems: list[str], thr: float, out: str, roi_size: int = 1024, dpi: int = 300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    photos = load_photos(os.path.join(data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    fig, axes = plt.subplots(len(stems), 3, figsize=(7.0, 2.4 * len(stems) + 0.15), squeeze=False)
    for row, stem in zip(axes, stems):
        c = cands["photos"][stem]
        pred = [rle_dict(r) for r, lv, s in zip(c["rle"], c["level"], c["score"]) if lv == cands["primary"] and s >= thr]
        gt = photo_gt_rles(photos[stem])
        y0, x0 = best_roi(list(gt.values()) + [pred], roi_size)
        roi = (y0, x0, roi_size)
        path = next(os.path.join(data_root, "train", "train_images", stem + e) for e in (".jpeg", ".jpg", ".png")
                    if os.path.exists(os.path.join(data_root, "train", "train_images", stem + e)))
        img = display_image(path)
        tp, miss = classify(gt, pred)
        sc = photo_scores(gt, pred)
        panels = [
            draw(img, [], roi),
            draw(img, [(g, COPY_COLORS[i % 3], 3) for i, g in enumerate(gt.values())], roi),
            draw(img, [([r for r, t in zip(pred, tp) if t], LIME, 3), ([r for r, t in zip(pred, tp) if not t], RED, 3),
                       (sum(miss.values(), []), YELLOW, 2)], roi),
        ]
        for ax, p in zip(row, panels):
            ax.imshow(p, interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        row[0].set_ylabel(f"{stem[:8]} {stem[8:10]}:{stem[10:12]} UT\n{stem[-2:]}  PQ {sc['PQ']:.2f}", fontsize=7)
        row[1].set_xlabel(f"{len(gt)} annotator cop{'y' if len(gt) == 1 else 'ies'}", fontsize=7)
        row[2].set_xlabel(f"TP {sc['TP']}  FP {sc['FP']}  FN {sc['FN']}  (per copy)", fontsize=7)
    for ax, t in zip(axes[0], ["H$\\alpha$ (flat-fielded)", "annotations", "prediction"]):
        ax.set_title(t, fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.955, bottom=0.035, wspace=0.02, hspace=0.16)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def examples_test(sub_csv: str, img_dir: str, stems: list[str], out: str, dpi: int = 300, number: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(sub_csv)
    df["stem"] = df["filament_id"].str.rsplit("_", n=1).str[0]
    fig, axes = plt.subplots(1, len(stems), figsize=(7.0, 2.33), squeeze=False)
    for ax, stem in zip(axes[0], stems):
        rles = [rle_dict(r) for r in df.loc[df["stem"] == stem, "segmentation_rle"]]
        path = next(os.path.join(img_dir, stem + e) for e in (".jpeg", ".jpg", ".png") if os.path.exists(os.path.join(img_dir, stem + e)))
        img = display_image(path)
        small = cv2.resize(draw(img, [(rles, LIME, 4)]), (1024, 1024), interpolation=cv2.INTER_AREA)
        ax.imshow(small, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(f"{stem[:8]} {stem[8:10]}:{stem[10:12]} UT  {stem[-2:]}  ({len(rles)} filaments)", fontsize=7)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.08, wspace=0.02)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def summary_figure(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tables: dict[str, np.ndarray], out: str, dpi: int = 300):
    """One row of four panels: IoU / Dice of overlapping annotation-prediction pairs, one-to-many and many-to-one overlaps,
    and the morphology (skeleton length, mean width) of annotations vs. predictions.
    tables: name -> (N, len(morphology.COLUMNS)) descriptors, the first entry is drawn in black (the annotations)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .evaluate import evaluate
    from .morphology import COLUMNS

    res = evaluate(gt_df, pred_df)
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.75))
    iou, dice = np.asarray(res["_iou"]), np.asarray(res["_dice"])
    bins = np.linspace(0, 1, 26)
    for v, lab, c in ((iou, "IoU", "tab:blue"), (dice, "Dice", "tab:orange")):
        ax[0].hist(v, bins=bins, weights=np.ones_like(v) / len(v), histtype="step", color=c, lw=1.3, label=f"{lab} (mean {v.mean():.2f})")
    ax[0].axvline(0.5, color="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("overlap of intersecting pairs", fontsize=7); ax[0].set_ylabel("fraction of pairs", fontsize=7)
    ks = np.arange(0, 5)
    for off, d, lab, c in ((-0.2, res["gt_split_into_n_predictions"], "annotation $\\to$ $k$ predictions", "tab:blue"),
                           (0.2, res["pred_covering_n_gt"], "prediction $\\to$ $k$ annotations", "tab:orange")):
        tot = sum(d.values())
        f = [sum(v for kk, v in d.items() if (kk == k or (k == 4 and kk >= 4))) / tot for k in ks]
        ax[1].bar(ks + off, f, width=0.4, color=c, label=lab)
    ax[1].set_xticks(ks); ax[1].set_xticklabels(["0", "1", "2", "3", "4+"])
    ax[1].set_xlabel("number of overlapping partners $k$", fontsize=7); ax[1].set_ylabel("fraction", fontsize=7)
    ax[1].legend(fontsize=5.5, frameon=False, loc="upper right", handlelength=1.0)
    cols = ["black", "tab:blue", "tab:orange", "tab:green"]
    for a, key, label, bins_, logx in ((ax[2], "length", "skeleton length [px]", np.geomspace(10, 1500, 28), True),
                                       (ax[3], "width", "mean width [px]", np.linspace(2, 40, 28), False)):
        j = COLUMNS.index(key)
        for (name, t), c, ls in zip(tables.items(), cols, ("-", "-", "--", "-")):
            a.hist(t[:, j], bins=bins_, weights=np.ones(len(t)) / len(t), histtype="step", lw=1.3, color=c, ls=ls,
                   label=f"{name.replace('prediction', 'pred.').replace('annotations', 'annot.')}: {np.median(t[:, j]):.0f}")
        a.set_xlabel(label, fontsize=7)
        if logx:
            a.set_xscale("log")
    ax[2].set_ylabel("fraction of filaments", fontsize=7)
    for a in (ax[0], ax[2], ax[3]):                      # head-room for the legends
        a.set_ylim(0, a.get_ylim()[1] * 1.55)
    ax[1].set_ylim(0, 1.25)
    ax[0].legend(fontsize=6, frameon=False, loc="upper left")
    ax[2].legend(fontsize=5.5, frameon=False, loc="upper left"); ax[3].legend(fontsize=5.5, frameon=False, loc="upper right")
    for a in ax:
        a.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return res


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof"); o.add_argument("--data_root", required=True); o.add_argument("--cands", required=True)
    o.add_argument("--thr", type=float, default=0.20); o.add_argument("--stems", nargs="+", required=True); o.add_argument("--out", required=True)
    t = sub.add_parser("test"); t.add_argument("--sub", required=True); t.add_argument("--img_dir", required=True)
    t.add_argument("--stems", nargs="+", required=True); t.add_argument("--out", required=True)
    args = a.parse_args(argv)
    if args.cmd == "oof":
        examples_oof(args.data_root, pickle.load(open(args.cands, "rb")), args.stems, args.thr, args.out)
    else:
        examples_test(args.sub, args.img_dir, args.stems, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
