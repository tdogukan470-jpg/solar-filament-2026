"""Out-of-fold ablation of the post-processing stages (Table 1 of the report) with day-level bootstrap intervals.

Rows, all on the same out-of-fold probability maps and the same ground truth, PQ pooled over annotator copies exactly
like the official metric:

    threshold 0.5 + connected components  ->  + fragment linking  ->  + hand-tuned area / score filter
    ->  + learned instance filter (cross-fitted scores stored in the candidate file)

The maps are the ones named in the candidate file (`cands.py`, OOF mode); intervals are 95 % percentile bootstrap
intervals that resample observation days (photographs of one day are near-duplicates), paired across the rows.

    python -m solarfil.ablate --data_root D --oof_root work --cands cands_oof.pkl [--out ablate_rows.pkl] [--tau 0.20]
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from pycocotools import mask as mask_util

from .annotations import load_photos, photo_gt_rles
from .data import make_folds
from .postprocess import PostCfg, instances, load_avg

PLAIN = PostCfg(thr=0.5, close_px=0, min_cc=40, link_px=0, border_thr=1.01, min_area=100, min_score=0.0, seed_thr=1.01,
                fill_holes=False)
LINK = replace(PLAIN, link_px=14)
HAND_GRID = ([100, 150, 200, 250, 300, 400, 500], [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
_G: dict = {}


def _init(g):
    _G.update(g)


def _job(args):
    stem, names = args
    p = load_avg([os.path.join(_G["oof_root"], n, stem + ".npz") for n in names])
    gts = _G["gts"][stem]
    out = {}
    for tag, cfg in (("plain", PLAIN), ("link", LINK)):
        masks, feats = instances(p[0], p[1], cfg, want_feats=True)
        rles = [mask_util.encode(np.asfortranarray(m)) for m in masks]
        ious = [np.asarray(mask_util.iou(rles, g, [0] * len(g))).reshape(len(rles), len(g)) for g in gts]
        X = np.array([[f[0], f[1]] for f in feats], np.float32).reshape(-1, 2)            # log area, mean foreground prob.
        out[tag] = (X, ious, [r["counts"].decode() for r in rles])
    return stem, out


def stats(ious, keep) -> np.ndarray:
    """Pooled over annotator copies: [sum of IoU of the matches, TP, FP, FN] of the kept candidates of one photograph."""
    s = np.zeros(4)
    for m in ious:
        mm = m[keep]
        hit = mm > 0.5
        s += [float(mm[hit].sum()), int(hit.sum()), int((~hit.any(axis=1)).sum()), int((~hit.any(axis=0)).sum())]
    return s


def pq(a: np.ndarray):
    den = a[1] + 0.5 * a[2] + 0.5 * a[3]
    return a[0] / den, a[0] / max(a[1], 1), a[1] / den


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--data_root", required=True)
    a.add_argument("--oof_root", required=True, help="folder holding the OOF map directories named in the candidate file")
    a.add_argument("--cands", required=True, help="candidate file written by `python -m solarfil.cands` (OOF mode)")
    a.add_argument("--tau", type=float, default=0.20, help="score threshold of the learned filter")
    a.add_argument("--boot", type=int, default=2000)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--limit", type=int, default=0, help="debug: only the first N photographs")
    a.add_argument("--out", default="", help="optional pickle with the per-photograph statistics of every row")
    args = a.parse_args(argv)

    cands = pickle.load(open(args.cands, "rb"))
    photos = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    stems = sorted(cands["photos"])[:args.limit or None]
    gts = {s: list(photo_gt_rles(photos[s]).values()) for s in stems}
    jobs = [(s, cands["photos"][s]["maps"]) for s in stems]
    t0, res = time.time(), {}
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=({"gts": gts, "oof_root": args.oof_root},)) as ex:
        for i, (s, o) in enumerate(ex.map(_job, jobs, chunksize=4)):
            res[s] = o
            if i % 100 == 0:
                print(i, len(stems), f"{time.time() - t0:.0f}s", flush=True)
    folds = make_folds(list(photos), 5, 42).set_index("stem")["fold"].to_dict()

    # hand-tuned (min_area, min_score) rule: a two-parameter grid search on these very candidates (optimistic for the rule)
    grid = {}
    for A in HAND_GRID[0]:
        for S in HAND_GRID[1]:
            tot = np.zeros(4)
            for s in stems:
                X, ious, _ = res[s]["link"]
                keep = (np.exp(X[:, 0]) >= A - 1e-6) & (X[:, 1] >= S) if len(X) else np.zeros(0, bool)
                tot += stats(ious, keep)
            grid[(A, S)] = pq(tot)[0]
    bA, bS = max(grid, key=grid.get)
    print(f"hand-tuned grid best (min_area {bA}, min_score {bS}) PQ {grid[(bA, bS)]:.4f}", flush=True)

    rows = {"plain": {}, "link": {}, "hand": {}, "learned": {}}
    for s in stems:
        c = cands["photos"][s]
        score = {r: v for r, lv, v in zip(c["rle"], c["level"], c["score"]) if lv == cands["primary"]}
        Xp, ioup, _ = res[s]["plain"]
        rows["plain"][s] = stats(ioup, np.ones(len(Xp), bool))
        X, ious, rl = res[s]["link"]
        rows["link"][s] = stats(ious, np.ones(len(X), bool))
        none = np.zeros(0, bool)
        rows["hand"][s] = stats(ious, (np.exp(X[:, 0]) >= bA - 1e-6) & (X[:, 1] >= bS)) if len(X) else stats(ious, none)
        keep = np.array([score.get(r, np.nan) >= args.tau for r in rl], bool)
        miss = sum(1 for r in rl if r not in score)
        if miss:
            print("warning: candidates not found in the candidate file for", s, miss)
        rows["learned"][s] = stats(ious, keep) if len(X) else stats(ious, none)
    if args.out:
        pickle.dump({"rows": rows, "folds": folds, "hand": (bA, bS), "grid": grid}, open(args.out, "wb"))

    days = sorted({s[:8] for s in stems})
    didx = {d: [s for s in stems if s[:8] == d] for d in days}
    per_day = {k: np.stack([sum(rows[k][s] for s in didx[d]) for d in days]) for k in rows}
    draws = np.random.default_rng(0).integers(0, len(days), (args.boot, len(days)))
    boot = {k: np.array([pq(v[d].sum(0))[0] for d in draws]) for k, v in per_day.items()}
    names = {"plain": "threshold 0.5 + connected components", "link": "+ fragment linking (14 px)",
             "hand": f"+ hand-tuned area/score filter (A>={bA}, S>={bS})", "learned": f"+ learned filter (tau={args.tau:.2f})"}
    prev = None
    for k in ("plain", "link", "hand", "learned"):
        tot = sum(rows[k].values())
        p, sq, dq = pq(tot)
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        line = f"{names[k]:52s} PQ {p:.4f} [{lo:.4f}, {hi:.4f}]  SQ {sq:.4f} DQ {dq:.4f}  TP {int(tot[1])} FP {int(tot[2])} FN {int(tot[3])}"
        if prev:
            d = boot[k] - boot[prev]
            line += f"   dPQ {p - pq(sum(rows[prev].values()))[0]:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
        print(line, flush=True)
        prev = k
    for k in ("hand", "learned"):
        fp = [pq(sum(rows[k][s] for s in stems if folds[s] == f))[0] for f in range(5)]
        print(k, "per-fold PQ", np.round(fp, 4), "min/max", round(min(fp), 4), round(max(fp), 4))


if __name__ == "__main__":
    main()
