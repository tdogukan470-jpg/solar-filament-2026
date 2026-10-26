"""Learned instance re-scoring (false-positive filter).

The segmentation network leaves many low-quality candidate instances (FP ~ 0.5 x TP on
validation). Each candidate is described by shape / confidence features (postprocess.FEATURES)
and a gradient-boosting regressor predicts the fraction of annotators that would accept it
(IoU > 0.5). Candidates below a threshold are dropped. Model and threshold are fitted on
out-of-fold predictions only, with photo-grouped cross-validation and the official PQ metric.

python -m solarfil.rescore --data_root <root> --oof <oof_dir> [...] --postcfg postcfg.json --out rescorer.pkl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingRegressor

from .annotations import load_photos, photo_gt_rles
from .metric import PQStats
from .postprocess import PostCfg, encode, instances, load_avg

_G = {}


def _candidates(stem):
    p = load_avg([os.path.join(d, f"{stem}.npz") for d in _G["dirs"]])
    masks, feats = instances(p[0], p[1], _G["cfg"], want_feats=True)
    rles = [{"size": [2048, 2048], "counts": r.encode()} for r in encode(masks)]
    ious = []  # per annotator: (n_pred, n_gt) IoU matrix
    for gt in _G["gts"][stem].values():
        if len(gt) == 0:
            continue
        ious.append(np.asarray(mask_util.iou(rles, gt, [0] * len(gt))).reshape(len(rles), len(gt))
                    if rles else np.zeros((0, len(gt))))
    return stem, np.asarray(feats, np.float32).reshape(-1, 17), ious


def pq_subset(items, keep_fn) -> PQStats:
    tot = PQStats()
    for feats, ious, score in items:
        k = keep_fn(score) if len(score) else np.zeros(0, bool)
        for m in ious:
            mm = m[k]
            hit = mm > 0.5
            tot.iou_sum += float(mm[hit].sum())
            tot.tp += int(hit.sum())
            tot.fp += int((hit.sum(axis=1) == 0).sum())
            tot.fn += int((hit.sum(axis=0) == 0).sum())
    return tot


def labels(ious):
    if not ious or ious[0].shape[0] == 0:
        return np.zeros(0)
    return np.mean([(m > 0.5).any(axis=1) for m in ious], axis=0)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--data_root", required=True)
    a.add_argument("--oof", nargs="+", required=True)
    a.add_argument("--postcfg", required=True, help="base post-processing; min_score/min_area are relaxed")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--out", default="rescorer.pkl")
    args = a.parse_args(argv)

    base = PostCfg(**json.load(open(args.postcfg))["cfg"])
    cand_cfg = replace(base, min_score=min(base.min_score, 0.35), min_area=min(base.min_area, 100))
    photos = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    stems = sorted(set.intersection(*[{os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, "*.npz"))}
                                      for d in args.oof]))
    _G.update(dirs=args.oof, cfg=cand_cfg, gts={s: photo_gt_rles(photos[s]) for s in stems})
    with ProcessPoolExecutor(args.workers, initializer=_G.update, initargs=(dict(_G),)) as pool:
        data = {s: (f, i) for s, f, i in pool.map(_candidates, stems, chunksize=2)}
    print(f"{len(stems)} photos, {sum(len(f) for f, _ in data.values())} candidates", flush=True)

    # baseline: the hand-tuned filter
    def base_keep(f):
        return (np.exp(f[:, 0]) >= base.min_area) & (f[:, 1] >= base.min_score)
    items = [(f, i, f) for f, i in data.values()]
    print("hand-tuned filter :", pq_subset(items, lambda f: base_keep(f) if len(f) else f), flush=True)

    # photo-grouped CV
    rng = np.random.default_rng(0)
    order = rng.permutation(stems)
    fold_of = {s: k % args.folds for k, s in enumerate(order)}
    oof_items = []
    params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)
    for k in range(args.folds):
        tr = [s for s in stems if fold_of[s] != k and len(data[s][0])]
        X = np.concatenate([data[s][0] for s in tr])
        y = np.concatenate([labels(data[s][1]) for s in tr])
        mdl = HistGradientBoostingRegressor(**params).fit(X, y)
        for s in stems:
            if fold_of[s] == k:
                f, i = data[s]
                oof_items.append((f, i, mdl.predict(f) if len(f) else np.zeros(0)))
    best_t, best = None, None
    for t in np.arange(0.10, 0.71, 0.025):
        st = pq_subset(oof_items, lambda sc, t=t: sc >= t)
        if best is None or st.pq > best.pq:
            best_t, best = float(t), st
    print(f"learned rescoring (CV): thr={best_t:.3f} {best}", flush=True)

    X = np.concatenate([data[s][0] for s in stems if len(data[s][0])])
    y = np.concatenate([labels(data[s][1]) for s in stems if len(data[s][0])])
    mdl = HistGradientBoostingRegressor(**params).fit(X, y)
    with open(args.out, "wb") as f:
        pickle.dump({"model": mdl, "thr": best_t, "cand_cfg": cand_cfg.as_dict(), "cv_pq": best.pq}, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
