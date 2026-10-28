"""Learned instance re-scoring (false-positive filter).

The segmentation network leaves many low-quality candidate instances (about one false positive per two true
positives).  Every candidate (a connected component after thresholding and fragment linking) is described by
shape / confidence / context features (postprocess.FEATURES_EXT) and a gradient-boosting regressor predicts its
expected contribution to the Panoptic-Quality numerator, i.e. the mean over the annotator copies of
IoU * 1[IoU > 0.5].  Keeping a candidate only pays off when this expected contribution exceeds PQ / 2
(the marginal rule of  PQ = sum(IoU_TP) / (TP + FP/2 + FN/2)), so a single threshold on the predicted score
replaces the hand-tuned (min_area, min_score) rule of configs/postcfg.json.

Everything is fitted on out-of-fold probability maps only (maps of photos the networks never saw in training):
photos are grouped by acquisition day for cross-validation and the threshold maximises the pooled official PQ of
the held-out predictions.  The resulting file stores the training table (features, targets), not a pickled model,
so it does not depend on the scikit-learn version: `load_rescorer` re-fits the (deterministic) model in seconds.

    python -m solarfil.rescore --data_root <root> --postcfg configs/postcfg.json \\
        --view work/oof_Unet_efficientnet-b4_s1536 \\
        --view work/oof_Unet_efficientnet-b4_s1536,work/oof_Unet_efficientnet-b4_s2048 \\
        --gt all=1,2,3,4 --gt c12=1,2 --out configs/rescorer_{name}.npz

`--view` may be repeated; the directories of one view are averaged exactly like the model ensemble used at test
time, and the candidates of all views are pooled for training (a view with more members is closer to the final
ensemble).  `--gt NAME=CLASSES` selects which MAGFiLO classes (1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous)
count as ground truth for targets and for the PQ used to pick the threshold; `all=1,2,3,4` is every annotated
filament, exactly like the official self-evaluation notebook on the training split.  One rescorer file is written
per `--gt` (the candidate table is computed once).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .annotations import load_photos, photo_gt_rles
from .metric import PQStats
from .postprocess import FEATURES_EXT, PostCfg, candidate_features, instances, load_avg, upsample

PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
              early_stopping=False, random_state=0)
THR_GRID = np.round(np.arange(0.08, 0.46, 0.02), 2)
_G: dict = {}


def _init(g: dict):
    _G.update(g)


def fit_model(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**PARAMS).fit(X, y)


def save_rescorer(path: str, X, y, thr: float, cand_cfg: PostCfg, cv_pq: float, gt_categories) -> None:
    np.savez_compressed(path, X=np.asarray(X, np.float32), y=np.asarray(y, np.float32), thr=np.float64(thr),
                        cv_pq=np.float64(cv_pq), names=np.array(FEATURES_EXT), params=json.dumps(PARAMS),
                        cand_cfg=json.dumps(cand_cfg.as_dict()), gt_categories=np.asarray(sorted(gt_categories)))


def load_rescorer(path: str) -> dict:
    """-> {"model", "thr", "cand_cfg", "cv_pq", "gt_categories"} as expected by postprocess.instances_rle."""
    z = np.load(path, allow_pickle=False)
    if [str(n) for n in z["names"]] != FEATURES_EXT:
        raise ValueError(f"{path} was built with another feature set")
    return {"model": fit_model(z["X"], z["y"]), "thr": float(z["thr"]), "cv_pq": float(z["cv_pq"]),
            "cand_cfg": json.loads(str(z["cand_cfg"])), "gt_categories": [int(c) for c in z["gt_categories"]]}


def _candidates(job):
    """(view, stem) -> candidate features and one (n_candidates, n_gt) IoU matrix per annotator copy."""
    v, stem = job
    p = load_avg([os.path.join(d, f"{stem}.npz") for d in _G["views"][v]])
    masks, feats = instances(p[0], p[1], _G["cfg"], want_feats=True)
    if not masks:
        return v, stem, np.zeros((0, len(FEATURES_EXT)), np.float32), [np.zeros((0, len(g))) for g in _G["gts"][stem]]
    X = candidate_features(upsample(p[0]), masks, feats)
    rles = [mask_util.encode(np.asfortranarray(m)) for m in masks]
    ious = [np.asarray(mask_util.iou(rles, g, [0] * len(g))).reshape(len(rles), len(g)) for g in _G["gts"][stem]]
    return v, stem, X, ious


def select_classes(ious: list, cats: list, allowed) -> list:
    """Keep only the ground-truth filaments of the allowed classes; annotator copies left empty disappear (as in the
    official ground-truth table)."""
    out = []
    for m, c in zip(ious, cats):
        sel = np.isin(c, list(allowed))
        if sel.any():
            out.append(m[:, sel])
    return out


def expected_gain(ious: list, n: int) -> np.ndarray:
    """Regression target: mean over annotator copies of IoU * 1[IoU > 0.5] for each of the n candidates."""
    acc = np.zeros(n)
    for m in ious:
        best = m.max(axis=1) if m.shape[1] and n else np.zeros(n)
        acc += np.where(best > 0.5, best, 0.0)
    return acc / max(len(ious), 1)


def pq_of(items) -> PQStats:
    """Official PQ (pooled over annotator copies) of the kept candidates.
    items: (ious, keep) per photo; ious = list of (n_candidates, n_gt) IoU matrices, keep = bool mask over candidates."""
    tot = PQStats()
    for ious, keep in items:
        for m in ious:
            mm = m[keep]
            hit = mm > 0.5
            tot.iou_sum += float(mm[hit].sum())
            tot.tp += int(hit.sum())
            tot.fp += int((~hit.any(axis=1)).sum())
            tot.fn += int((~hit.any(axis=0)).sum())
    return tot


def fit_gt_set(name, allowed, rows, n_views, base, cand_cfg, folds, out, thr_override=None):
    """Cross-validate, choose the threshold and save one rescorer for the ground-truth class set `allowed`."""
    rows = [(v, s, X, select_classes(ious, cats, allowed)) for v, s, X, ious, cats in rows]
    rows = [(v, s, X, ious, expected_gain(ious, len(X))) for v, s, X, ious in rows if ious]
    print(f"\n=== {name}: classes {sorted(allowed)}, {len({r[1] for r in rows})} photos, "
          f"{sum(len(r[2]) for r in rows)} candidates", flush=True)

    def hand_keep(X):                                                  # the deployed rule, applied to the candidates
        return (np.exp(X[:, 0]) >= base.min_area - 1e-6) & (X[:, 1] >= base.min_score)
    for v in range(n_views):
        print(f"view {v}: hand-tuned filter   ", pq_of([(r[3], hand_keep(r[2])) for r in rows if r[0] == v]), flush=True)

    # day-grouped K-fold over photos (all views of one photo stay together)
    stems = sorted({r[1] for r in rows})
    fold_of = {}
    for k, (_, te) in enumerate(GroupKFold(folds).split(stems, groups=[s[:8] for s in stems])):
        fold_of.update({stems[i]: k for i in te})
    held = []                                                          # (view, ious, held-out scores)
    for k in range(folds):
        tr = [r for r in rows if fold_of[r[1]] != k and len(r[2])]
        mdl = fit_model(np.concatenate([r[2] for r in tr]), np.concatenate([r[4] for r in tr]))
        held += [(r[0], r[3], mdl.predict(r[2]) if len(r[2]) else np.zeros(0)) for r in rows if fold_of[r[1]] == k]
    sweep = {float(t): pq_of([(i, sc >= t) for _, i, sc in held]) for t in THR_GRID}
    best_t = max(sweep, key=lambda t: sweep[t].pq)
    print(f"learned filter (CV, pooled views) thr={best_t:.2f}  {sweep[best_t]}", flush=True)
    if thr_override is not None:                                       # e.g. the centre of the PQ plateau, see README
        best_t = float(min(sweep, key=lambda t: abs(t - thr_override)))
        print(f"threshold fixed to {best_t:.2f}  {sweep[best_t]}", flush=True)
    for v in range(n_views):
        pv = {float(t): pq_of([(i, sc >= t) for vv, i, sc in held if vv == v]) for t in THR_GRID}
        tb = max(pv, key=lambda t: pv[t].pq)
        print(f"  view {v}: at thr {best_t:.2f}  {pv[best_t]}   (own optimum {tb:.2f}: PQ {pv[tb].pq:.4f})", flush=True)

    keep = [r for r in rows if len(r[2])]
    X, y = np.concatenate([r[2] for r in keep]), np.concatenate([r[4] for r in keep])
    save_rescorer(out, X, y, best_t, cand_cfg, sweep[best_t].pq, allowed)
    print(f"saved {out}: {X.shape[0]} training candidates, {X.shape[1]} features", flush=True)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--data_root", required=True)
    a.add_argument("--view", action="append", required=True, help="comma-separated OOF dirs (averaged); repeatable")
    a.add_argument("--postcfg", required=True, help="deployed post-processing (extraction part + hand-tuned filter)")
    a.add_argument("--gt", action="append", default=[], help="NAME=CLASSES, e.g. all=1,2,3,4 or c12=1,2 (repeatable)")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--thr", type=float, default=None, help="use this score threshold instead of the PQ-optimal one of the sweep")
    a.add_argument("--limit", type=int, default=0, help="debug: only the first N photos of every view")
    a.add_argument("--out", default="rescorer_{name}.npz")
    args = a.parse_args(argv)

    gt_sets = {}
    for g in args.gt or ["all=1,2,3,4"]:
        name, cls = g.split("=")
        gt_sets[name] = {int(c) for c in cls.split(",")}
    base = PostCfg(**json.load(open(args.postcfg))["cfg"])
    cand_cfg = replace(base, min_score=0.0, min_area=100)             # permissive: the learned score decides
    photos = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))

    views = [v.split(",") for v in args.view]
    stems_of = []
    for dirs in views:
        s = set.intersection(*[{os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, "*.npz"))} for d in dirs])
        stems_of.append([x for x in sorted(s & set(photos)) if any(photos[x].annotators.values())][:args.limit or None])
    gts, cats = {}, {}
    for s in sorted(set().union(*stems_of)):
        rl = photo_gt_rles(photos[s])
        aids = [aid for aid in rl if rl[aid]]                          # annotator copies with at least one filament
        gts[s] = [rl[aid] for aid in aids]
        cats[s] = [np.asarray(photos[s].categories[aid]) for aid in aids]
    _G.update(views=views, cfg=cand_cfg, gts=gts)
    jobs = [(v, s) for v, ss in enumerate(stems_of) for s in ss]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(dict(_G),)) as pool:
        rows = list(pool.map(_candidates, jobs, chunksize=2))
    rows = [(v, s, X, ious, cats[s]) for v, s, X, ious in rows]
    print(f"{len(gts)} photos, views {[len(s) for s in stems_of]}, {sum(len(r[2]) for r in rows)} candidates", flush=True)
    for name, allowed in gt_sets.items():
        fit_gt_set(name, allowed, rows, len(views), base, cand_cfg, args.folds, args.out.format(name=name), args.thr)


if __name__ == "__main__":
    main()
