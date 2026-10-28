"""Candidate instances of every photo for stage 2 (`refine.py`).

For every photo the probability maps are thresholded at the *primary* level (the level of the deployed pipeline) and at a
few extra levels.  Primary candidates are scored by the learned re-scorer (`rescore.py`); out-of-fold photos get
cross-fitted scores (the scorer of fold k is trained on the other folds of stage 1), test photos the scores of the
final scorer.  Extra-level candidates only serve as additional, differently shaped training seeds for the refiner.

    python -m solarfil.cands --data_root D --postcfg configs/postcfg.json --view OOF1536 --view OOF1536,OOF2048 --out cands_oof.pkl
    python -m solarfil.cands --probs PROBS_TEST --rescorer configs/rescorer_all.npz --out cands_test.pkl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from pycocotools import mask as mask_util

from .postprocess import PostCfg, candidate_features, encode, instances, load_avg, upsample

_G: dict = {}


def _init(g):
    _G.update(g)


def photo_candidates(fg: np.ndarray, bd: np.ndarray, cfg: PostCfg, levels, primary: float, max_extra: int, key: str):
    """-> dict(rle, level, X) : RLE strings of all candidates, the level each came from and the re-scorer features of
    the primary ones (rows in the order of the primary entries)."""
    fg, bd = upsample(fg), upsample(bd)
    masks, feats = instances(fg, bd, replace(cfg, thr=primary), want_feats=True)
    X = candidate_features(fg, masks, feats)
    rle, level = encode(masks), [primary] * len(masks)
    prim = [{"size": [2048, 2048], "counts": r.encode()} for r in rle]
    extra = []
    for lv in levels:
        if lv == primary:
            continue
        ms = instances(fg, bd, replace(cfg, thr=lv, min_area=100, min_score=0.0))
        er = encode(ms)
        if prim and er:
            iou = np.asarray(mask_util.iou([{"size": [2048, 2048], "counts": r.encode()} for r in er], prim, [0] * len(prim)))
            er = [r for r, m in zip(er, iou.max(1)) if m < 0.9]                   # keep seeds that differ from a primary one
        extra += [(r, lv) for r in er]
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    if len(extra) > max_extra:
        extra = [extra[i] for i in sorted(rng.choice(len(extra), max_extra, replace=False))]
    rle += [r for r, _ in extra]
    level += [lv for _, lv in extra]
    return {"rle": rle, "level": np.array(level, np.float32), "X": X}


def _job(args):
    stem, paths = args
    p = load_avg(paths)
    return stem, photo_candidates(p[0], p[1], _G["cfg"], _G["levels"], _G["primary"], _G["max_extra"], stem)


def gather(stems_paths, cfg, levels, primary, max_extra, workers):
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(dict(cfg=cfg, levels=levels, primary=primary, max_extra=max_extra),)) as pool:
        return dict(pool.map(_job, stems_paths, chunksize=2))


def to_frame(cands: dict, thr: float | None = None):
    """Selected primary candidates (score >= thr, default: the scorer's own threshold) as a submission-format DataFrame."""
    import pandas as pd
    thr = cands.get("thr", 0.0) if thr is None else thr
    rows = []
    for stem, c in sorted(cands["photos"].items()):
        k = 0
        for r, lv, s in zip(c["rle"], c["level"], c["score"]):
            if lv == cands["primary"] and s >= thr:
                k += 1
                rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": r})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--emit", default="", help="read this candidate pickle and write the selected instances to --out (csv)")
    a.add_argument("--thr", type=float, default=None, help="--emit: score threshold (default: the one stored in the pickle)")
    a.add_argument("--data_root", default="")
    a.add_argument("--postcfg", default="", help="candidate extraction cfg (default: the one stored in --rescorer)")
    a.add_argument("--view", action="append", default=[], help="OOF mode: comma-separated dirs (averaged); the first view "
                   "that has a photo supplies its maps")
    a.add_argument("--probs", default="", help="test mode: directory of <stem>.npz maps")
    a.add_argument("--rescorer", default="", help="test mode: scorer file; OOF mode: only the features / cfg are taken")
    a.add_argument("--levels", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    a.add_argument("--primary", type=float, default=0.5)
    a.add_argument("--max_extra", type=int, default=6)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--out", required=True)
    args = a.parse_args(argv)
    from .rescore import expected_gain, fit_model, load_rescorer, select_classes

    if args.emit:
        df = to_frame(pickle.load(open(args.emit, "rb")), args.thr)
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}: {len(df)} instances in {df['filament_id'].str.rsplit('_', n=1).str[0].nunique()} photos")
        return
    if args.rescorer:
        resc = load_rescorer(args.rescorer)
        cfg = PostCfg(**resc["cand_cfg"])
    else:
        cfg = replace(PostCfg(**json.load(open(args.postcfg))["cfg"]), min_score=0.0, min_area=100)
    cfg = replace(cfg, thr=args.primary)

    if args.probs:                                                           # ---- test photos
        sp = [(os.path.basename(f)[:-4], [f]) for f in sorted(glob.glob(os.path.join(args.probs, "*.npz")))][:args.limit or None]
        names = {s: [""] for s, _ in sp}
        cands = gather(sp, cfg, args.levels, args.primary, args.max_extra, args.workers)
        photos = {}
        for s, c in cands.items():
            score = np.full(len(c["rle"]), np.nan, np.float32)
            n_p = int((c["level"] == args.primary).sum())
            if n_p:
                score[:n_p] = resc["model"].predict(c["X"])
            photos[s] = {"rle": c["rle"], "level": c["level"], "score": score, "maps": names[s], "X": c["X"]}
        pickle.dump({"photos": photos, "primary": args.primary, "thr": resc["thr"]}, open(args.out, "wb"))
        print(f"wrote {args.out}: {sum(len(p['rle']) for p in photos.values())} candidates in {len(photos)} photos")
        return

    from .annotations import load_photos, photo_gt_rles                      # ---- out-of-fold photos
    from .data import make_folds
    gt_photos = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    folds = make_folds(list(gt_photos), 5, 42).set_index("stem")["fold"].to_dict()
    views = [v.split(",") for v in args.view]
    sp, names = [], {}
    for stem in sorted(gt_photos):
        if not any(gt_photos[stem].annotators.values()):
            continue
        for dirs in reversed(views):                                         # prefer the richest view that has the photo
            paths = [os.path.join(d, stem + ".npz") for d in dirs]
            if all(os.path.exists(p) for p in paths):
                sp.append((stem, paths))
                names[stem] = [os.path.basename(d) for d in dirs]
                break
    sp = sp[:args.limit or None]
    cands = gather(sp, cfg, args.levels, args.primary, args.max_extra, args.workers)
    # cross-fitted scores of the primary candidates, targets = expected PQ gain against all annotator copies
    ys, rows = {}, {}
    for stem, c in cands.items():
        n_p = int((c["level"] == args.primary).sum())
        gt = {k: v for k, v in photo_gt_rles(gt_photos[stem]).items() if v}
        cats = {k: np.asarray(gt_photos[stem].categories[k]) for k in gt}
        if n_p:
            pr = [{"size": [2048, 2048], "counts": r.encode()} for r in c["rle"][:n_p]]
            ious = [np.asarray(mask_util.iou(pr, g, [0] * len(g))) for g in gt.values()]
            ious = select_classes(ious, list(cats.values()), {1, 2, 3, 4})
            ys[stem] = expected_gain(ious, n_p) if ious else np.zeros(n_p)
        rows[stem] = n_p
    score = {s: np.full(len(c["rle"]), np.nan, np.float32) for s, c in cands.items()}
    for k in range(5):
        tr = [s for s in cands if folds[s] != k and rows[s]]
        mdl = fit_model(np.concatenate([cands[s]["X"] for s in tr]), np.concatenate([ys[s] for s in tr]))
        for s in cands:
            if folds[s] == k and rows[s]:
                score[s][:rows[s]] = mdl.predict(cands[s]["X"])
    photos = {s: {"rle": c["rle"], "level": c["level"], "score": score[s], "maps": names[s], "X": c["X"]} for s, c in cands.items()}
    pickle.dump({"photos": photos, "primary": args.primary, "folds": {s: folds[s] for s in photos}}, open(args.out, "wb"))
    print(f"wrote {args.out}: {sum(len(p['rle']) for p in photos.values())} candidates in {len(photos)} photos")


if __name__ == "__main__":
    main()
