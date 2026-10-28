"""Coordinate-descent tuner of the hand-tuned post-processing (`PostCfg`, `configs/postcfg.json`) on out-of-fold maps.

Legacy of the first version of the pipeline: it produced the baseline rule that Table 1 of the report compares the learned
filter with (the table itself uses the two-parameter grid of `ablate.py`).  The submitted pipeline does not use its output.

python -m solarfil.tune --data_root <root> --oof work/oof_Unet_efficientnet-b4_s1536 [--oof other_dir ...]
Multiple --oof dirs are averaged (model ensemble).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from .annotations import load_photos, photo_gt_rles
from .metric import PQStats, photo_stats
from .postprocess import PostCfg, instances_rle, load_avg

_G = {}


def _load(stem):
    return load_avg([os.path.join(d, f"{stem}.npz") for d in _G["dirs"]])


def _score_one(args):
    stem, cfg = args
    p = _load(stem)
    return photo_stats(_G["gts"][stem], instances_rle(p[0], p[1], cfg))


_CACHE = {}


def score(stems, cfg, pool) -> PQStats:
    key = json.dumps(cfg.as_dict(), sort_keys=True)
    if key in _CACHE:
        return PQStats(**_CACHE[key])
    tot = PQStats()
    for s in pool.map(_score_one, [(st, cfg) for st in stems], chunksize=2):
        tot.add(s)
    _CACHE[key] = dict(iou_sum=tot.iou_sum, tp=tot.tp, fp=tot.fp, fn=tot.fn)
    if _G.get("cache_file"):
        with open(_G["cache_file"], "a") as f:
            f.write(json.dumps({"k": key, "v": _CACHE[key]}) + "\n")
    return tot


GRID = {
    "thr": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "link_px": [0, 4, 8, 14, 22, 32],
    "border_thr": [1.01, 0.7, 0.5, 0.35],
    "min_area": [50, 100, 150, 200, 300, 450, 650],
    "min_score": [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "close_px": [0, 2, 4],
    "seed_thr": [1.01, 0.9, 0.8, 0.7, 0.6],
    "min_cc": [10, 40, 100],
    "fill_holes": [True, False],
}


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--data_root", required=True)
    a.add_argument("--oof", nargs="+", required=True)
    a.add_argument("--max_photos", type=int, default=0)
    a.add_argument("--rounds", type=int, default=2)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", default="postcfg.json")
    a.add_argument("--init", default="", help="start from this postcfg.json")
    a.add_argument("--cache", default="", help="jsonl of already-scored configs (makes the run resumable)")
    args = a.parse_args(argv)

    photos = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    stems = sorted(set.intersection(*[{os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, "*.npz"))}
                                      for d in args.oof]))
    if args.max_photos:
        rng = np.random.default_rng(0)
        stems = sorted(rng.choice(stems, size=min(args.max_photos, len(stems)), replace=False).tolist())
    _G["dirs"] = args.oof
    _G["gts"] = {s: photo_gt_rles(photos[s]) for s in stems}
    print(f"tuning on {len(stems)} OOF photos", flush=True)

    cfg = PostCfg(**json.load(open(args.init))["cfg"]) if args.init else PostCfg()
    if args.cache:
        _G["cache_file"] = args.cache
        if os.path.exists(args.cache):
            for line in open(args.cache):
                r = json.loads(line)
                _CACHE[r["k"]] = r["v"]
            print(f"resumed {len(_CACHE)} cached configs", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_G.update, initargs=(dict(_G),)) as pool:
        best = score(stems, cfg, pool)
        print("start", cfg, best, flush=True)
        for r in range(args.rounds):
            for k, vals in GRID.items():
                for v in vals:
                    if getattr(cfg, k) == v:
                        continue
                    c = replace(cfg, **{k: v})
                    st = score(stems, c, pool)
                    flag = ""
                    if st.pq > best.pq + 1e-5:
                        cfg, best, flag = c, st, "  <-- best"
                    print(f"r{r} {k}={v}: {st}{flag}", flush=True)
            print(f"after round {r}: {cfg} {best}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"cfg": cfg.as_dict(), "pq": best.pq, "sq": best.sq, "dq": best.dq,
                   "n_photos": len(stems), "oof": args.oof}, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
