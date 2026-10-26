"""Predict the test set and write submission.csv.

python -m solarfil.infer --data_root <root> --ckpts work/*.pth --postcfg postcfg.json --out submission.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

from .data import build_test_cache, load_full
from .model import FilamentNet, predict_full
from .postprocess import PostCfg, instances_rle


def load_models(paths, dev):
    groups = {}
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = FilamentNet(a["arch"], a["encoder"], weights=None)
        m.load_state_dict(ck["state_dict"])
        groups.setdefault(a["size"], []).append(m.to(dev).eval())
        print(f"loaded {os.path.basename(p)} (val PQ {ck.get('pq', float('nan')):.4f}, size {a['size']})")
    return groups


def check_submission(df: pd.DataFrame):
    assert list(df.columns) == ["filament_id", "segmentation_rle"]
    assert df["filament_id"].is_unique
    assert not df.isnull().any().any()
    for stem, g in df.groupby(df["filament_id"].str.split("_", n=1).str[0]):
        rles = [{"size": [2048, 2048], "counts": r.encode()} for r in g["segmentation_rle"]]
        if len(rles) > 1:
            inter = mask_util.area(mask_util.merge(rles, intersect=False))
            assert inter == mask_util.area(rles).sum(), f"overlapping masks in {stem}"


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--data_root", required=True)
    a.add_argument("--ckpts", nargs="+", required=True)
    a.add_argument("--postcfg", default="")
    a.add_argument("--work", default="work")
    a.add_argument("--tta", type=int, default=8)
    a.add_argument("--out", default="submission.csv")
    a.add_argument("--save_probs", default="")
    a.add_argument("--rescorer", default="", help="pickle from rescore.py (overrides min_score/min_area)")
    args = a.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PostCfg(**json.load(open(args.postcfg))["cfg"]) if args.postcfg else PostCfg()
    resc = None
    if args.rescorer:
        import pickle
        resc = pickle.load(open(args.rescorer, "rb"))
        cfg = PostCfg(**resc["cand_cfg"])
        print(f"rescorer thr={resc['thr']:.3f} (CV PQ {resc['cv_pq']:.4f})")
    print("post-processing:", cfg)
    paths = sorted(glob.glob(os.path.join(args.data_root, "test", "test_images", "*.jp*g")))
    groups = load_models(args.ckpts, dev)
    for size in groups:
        build_test_cache(paths, os.path.join(args.work, f"test_cache_{size}"), size)

    rows, t0 = [], time.time()
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        probs, top = [], max(groups)
        for size, models in groups.items():
            p = predict_full(models, load_full(stem, os.path.join(args.work, f"test_cache_{size}")), tta=args.tta)
            if size != top:  # models trained at different resolutions: average at the largest one
                p = torch.nn.functional.interpolate(p[None], size=(top, top), mode="bilinear", align_corners=False)[0]
            probs.append((p.numpy(), len(models)))
        p = sum(q * n for q, n in probs) / sum(n for _, n in probs)
        if args.save_probs:
            os.makedirs(args.save_probs, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_probs, f"{stem}.npz"), p=np.round(p * 255).astype(np.uint8))
        for i, r in enumerate(instances_rle(p[0], p[1], cfg, rescorer=resc), 1):
            rows.append({"filament_id": f"{stem}_{i}", "segmentation_rle": r})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    check_submission(df)
    df.to_csv(args.out, index=False)
    n_img = df["filament_id"].str.split("_", n=1).str[0].nunique()
    print(f"wrote {args.out}: {len(df)} filaments in {n_img}/{len(paths)} images ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
