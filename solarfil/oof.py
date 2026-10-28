"""Out-of-fold probability maps from trained checkpoints (no retraining).

Every checkpoint was trained on four of the five folds (grouped by observation day, `data.make_folds`) and selected on the
fifth.  This predicts the held-out photographs of each checkpoint with the same 8-fold test-time augmentation as `train.py`
uses for its own out-of-fold maps, so the maps on which the learned filter (`rescore.py`) is fitted can be regenerated from the
released weights alone:

    python -m solarfil.oof --data_root D --ckpts weights/Unet_efficientnet-b4_s1536_f?.pth weights/Unet_efficientnet-b4_s2048_f?.pth --out work

writes `<out>/oof_<arch>_<encoder>_s<size>/<photo>.npz` (uint8 probabilities (2, S, S): foreground, border - the format of
`train.py`).  `--skip_existing` / `--skip_list` leave photographs alone whose maps already exist.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .annotations import load_photos
from .data import make_folds
from .infer import load_models
from .model import predict_full
from .preprocess import load_input


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--data_root", required=True)
    a.add_argument("--ckpts", nargs="+", required=True)
    a.add_argument("--out", default="work")
    a.add_argument("--tta", type=int, default=8)
    a.add_argument("--skip_existing", action="store_true")
    a.add_argument("--skip_list", default="", help="text file with photograph stems to skip, one per line")
    a.add_argument("--limit", type=int, default=0, help="only the first N held-out photographs of every checkpoint")
    args = a.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = os.path.join(args.data_root, "train", "train_images")
    photos_all = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    photos = {k: v for k, v in photos_all.items() if os.path.exists(os.path.join(img_dir, v.file_name))}
    skip = set(open(args.skip_list).read().split()) if args.skip_list else set()
    pool = ThreadPoolExecutor(2)                         # pre-processing of the next photographs overlaps with the GPU
    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ca = ck["args"]
        model = next(iter(load_models([path], dev).values()))
        folds = make_folds(list(photos_all), ca.get("n_folds", 5), ca.get("seed", 42))        # the folds of train.py
        stems = [s for s in folds.loc[folds.fold == ca["fold"], "stem"] if s in photos][:args.limit or None]
        out_dir = os.path.join(args.out, f"oof_{ca['arch']}_{ca['encoder']}_s{ca['size']}")
        os.makedirs(out_dir, exist_ok=True)
        todo = [s for s in stems if s not in skip and not (args.skip_existing and os.path.exists(os.path.join(out_dir, s + ".npz")))]
        print(f"{os.path.basename(path)}: fold {ca['fold']}, {len(todo)}/{len(stems)} photographs -> {out_dir}", flush=True)
        t0 = time.time()
        it, queue = iter(todo), deque()

        def push():
            s = next(it, None)
            if s is not None:
                queue.append((s, pool.submit(load_input, os.path.join(img_dir, photos[s].file_name), ca["size"])))
        for _ in range(4):
            push()
        i = 0
        while queue:
            s, f = queue.popleft()
            push()
            x = torch.from_numpy(f.result().transpose(2, 0, 1).astype(np.float32) / 255.0)
            p = predict_full(model, x, tta=args.tta).numpy()
            np.savez_compressed(os.path.join(out_dir, f"{s}.npz"), p=np.round(p * 255).astype(np.uint8))
            if i % 20 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
            i += 1
    print("done", flush=True)


if __name__ == "__main__":
    main()
