"""Train one fold.

python -m solarfil.train --data_root <MAGFiLO_1.0_Kaggle_2026> --work /kaggle/working --fold 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .annotations import load_photos, photo_gt_rles
from .data import CropDataset, build_cache, load_full, make_folds
from .metric import PQStats, photo_stats
from .model import EMA, FilamentLoss, FilamentNet, predict_full
from .postprocess import PostCfg, instances_rle


def get_args(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--data_root", required=True)
    a.add_argument("--work", default="work")
    a.add_argument("--fold", type=int, default=0)
    a.add_argument("--n_folds", type=int, default=5)
    a.add_argument("--size", type=int, default=1536, help="full-disk working resolution")
    a.add_argument("--crop", type=int, default=768)
    a.add_argument("--arch", default="Unet")
    a.add_argument("--encoder", default="efficientnet-b4")
    a.add_argument("--weights", default="imagenet")
    a.add_argument("--epochs", type=int, default=40)
    a.add_argument("--bs", type=int, default=6)
    a.add_argument("--lr", type=float, default=6e-4)
    a.add_argument("--wd", type=float, default=1e-4)
    a.add_argument("--samples_per_epoch", type=int, default=0, help="0 -> 3 crops per training photo")
    a.add_argument("--border_weight", type=float, default=0.5)
    a.add_argument("--val_every", type=int, default=4)
    a.add_argument("--val_tta", type=int, default=2)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--limit", type=int, default=0, help="debug: use only N photos")
    a.add_argument("--ema", type=float, default=0.998)
    a.add_argument("--cache_root", default="", help="where to put the image cache (default: --work)")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="MAGFiLO classes used as foreground (1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous)")
    a.add_argument("--init_ckpt", default="", help="start from the EMA weights of this checkpoint (fine-tuning)")
    return a.parse_args(argv)


def evaluate(model, stems, cache_dir, gts, cfg: PostCfg, tta: int) -> PQStats:
    total = PQStats()
    for s in stems:
        p = predict_full(model, load_full(s, cache_dir), tta=tta).numpy()
        total.add(photo_stats(gts[s], instances_rle(p[0], p[1], cfg)))
    return total


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_path = os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    img_dir = os.path.join(args.data_root, "train", "train_images")
    photos = load_photos(json_path)
    photos = {k: v for k, v in photos.items() if os.path.exists(os.path.join(img_dir, v.file_name))}
    if args.limit:
        photos = dict(list(photos.items())[: args.limit])
    classes = None if set(args.classes) == {1, 2, 3, 4} else set(args.classes)
    suffix = "" if classes is None else "_c" + "".join(map(str, sorted(classes)))
    cache_dir = os.path.join(args.cache_root or args.work, f"cache_{args.size}{suffix}")
    print(f"photos={len(photos)} device={dev} cache={cache_dir} classes={sorted(classes or {1, 2, 3, 4})}", flush=True)
    build_cache(photos, img_dir, cache_dir, args.size, workers=args.workers, classes=classes)

    folds = make_folds(list(photos), args.n_folds, args.seed)
    folds.to_csv(os.path.join(args.work, "folds.csv"), index=False)
    tr = folds.loc[folds.fold != args.fold, "stem"].tolist()
    va = folds.loc[folds.fold == args.fold, "stem"].tolist()
    gts = {s: photo_gt_rles(photos[s], classes) for s in va}
    print(f"fold {args.fold}: train={len(tr)} val={len(va)}", flush=True)

    spe = args.samples_per_epoch or 3 * len(tr)
    ds = CropDataset(tr, cache_dir, crop=args.crop, samples_per_epoch=spe, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=dev.type == "cuda", persistent_workers=args.workers > 0)

    model = FilamentNet(args.arch, args.encoder, None if args.weights == "none" else args.weights).to(dev)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=dev, weights_only=False)["state_dict"])
        print("initialised from", args.init_ckpt, flush=True)
    ema = EMA(model, decay=args.ema)
    crit = FilamentLoss(border_weight=args.border_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl)
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))) * 0.98 + 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    tag = f"{args.arch}_{args.encoder}_s{args.size}{suffix}_f{args.fold}"
    ckpt = os.path.join(args.work, f"{tag}.pth")
    best, log = -1.0, []
    cfg = PostCfg()
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(x)
            loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            tot += loss.item()
        msg = f"ep {ep:02d} loss {tot / len(dl):.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s"
        if ep % args.val_every == 0 or ep == args.epochs:
            st = evaluate(ema.ema, va, cache_dir, gts, cfg, args.val_tta)
            msg += f" | val {st}"
            log.append({"epoch": ep, "pq": st.pq, "sq": st.sq, "dq": st.dq})
            if st.pq > best:
                best = st.pq
                torch.save({"state_dict": ema.ema.state_dict(), "args": vars(args), "epoch": ep, "pq": best}, ckpt)
                msg += " *"
        print(msg, flush=True)

    # out-of-fold probabilities of the best checkpoint (for post-processing tuning)
    sd = torch.load(ckpt, map_location=dev)["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    oof_dir = os.path.join(args.work, f"oof_{args.arch}_{args.encoder}_s{args.size}{suffix}")
    os.makedirs(oof_dir, exist_ok=True)
    for s in va:
        p = predict_full(model, load_full(s, cache_dir), tta=8).numpy()
        np.savez_compressed(os.path.join(oof_dir, f"{s}.npz"), p=np.round(p * 255).astype(np.uint8))
    with open(os.path.join(args.work, f"{tag}_log.json"), "w") as f:
        json.dump({"best_pq": best, "log": log, "args": vars(args)}, f, indent=1)
    print(f"done fold {args.fold}: best val PQ {best:.4f} -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
