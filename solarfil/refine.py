"""Stage 2 - instance-conditioned refinement ("a second look" at every selected filament).

Stage 1 sees a whole disk at 1536 / 2048 px and decides *which* regions are filaments.  Under PQ, however, a thin
object only counts when its mask overlaps an annotator's mask with IoU > 0.5, so one or two pixels of boundary error
or a missed tip decide between a hit and a miss.  Stage 2 therefore re-segments every selected instance (the "seed")
in its own crop:

* the crop is rotated so that the long axis of the seed's minimum-area rectangle is horizontal, centred on the seed and
  padded by half its length at both ends (the network must see where the filament *ends*); it is never scaled up and
  is scaled down only if the padded seed does not fit into 128 x 384 px (`crop_affine`);
* channels = the three preprocessed image channels, the stage-1 foreground and border probabilities, the seed mask and
  the masks of all other candidates (`make_crops`);
* a small U-Net predicts the mask of *that* instance; the training target is the mean of the ground-truth filaments
  (one per annotator copy) that overlap the seed with IoU >= 0.3, i.e. the consensus drawn by the annotators who drew
  this filament, and empty if nobody did (`make_targets`);
* the prediction is warped back to the full disk, thresholded and pasted into a disjoint instance map (`paste_back`).

Cross-fitting: the crop network of fold k never sees a photo of fold k (the folds of stage 1, grouped by observation
day), so refined out-of-fold masks can be scored with the official metric without leakage.

    python -m solarfil.refine prep  --data_root D --cands cands_oof.pkl --maps_root OOF_ROOT --out crops_oof.npz
    python -m solarfil.refine prep  --data_root D --cands cands_test.pkl --maps_root PROBS_TEST --split test --out crops_test.npz
    python -m solarfil.refine train --crops crops_oof.npz --test_crops crops_test.npz --out_dir work/refine   # GPU
    python -m solarfil.refine apply --cands cands_oof.pkl --crops_meta crops_oof_meta.npz --preds work/refine/pred_oof.npz --data_root D   # CPU

`cands_*.pkl` is written by `python -m solarfil.cands` (candidate instances of every photo, their learned scores and
the fold of their photo).
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from pycocotools import mask as mask_util

NATIVE = 2048
CROP_H, CROP_W = 128, 384
N_CH = 7                      # image(3) + p_fg + p_border + seed + others
MATCH_IOU = 0.3               # a ground-truth filament belongs to the seed's consensus target above this IoU


# ----------------------------------------------------------------------------------------------------- geometry
def crop_affine(seed: np.ndarray, pad: float = 0.5, h: int = CROP_H, w: int = CROP_W, margin: float = 24.0) -> np.ndarray:
    """2x3 float64 matrix mapping native (x, y) pixel coordinates to crop coordinates (long axis -> horizontal)."""
    pts = cv2.findNonZero(seed.astype(np.uint8))
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:                                         # make rw the long side; `ang` is then the angle of the long axis
        rw, rh, ang = rh, rw, ang + 90.0
    length, width = max(rw, 1.0), max(rh, 1.0)
    s = min(1.0, w / ((1.0 + 2.0 * pad) * length), h / (width + 2.0 * margin))
    m = cv2.getRotationMatrix2D((cx, cy), ang, s)
    m[0, 2] += w / 2.0 - cx
    m[1, 2] += h / 2.0 - cy
    return m


def warp(img: np.ndarray, m: np.ndarray, h: int = CROP_H, w: int = CROP_W, nearest: bool = False) -> np.ndarray:
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def unwarp_window(crop: np.ndarray, m: np.ndarray, h: int = CROP_H, w: int = CROP_W):
    """Warp a crop-space map back to native coordinates; returns (y0, x0, patch) covering the footprint of the crop."""
    inv = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64) @ inv.T
    x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int), NATIVE)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, np.zeros((0, 0), np.float32)
    inv = inv.copy()
    inv[0, 2] -= x0
    inv[1, 2] -= y0
    patch = cv2.warpAffine(np.ascontiguousarray(crop, dtype=np.float32), inv, (int(x1 - x0), int(y1 - y0)),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return int(y0), int(x0), patch


# ----------------------------------------------------------------------------------------------------- crops
def photo_stack(img3: np.ndarray, fg: np.ndarray, bd: np.ndarray) -> np.ndarray:
    """(2048, 2048, 5) uint8: preprocessed image channels + stage-1 foreground / border probabilities."""
    q = lambda a: np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    return np.dstack([img3, q(fg), q(bd)])


def make_crops(stack: np.ndarray, seeds: list[np.ndarray], groups: list[int]):
    """Crops (N, 7, H, W) uint8 and affines (N, 2, 3) for `seeds`; `groups[i]` says which seeds are "the others" of i
    (seeds sharing a group id come from the same candidate pool, e.g. the same threshold level)."""
    n = len(seeds)
    X = np.zeros((n, N_CH, CROP_H, CROP_W), np.uint8)
    M = np.zeros((n, 2, 3), np.float64)
    union = {g: np.zeros((NATIVE, NATIVE), np.uint8) for g in set(groups)}
    for s, g in zip(seeds, groups):
        union[g] |= s
    for i, (s, g) in enumerate(zip(seeds, groups)):
        m = crop_affine(s)
        M[i] = m
        for c in range(5):
            X[i, c] = warp(stack[:, :, c], m)
        X[i, 5] = warp(s * 255, m)
        X[i, 6] = warp((union[g] & (1 - s)) * 255, m)
    return X, M


def make_targets(seeds: list[np.ndarray], seed_rles: list[dict], gt_by_record: dict, affines: np.ndarray):
    """Consensus targets of every seed.  Returns (T, T_frag, n_match, n_frag):

    T      (N, H, W) uint8 : mean of the ground-truth filaments (one per annotator copy) with IoU >= MATCH_IOU,
    T_frag (N, H, W) uint8 : the same, but a seed that is only a *fragment* of a ground-truth filament (at least 60 % of
                             the seed lies inside it, and the seed covers at least 12 % of it) is also matched, so that
                             the network learns to complete broken filaments,
    n_match / n_frag       : number of annotator copies behind the two targets."""
    n = len(seeds)
    T = np.zeros((n, CROP_H, CROP_W), np.uint8)
    T2 = np.zeros((n, CROP_H, CROP_W), np.uint8)
    nm, nf = np.zeros(n, np.int16), np.zeros(n, np.int16)
    acc, acc2 = [None] * n, [None] * n
    a_s = np.asarray(mask_util.area(seed_rles), np.float64) if n else np.zeros(0)
    for rles in gt_by_record.values():
        if not rles or not n:
            continue
        iou = np.asarray(mask_util.iou(seed_rles, rles, [0] * len(rles)))           # (n_seed, n_gt)
        cont = np.asarray(mask_util.iou(seed_rles, rles, [1] * len(rles)))          # |seed & gt| / |seed|
        a_g = np.asarray(mask_util.area(rles), np.float64)
        best = iou.argmax(1)
        for i in range(n):
            j = best[i]
            hit = iou[i, j] >= MATCH_IOU
            if not hit:                                                              # fragment of a bigger filament?
                inter = cont[i] * a_s[i]
                ok = (cont[i] >= 0.6) & (inter >= 0.12 * a_g)
                j = int(np.argmax(np.where(ok, inter, -1))) if ok.any() else -1
            if j < 0:
                continue
            g = mask_util.decode(rles[j]).astype(np.float32)
            acc2[i] = g if acc2[i] is None else acc2[i] + g
            nf[i] += 1
            if hit:
                acc[i] = g if acc[i] is None else acc[i] + g
                nm[i] += 1
    for i in range(n):
        if acc[i] is not None:
            T[i] = np.clip(np.rint(warp(acc[i] / nm[i], affines[i]) * 255.0), 0, 255).astype(np.uint8)
        if acc2[i] is not None:
            T2[i] = np.clip(np.rint(warp(acc2[i] / nf[i], affines[i]) * 255.0), 0, 255).astype(np.uint8)
    return T, T2, nm, nf


def _prep_photo(job):
    """One photo -> crops of all its candidates (worker process)."""
    stem, cands, img_path, map_paths, gt = job
    from .postprocess import load_avg, upsample
    from .preprocess import make_input, read_gray
    stack = photo_stack(make_input(read_gray(img_path)), *[upsample(c) for c in load_avg(map_paths)])
    rles = [{"size": [NATIVE, NATIVE], "counts": c.encode()} for c in cands["rle"]]
    seeds = [mask_util.decode(r) for r in rles]
    X, M = make_crops(stack, seeds, [int(round(float(l) * 100)) for l in cands["level"]])
    T = T2 = nm = nf = None
    if gt is not None:
        T, T2, nm, nf = make_targets(seeds, rles, gt, M)
    return stem, X, M, (T, T2), (nm, nf)


def prep(args):
    from .annotations import load_photos, photo_gt_rles
    from .data import make_folds
    cands = pickle.load(open(args.cands, "rb"))
    photos_c = cands["photos"]
    train = args.split == "train"
    if train:
        photos = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
        folds = make_folds(list(photos), 5, 42).set_index("stem")["fold"].to_dict()
    jobs = []
    for stem, c in sorted(photos_c.items()):
        if not len(c["rle"]):
            continue
        img = (os.path.join(args.data_root, "train", "train_images", photos[stem].file_name) if train else
               next(p for p in (os.path.join(args.data_root, "test", "test_images", stem + e) for e in (".jpg", ".jpeg", ".png"))
                    if os.path.exists(p)))
        mp = [os.path.join(args.maps_root, m, stem + ".npz") for m in c["maps"]]
        gt = photo_gt_rles(photos[stem]) if train else None
        gt = {k: v for k, v in gt.items() if v} if gt else gt
        jobs.append((stem, c, img, mp, gt))
    if args.limit:
        jobs = jobs[:args.limit]
    Xs, Ms, Ts, T2s, nms, nfs, meta = [], [], [], [], [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (stem, X, M, T, nm) in enumerate(pool.map(_prep_photo, jobs, chunksize=1)):
            c = photos_c[stem]
            Xs.append(X); Ms.append(M)
            if train:
                Ts.append(T[0]); T2s.append(T[1]); nms.append(nm[0]); nfs.append(nm[1])
            meta.append(np.column_stack([np.full(len(X), k), np.arange(len(X)), c["level"], c["score"],
                                         np.full(len(X), folds[stem] if train else -1)]))
            if k % 50 == 0:
                print(f"prep {k}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    out = dict(X=np.concatenate(Xs), M=np.concatenate(Ms), meta=np.concatenate(meta).astype(np.float32),
               stems=np.array([j[0] for j in jobs]))
    if train:
        out.update(T=np.concatenate(Ts), T_frag=np.concatenate(T2s), n_match=np.concatenate(nms), n_frag=np.concatenate(nfs))
    np.savez_compressed(args.out, **out)
    np.savez_compressed(args.out.replace(".npz", "_meta.npz"), **{k: v for k, v in out.items() if k not in ("X", "T", "T_frag")})
    print(f"wrote {args.out}: {len(out['X'])} crops from {len(jobs)} photos", flush=True)


# ----------------------------------------------------------------------------------------------------- network
def build_net(encoder: str = "resnet34", weights: str | None = "imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=N_CH, classes=1,
                    decoder_attention_type="scse")


def augment(x: np.ndarray, t: np.ndarray, rng: np.random.Generator):
    """x (7,H,W) uint8, t (H,W) uint8 -> float32 tensors-to-be.  Flips, photometric jitter of the image channels and
    morphological jitter of the seed (the stage-1 seed is never exactly the annotators' outline)."""
    x = x.astype(np.float32) / 255.0
    t = t.astype(np.float32) / 255.0
    if rng.random() < 0.5:
        x, t = x[:, :, ::-1], t[:, ::-1]
    if rng.random() < 0.5:
        x, t = x[:, ::-1], t[::-1]
    x = np.ascontiguousarray(x)
    t = np.ascontiguousarray(t)
    if rng.random() < 0.8:                                  # brightness / contrast / gamma on the three image channels
        c, b, g = rng.uniform(0.8, 1.25), rng.uniform(-0.06, 0.06), rng.uniform(0.8, 1.25)
        x[:3] = np.clip((x[:3] - 0.5) * c + 0.5 + b, 0, 1) ** g
    if rng.random() < 0.5:                                  # seed jitter: +-1..2 px dilation / erosion, sometimes none
        k = int(rng.integers(1, 3))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        s = (x[5] > 0.5).astype(np.uint8)
        s = cv2.dilate(s, ker) if rng.random() < 0.5 else cv2.erode(s, ker)
        if s.any():
            x[5] = s.astype(np.float32)
    return x, t


class CropSet:
    """Training crops of the photos outside one fold; samples are augmented on the fly (num_workers = 0 is enough:
    the arrays are in memory and the augmentations are a few cv2 / numpy calls)."""

    def __init__(self, X, T, idx, seed=0, neg_weight=1.0):
        self.X, self.T, self.idx = X, T, np.asarray(idx)
        self.rng = np.random.default_rng(seed)
        has = T[self.idx].reshape(len(self.idx), -1).max(1) > 0
        self.pos, self.neg = self.idx[has], self.idx[~has]
        self.neg_weight = neg_weight

    def batch(self, bs):
        n_neg = int(round(bs * self.neg_weight * len(self.neg) / max(len(self.idx), 1)))
        sel = np.concatenate([self.rng.choice(self.pos, bs - n_neg),
                              self.rng.choice(self.neg, n_neg) if n_neg and len(self.neg) else np.zeros(0, int)])
        xs, ts = zip(*[augment(self.X[i], self.T[i], self.rng) for i in sel])
        return np.stack(xs), np.stack(ts)[:, None]


def prefetch(ds: CropSet, bs: int, n: int, depth: int = 6):
    """Yield n augmented batches, produced by a background thread (numpy / cv2 release the GIL)."""
    import queue
    import threading
    q: queue.Queue = queue.Queue(depth)

    def work():
        for _ in range(n):
            q.put(ds.batch(bs))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def dice_bce(logits, target, eps=1.0):
    import torch.nn.functional as F
    import torch
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return bce + (1 - (2 * inter + eps) / (den + eps)).mean()


def predict_crops(models, X: np.ndarray, bs: int = 64, flips: bool = True, dev=None) -> np.ndarray:
    """Averaged probabilities (N, H, W) as uint8 over the models and the (h, v, hv) flips of the crop."""
    import torch
    dev = dev or next(models[0].parameters()).device
    out = np.zeros((len(X), CROP_H, CROP_W), np.uint8)
    variants = [(), (3,), (2,), (2, 3)] if flips else [()]
    with torch.no_grad():
        for a in range(0, len(X), bs):
            xb = torch.from_numpy(X[a:a + bs].astype(np.float32) / 255.0).to(dev)
            acc, n = 0, 0
            for m in models:
                for fl in variants:
                    xt = torch.flip(xb, fl) if fl else xb
                    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                        p = torch.sigmoid(m(xt).float())
                    acc = acc + (torch.flip(p, fl) if fl else p)
                    n += 1
            out[a:a + bs] = np.clip(np.rint((acc / n)[:, 0].cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    return out


def train_fold(X, T, tr_idx, va_idx, args, dev, log=print):
    import torch
    torch.manual_seed(args.seed)
    net = build_net(args.encoder, None if args.weights == "none" else args.weights).to(dev)
    ds = CropSet(X, T, tr_idx, args.seed, args.neg_weight)
    steps_per_epoch = max(len(tr_idx) // args.bs, 1)
    steps = args.epochs * steps_per_epoch
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))) * 0.98 + 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ema = [p.detach().clone() for p in net.parameters()]
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        tot = 0.0
        for xb, tb in prefetch(ds, args.bs, steps_per_epoch):
            xb, tb = torch.from_numpy(xb).to(dev), torch.from_numpy(tb).to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                lo = net(xb)
            loss = dice_bce(lo.float(), tb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                for e, p in zip(ema, net.parameters()):
                    e.mul_(args.ema).add_(p.detach(), alpha=1 - args.ema)
            tot += loss.item()
        msg = f"ep {ep:02d} loss {tot / steps_per_epoch:.4f} {time.time() - t0:.0f}s"
        if len(va_idx) and (ep % args.val_every == 0 or ep == args.epochs):
            msg += f" | val soft-IoU {val_iou(net, ema, X, T, va_idx, dev):.4f}"
        log(msg)
    with torch.no_grad():                                    # load the EMA weights
        for e, p in zip(ema, net.parameters()):
            p.copy_(e)
    return net.eval()


def val_iou(net, ema, X, T, va_idx, dev, max_n=1500) -> float:
    """Mean IoU (threshold 0.5) between the prediction and the consensus target on validation crops that have one."""
    import torch
    backup = [p.detach().clone() for p in net.parameters()]
    with torch.no_grad():
        for e, p in zip(ema, net.parameters()):
            p.copy_(e)
    net.eval()
    idx = [i for i in va_idx if T[i].max() > 0][:max_n]
    pr = predict_crops([net], X[idx], flips=False, dev=dev) > 127
    tg = T[idx] > 127
    iou = ((pr & tg).sum((1, 2)) / np.maximum((pr | tg).sum((1, 2)), 1)).mean() if len(idx) else float("nan")
    with torch.no_grad():
        for b, p in zip(backup, net.parameters()):
            p.copy_(b)
    net.train()
    return float(iou)


def train(args):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.crops)
    X, T, meta = d["X"], d["T_frag" if args.frag else "T"], d["meta"]
    fold = meta[:, 4].astype(int)
    Xt = np.load(args.test_crops)["X"] if args.test_crops else None
    pred_oof = np.zeros((len(X), CROP_H, CROP_W), np.uint8)
    pred_test = np.zeros((len(Xt), CROP_H, CROP_W), np.float32) if Xt is not None else None
    folds = args.folds or list(range(5))
    for k in folds:
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        print(f"=== fold {k}: train {len(tr)} crops, val {len(va)}", flush=True)
        net = train_fold(X, T, tr, va, args, dev, log=lambda s: print(s, flush=True))
        torch.save({"state_dict": net.state_dict(), "args": vars(args), "fold": k}, os.path.join(args.out_dir, f"refine_f{k}.pth"))
        pred_oof[va] = predict_crops([net], X[va], dev=dev)
        if Xt is not None:
            pred_test += predict_crops([net], Xt, dev=dev).astype(np.float32) / len(folds)
        np.savez_compressed(os.path.join(args.out_dir, "pred_oof.npz"), p=pred_oof, fold=fold)
        if Xt is not None:
            np.savez_compressed(os.path.join(args.out_dir, "pred_test.npz"), p=np.rint(pred_test).astype(np.uint8))
    print("done", flush=True)


# ----------------------------------------------------------------------------------------------------- paste back
def _rle(m: np.ndarray) -> dict:
    return mask_util.encode(np.asfortranarray(m.astype(np.uint8)))


def paste_back(probs: np.ndarray, affines: np.ndarray, seeds: list[np.ndarray], t: float = 0.5,
               fallback: str = "seed", min_ratio: float = 0.25, merge: bool = True):
    """Refined instances of one photo.  Every crop probability is warped back to the disk and thresholded at `t`; only
    the connected parts that touch the seed are kept, and a seed whose refined area falls below `min_ratio` x its own
    area keeps the seed ("seed") or is dropped ("drop").  Seeds that were completed to (nearly) the same filament -
    IoU > 0.5, or one lies >= 80 % inside the other - are merged.  Contested pixels go to the instance with the higher
    probability, so the result is disjoint.  Returns (masks, members, stats): members[k] lists the seeds behind masks[k],
    stats[k] = [mean prob, 10th percentile prob, max prob, area / seed area, IoU with the seed(s)] of the instance."""
    n = len(seeds)
    patches = [unwarp_window(probs[i].astype(np.float32) / 255.0, affines[i]) for i in range(n)]
    raw = []
    for i in range(n):
        y0, x0, patch = patches[i]
        m = np.zeros((NATIVE, NATIVE), np.uint8)
        if patch.size:
            m[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch > t
        if m.any():
            _, lab = cv2.connectedComponents(m, connectivity=8)
            touch = np.unique(lab[(seeds[i] > 0) & (m > 0)])
            m = np.isin(lab, touch[touch > 0]).astype(np.uint8)
        if m.sum() < min_ratio * seeds[i].sum():
            m = seeds[i].copy() if fallback == "seed" else np.zeros_like(m)
        raw.append(m)
    group = list(range(n))
    if merge and n > 1:
        rl = [_rle(m) for m in raw]
        iou = np.asarray(mask_util.iou(rl, rl, [0] * n))
        cont = np.asarray(mask_util.iou(rl, rl, [1] * n))                           # cont[i, j] = |i & j| / |i|
        area = np.array([m.sum() for m in raw])
        for i in range(n):
            for j in range(i + 1, n):
                if area[i] == 0 or area[j] == 0:
                    continue
                small = i if area[i] <= area[j] else j
                big = j if small == i else i
                if iou[i, j] > 0.5 or cont[small, big] >= 0.8:
                    gi, gj = group[i], group[j]
                    group = [gi if g == gj else g for g in group]
    ids = sorted(set(group), key=group.index)
    best = np.zeros((NATIVE, NATIVE), np.float32)
    owner = np.full((NATIVE, NATIVE), -1, np.int32)
    for i in range(n):
        y0, x0, patch = patches[i]
        if not patch.size or not raw[i].any():
            continue
        sl = (slice(y0, y0 + patch.shape[0]), slice(x0, x0 + patch.shape[1]))
        win = (patch > t) & (raw[i][sl] > 0) & (patch > best[sl])
        best[sl][win] = patch[win]
        owner[sl][win] = ids.index(group[i])
    masks, members, stats = [], [], []
    for k, g in enumerate(ids):
        mem = [i for i in range(n) if group[i] == g]
        m = (owner == k).astype(np.uint8)
        if not m.any() and fallback == "seed":                                       # e.g. the whole mask was contested
            m = np.maximum.reduce([seeds[i] for i in mem])
        seed = np.maximum.reduce([seeds[i] for i in mem])
        pv = best[m > 0]
        inter = float((m & seed).sum())
        stats.append([float(pv.mean()) if pv.size else 0.0, float(np.percentile(pv, 10)) if pv.size else 0.0,
                      float(pv.max()) if pv.size else 0.0, float(m.sum()) / max(float(seed.sum()), 1.0),
                      inter / max(float(m.sum() + seed.sum() - inter), 1.0)])
        masks.append(m)
        members.append(mem)
    return masks, members, np.asarray(stats, np.float32).reshape(-1, 5)


# ----------------------------------------------------------------------------------------------------- apply
def refined_photo(c, idx, M, P, thr, t, fallback="seed", primary=0.5, merge=True):
    """Refined masks of the selected (primary, score >= thr) candidates of one photo -> (seeds, masks)."""
    sel = [j for j in range(len(c["rle"])) if abs(float(c["level"][j]) - primary) < 1e-6 and c["score"][j] >= thr]
    seeds = [mask_util.decode({"size": [NATIVE, NATIVE], "counts": c["rle"][j].encode()}) for j in sel]
    if not sel:
        return seeds, []
    masks, _, _ = paste_back(P[idx[sel]], M[idx[sel]], seeds, t=t, fallback=fallback, merge=merge)
    return seeds, [m for m in masks if m.any()]


def apply(args):
    from .postprocess import encode
    cands = pickle.load(open(args.cands, "rb"))
    photos, primary = cands["photos"], cands["primary"]
    meta = np.load(args.crops_meta, allow_pickle=True)
    stems, mm, M = meta["stems"], meta["meta"], meta["M"]
    P = np.load(args.preds)["p"]
    thr = args.thr if args.thr is not None else cands.get("thr", 0.24)
    if args.data_root:                                       # out-of-fold evaluation against the annotators
        from .annotations import load_photos, photo_gt_rles
        from .metric import PQStats, photo_stats
        gtp = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
        base, ref = PQStats(), {t: PQStats() for t in args.t}
        for k, stem in enumerate(stems):
            idx = np.where(mm[:, 0] == k)[0]
            gt = {a: v for a, v in photo_gt_rles(gtp[str(stem)]).items() if v}
            seeds, _ = refined_photo(photos[str(stem)], idx, M, P, thr, 0.5, args.fallback, primary)
            base.add(photo_stats(gt, encode(seeds)))
            for t in args.t:
                _, out = refined_photo(photos[str(stem)], idx, M, P, thr, t, args.fallback, primary, not args.no_merge)
                ref[t].add(photo_stats(gt, encode(out)))
        print("selected seeds (no refinement):", base)
        for t in args.t:
            print(f"refined, t={t:.2f}:", ref[t])
        return
    import pandas as pd
    rows = []
    for k, stem in enumerate(stems):
        idx = np.where(mm[:, 0] == k)[0]
        _, out = refined_photo(photos[str(stem)], idx, M, P, thr, args.t[0], args.fallback, primary, not args.no_merge)
        rows += [{"filament_id": f"{stem}_{i}", "segmentation_rle": r} for i, r in enumerate(encode(out), 1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(rows)} filaments")


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data_root", required=True)
    p.add_argument("--cands", required=True)
    p.add_argument("--maps_root", required=True, help="folder holding the probability dirs named in the candidate file")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    q = sub.add_parser("train")
    q.add_argument("--crops", required=True)
    q.add_argument("--test_crops", default="")
    q.add_argument("--out_dir", default="work/refine")
    q.add_argument("--folds", type=int, nargs="*", default=[])
    q.add_argument("--encoder", default="resnet34")
    q.add_argument("--weights", default="imagenet")
    q.add_argument("--epochs", type=int, default=30)
    q.add_argument("--bs", type=int, default=48)
    q.add_argument("--lr", type=float, default=6e-4)
    q.add_argument("--ema", type=float, default=0.995)
    q.add_argument("--frag", type=int, default=0, help="1: train on the fragment-completing targets (T_frag)")
    q.add_argument("--neg_weight", type=float, default=1.0, help="share of crops without any ground truth (relative to their frequency)")
    q.add_argument("--val_every", type=int, default=5)
    q.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("apply")
    s.add_argument("--cands", required=True)
    s.add_argument("--crops_meta", required=True)
    s.add_argument("--preds", required=True)
    s.add_argument("--thr", type=float, default=None, help="score threshold of the selected seeds (default: the one in --cands)")
    s.add_argument("--t", type=float, nargs="+", default=[0.5], help="probability threshold(s) of the refined masks")
    s.add_argument("--fallback", default="seed", choices=["seed", "drop"])
    s.add_argument("--no_merge", action="store_true", help="do not merge seeds that were completed to the same filament")
    s.add_argument("--data_root", default="", help="evaluate against the annotators instead of writing a submission")
    s.add_argument("--out", default="submission_refined.csv")
    args = a.parse_args(argv)
    {"prep": prep, "train": train, "apply": apply}[args.cmd](args)


if __name__ == "__main__":
    main()
