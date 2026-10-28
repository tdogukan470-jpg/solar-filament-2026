"""Caching, folds and PyTorch datasets."""
from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .annotations import Photo, soft_targets
from .preprocess import load_input


# ----------------------------------------------------------------------------- folds
def make_folds(stems: list[str], n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Group by observation date: same-day photos from different GONG sites see the same filaments."""
    df = pd.DataFrame({"stem": sorted(stems)})
    df["group"] = df["stem"].str[:8]
    groups = df["group"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(groups)
    gfold = {g: i % n_folds for i, g in enumerate(groups)}
    df["fold"] = df["group"].map(gfold)
    return df


# ----------------------------------------------------------------------------- cache
def _cache_one(args):
    photo, img_path, cache_dir, size, border_px, classes = args
    xp = os.path.join(cache_dir, f"{photo.stem}_x.npy")
    yp = os.path.join(cache_dir, f"{photo.stem}_y.npy")
    if os.path.exists(xp) and os.path.exists(yp):
        return photo.stem
    x = load_input(img_path, size)
    fg, bd = soft_targets(photo, size, border_px=border_px, classes=classes)
    y = np.dstack([np.round(fg * 255), np.round(bd * 255)]).astype(np.uint8)
    np.save(xp, x)
    np.save(yp, y)
    return photo.stem


def build_cache(photos: dict[str, Photo], img_dir: str, cache_dir: str, size: int,
                border_px: int = 5, workers: int = 4, classes=None):
    os.makedirs(cache_dir, exist_ok=True)
    jobs = [(p, os.path.join(img_dir, p.file_name), cache_dir, size, border_px, classes) for p in photos.values()]
    if workers <= 1:
        for j in jobs:
            _cache_one(j)
    else:
        with ProcessPoolExecutor(workers) as ex:
            for i, _ in enumerate(ex.map(_cache_one, jobs, chunksize=4)):
                if i % 100 == 0:
                    print(f"cache {i}/{len(jobs)}", flush=True)


def build_test_cache(paths: list[str], cache_dir: str, size: int, workers: int = 4):
    os.makedirs(cache_dir, exist_ok=True)

    todo = [p for p in paths
            if not os.path.exists(os.path.join(cache_dir, os.path.splitext(os.path.basename(p))[0] + "_x.npy"))]
    with ProcessPoolExecutor(max(workers, 1)) as ex:
        for p, x in zip(todo, ex.map(load_input, todo, [size] * len(todo))):
            np.save(os.path.join(cache_dir, os.path.splitext(os.path.basename(p))[0] + "_x.npy"), x)


# ----------------------------------------------------------------------------- augmentation
def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7: rot90 by k%4, then horizontal flip if k>=4."""
    a = np.rot90(a, k % 4, axes=(0, 1))
    if k >= 4:
        a = a[:, ::-1]
    return a


def photometric(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    if rng.random() < 0.8:  # contrast / brightness / gamma, per channel jitter
        c = rng.uniform(0.75, 1.3)
        b = rng.uniform(-0.08, 0.08)
        g = rng.uniform(0.75, 1.35)
        x = np.clip((x - 0.5) * c + 0.5 + b, 0, 1) ** g
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape).astype(np.float32)
    if rng.random() < 0.2:
        import cv2
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.2))
    return np.clip(x, 0, 1)


# ----------------------------------------------------------------------------- datasets
class CropDataset(Dataset):
    """Random crops from cached full-disk arrays; `pos_frac` of crops are centred on a filament."""

    def __init__(self, stems, cache_dir, crop=768, samples_per_epoch=None, pos_frac=0.7, seed=0):
        self.stems = list(stems)
        self.cache_dir = cache_dir
        self.crop = crop
        self.n = samples_per_epoch or 2 * len(self.stems)
        self.pos_frac = pos_frac
        self.seed = seed
        self._fgidx = {}

    def __len__(self):
        return self.n

    def _arrays(self, stem):
        x = np.load(os.path.join(self.cache_dir, f"{stem}_x.npy"), mmap_mode="r")
        y = np.load(os.path.join(self.cache_dir, f"{stem}_y.npy"), mmap_mode="r")
        return x, y

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1_000_003 + i + random.getrandbits(31)) % 2**32)
        stem = self.stems[rng.integers(len(self.stems))]
        x, y = self._arrays(stem)
        S, c = x.shape[0], self.crop
        if rng.random() < self.pos_frac:
            if stem not in self._fgidx:
                ys, xs = np.nonzero(np.asarray(y[::8, ::8, 0]) > 64)
                self._fgidx[stem] = (ys * 8, xs * 8)
            ys, xs = self._fgidx[stem]
            if len(ys):
                j = rng.integers(len(ys))
                cy = int(ys[j] + rng.integers(-c // 3, c // 3 + 1))
                cx = int(xs[j] + rng.integers(-c // 3, c // 3 + 1))
            else:
                cy, cx = rng.integers(c // 2, S - c // 2 + 1, size=2)
        else:
            cy, cx = rng.integers(c // 2, S - c // 2 + 1, size=2)
        y0 = int(np.clip(cy - c // 2, 0, S - c))
        x0 = int(np.clip(cx - c // 2, 0, S - c))
        xc = np.asarray(x[y0:y0 + c, x0:x0 + c])
        yc = np.asarray(y[y0:y0 + c, x0:x0 + c]).astype(np.float32) / 255.0
        k = int(rng.integers(8))
        xc, yc = dihedral(xc, k), dihedral(yc, k)
        xc = photometric(np.ascontiguousarray(xc), rng)
        return (torch.from_numpy(np.ascontiguousarray(xc.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(yc.transpose(2, 0, 1))))


def load_full(stem: str, cache_dir: str) -> np.ndarray:
    """Full cached input as float32 CHW tensor in [0,1]."""
    x = np.load(os.path.join(cache_dir, f"{stem}_x.npy"))
    return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32) / 255.0)
