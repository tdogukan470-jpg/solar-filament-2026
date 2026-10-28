"""Probability maps -> non-overlapping filament instances (COCO RLE at 2048x2048).

Panoptic Quality punishes two failure modes that plain "threshold + connected components"
produces on filaments:
  * fragmentation: one long faint filament breaks into several blobs -> every blob is an FP
    and the real filament an FN.            -> fix: link fragments closer than `link_px`.
  * over-merging : two neighbouring filaments touch -> one blob, IoU < 0.5 with both.
                                             -> possible fix: cut along the predicted border map (`border_thr`).
Annotators draw filled polygons, so instance holes are filled.

The submitted pipeline uses the candidate settings stored in `configs/rescorer_all.npz` (`cand_cfg`: threshold 0.5,
specks < 40 px removed, fragments closer than 14 px linked, candidates >= 100 px, border cutting and watershed seeds
off) and keeps the candidates that the learned filter (`rescore.py`) scores >= its threshold.  The defaults of `PostCfg`
and the (min_area, min_score) rule of `configs/postcfg.json` belong to the earlier, hand-tuned version of the pipeline;
they are used only by `python -m solarfil.infer` without `--rescorer` (the baseline row of the ablation, `tune.py`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage as ndi

NATIVE = 2048


@dataclass
class PostCfg:
    thr: float = 0.45          # foreground probability threshold
    close_px: int = 0          # morphological closing radius before labelling
    min_cc: int = 40           # drop tiny specks before linking (native px)
    link_px: int = 8           # fragments closer than this are one filament (native px)
    border_thr: float = 1.01   # >1 disables border splitting
    min_area: int = 250        # final instance area filter (native px)
    min_score: float = 0.50    # mean foreground prob inside instance
    seed_thr: float = 1.01     # <=1: split instances by watershed from high-confidence seeds
    fill_holes: bool = True

    def as_dict(self):
        return asdict(self)


def _disk(r: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def upsample(p: np.ndarray, size: int = NATIVE) -> np.ndarray:
    p = np.ascontiguousarray(p, dtype=np.float32)
    if p.shape[0] == size:
        return p
    return cv2.resize(p, (size, size), interpolation=cv2.INTER_LINEAR)


def instances(fg: np.ndarray, border: np.ndarray | None, cfg: PostCfg, want_feats: bool = False):
    """fg/border: float maps in [0,1] at any square resolution. Returns list of uint8 masks (2048²)."""
    fg = upsample(fg)
    binary = (fg > cfg.thr).astype(np.uint8)
    if cfg.close_px > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _disk(cfg.close_px))
    if not binary.any():
        return ([], []) if want_feats else []

    # 1) remove specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_cc
    binary = keep[lab].astype(np.uint8)
    if not binary.any():
        return ([], []) if want_feats else []

    # 2) link nearby fragments, 3) cut along predicted borders
    core = cv2.dilate(binary, _disk(cfg.link_px // 2)) if cfg.link_px > 1 else binary.copy()
    use_border = border is not None and cfg.border_thr <= 1.0
    if use_border:
        cut = upsample(border) > cfg.border_thr
        core[cut] = 0
    if cfg.seed_thr <= 1.0:
        # seeds = linked high-confidence cores; grow them over the foreground along the probability
        seeds = ((fg > cfg.seed_thr) & (core > 0)).astype(np.uint8)
        if cfg.link_px > 1:
            seeds = cv2.dilate(seeds, _disk(cfg.link_px // 2)) & core
        n, markers = cv2.connectedComponents(seeds, connectivity=8)
        from skimage.segmentation import watershed
        lab = watershed(-fg, markers, mask=core > 0).astype(np.int32)
        # components without any seed keep their own label
        n2, lab2 = cv2.connectedComponents(((core > 0) & (lab == 0)).astype(np.uint8), connectivity=8)
        lab[lab2 > 0] = lab2[lab2 > 0] + n
    else:
        n, lab = cv2.connectedComponents(core, connectivity=8)
        lab = lab.astype(np.int32)
    lab[binary == 0] = 0
    if use_border:
        # border pixels that belong to the foreground: give them to the nearest instance
        orphan = (binary > 0) & (lab == 0)
        if orphan.any() and n > 1:
            _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
            lab[orphan] = lab[iy[orphan], ix[orphan]]

    # 4) per-instance filtering
    out, feats = [], []
    objs = ndi.find_objects(lab)
    n_obj = sum(o is not None for o in objs)
    bd_full = upsample(border) if (border is not None and want_feats) else None
    for i, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = lab[sl] == i
        if cfg.fill_holes:
            m = ndi.binary_fill_holes(m)
            m &= (lab[sl] == i) | (lab[sl] == 0)  # never steal pixels of another instance
        area = int(m.sum())
        if area < cfg.min_area:
            continue
        pv = fg[sl][m]
        if float(pv.mean()) < cfg.min_score:
            continue
        full = np.zeros((NATIVE, NATIVE), np.uint8)
        full[sl][m] = 1
        out.append(full)
        if want_feats:
            feats.append(_features(m, sl, pv, bd_full[sl][m] if bd_full is not None else None, n_obj))
    return (out, feats) if want_feats else out


FEATURES = ["log_area", "mean_p", "max_p", "p10", "p50", "p90", "std_p", "f50", "f70", "f90",
            "mean_bd", "h", "w", "fill", "elong", "r_center", "n_obj"]


def _features(m, sl, pv, bv, n_obj):
    h = sl[0].stop - sl[0].start
    w = sl[1].stop - sl[1].start
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
    if len(ys) > 2:
        ev = np.linalg.eigvalsh(np.cov(np.vstack([ys, xs])))
        elong = float(np.sqrt(max(ev[-1], 1e-6) / max(ev[0], 1e-6)))
    else:
        elong = 1.0
    q = np.percentile(pv, [10, 50, 90])
    return [np.log(m.sum()), float(pv.mean()), float(pv.max()), q[0], q[1], q[2], float(pv.std()),
            float((pv > .5).mean()), float((pv > .7).mean()), float((pv > .9).mean()),
            float(bv.mean()) if bv is not None else 0.0, h, w, m.sum() / max(h * w, 1), min(elong, 50.0),
            float(np.hypot(cy - 1024, cx - 1024) / 1024), n_obj]


FEATURES_EXT = FEATURES + [
    "skel_len", "width", "solidity", "cos2t", "sin2t", "radial_cos", "cx", "cy", "ring_p", "contrast",
    "n_near150", "d_nearest", "rank_mean", "ph_ncand", "ph_fgmean", "ph_fgfrac", "ph_p99", "ph_area_cand",
    "frac_p80",
]


def candidate_features(fg: np.ndarray, masks: list[np.ndarray], base_feats: list) -> np.ndarray:
    """Per-candidate descriptors used by the learned re-scorer (rescore.py): the 17 `FEATURES` plus
    shape (skeleton length, mean width, solidity, orientation), position, contrast against a 12 px ring
    and context (neighbours, rank of the instance's confidence, photo-level activity).
    fg: foreground probability at 2048^2; masks/base_feats come from instances(..., want_feats=True)."""
    from skimage.morphology import skeletonize

    n = len(masks)
    if n == 0:
        return np.zeros((0, len(FEATURES_EXT)), np.float32)
    cents = np.array([np.argwhere(m).mean(0) for m in masks])           # (y, x)
    mean_p = np.array([b[1] for b in base_feats])
    rank = (-mean_p).argsort().argsort() / max(n - 1, 1)
    areas = np.array([np.exp(b[0]) for b in base_feats])
    ph = [int((areas >= 300).sum()), float(fg.mean()), float((fg > 0.35).mean()),
          float(np.percentile(fg[::4, ::4], 99.5)), float(areas[areas >= 300].sum())]
    ring_k = _disk(12)
    out = []
    for i, (m, b) in enumerate(zip(masks, base_feats)):
        ys, xs = np.nonzero(m)
        y0, y1 = max(ys.min() - 14, 0), min(ys.max() + 15, NATIVE)
        x0, x1 = max(xs.min() - 14, 0), min(xs.max() + 15, NATIVE)
        mm = m[y0:y1, x0:x1].astype(np.uint8)
        fgc = fg[y0:y1, x0:x1]
        skel_len = float(skeletonize(mm > 0).sum())
        area = float(mm.sum())
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = max(sum(cv2.contourArea(cv2.convexHull(c)) for c in cnts), 1.0)
        cov = np.cov(np.vstack([ys, xs])) if len(ys) > 2 else np.eye(2)
        _, evec = np.linalg.eigh(cov)
        v = evec[:, -1]                                                   # principal direction (dy, dx)
        th = np.arctan2(v[0], v[1])
        cy, cx = cents[i]
        ry, rx = cy - NATIVE / 2, cx - NATIVE / 2
        radial = abs((v[0] * ry + v[1] * rx) / (np.hypot(ry, rx) + 1e-6))
        ring = (cv2.dilate(mm, ring_k) > 0) & (mm == 0)
        ring_p = float(fgc[ring].mean()) if ring.any() else 0.0
        if n > 1:
            d = np.hypot(cents[:, 0] - cy, cents[:, 1] - cx)
            d[i] = 1e9
            near, nearest = int((d < 150).sum()), float(min(d.min(), 2000))
        else:
            near, nearest = 0, 2000.0
        out.append(list(b) + [skel_len, area / max(skel_len, 1.0), area / hull, np.cos(2 * th), np.sin(2 * th),
                              radial, cx / NATIVE, cy / NATIVE, ring_p, b[1] - ring_p, near, nearest, rank[i],
                              *ph, float((fgc[mm > 0] > 0.8).mean())])
    return np.asarray(out, np.float32)


def mask_features(fg: np.ndarray, bd: np.ndarray | None, masks: list[np.ndarray]) -> np.ndarray:
    """FEATURES_EXT of arbitrary instance masks (e.g. the refined masks of stage 2), computed exactly like those of the
    candidates found by `instances` so that the learned re-scorer can be applied to them."""
    fg = upsample(fg)
    bd = upsample(bd) if bd is not None else None
    feats = []
    for m in masks:
        ys, xs = np.nonzero(m)
        sl = (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))
        mm = m[sl] > 0
        feats.append(_features(mm, sl, fg[sl][mm], bd[sl][mm] if bd is not None else None, len(masks)))
    return candidate_features(fg, masks, feats)


def load_avg(paths: list[str]) -> np.ndarray:
    """Average uint8 probability npz files (2,S,S) of possibly different resolutions."""
    ps = [np.load(f)["p"].astype(np.float32) / 255.0 for f in paths]
    top = max(p.shape[-1] for p in ps)
    ps = [p if p.shape[-1] == top else np.stack([upsample(c, top) for c in p]) for p in ps]
    return np.mean(ps, axis=0)


def encode(masks: list[np.ndarray]) -> list[str]:
    return [mask_util.encode(np.asfortranarray(m))["counts"].decode() for m in masks]


def instances_rle(fg, border, cfg: PostCfg, rescorer=None) -> list[str]:
    """rescorer: optional dict {"model": fitted regressor, "thr": float} (see rescore.py). With a rescorer, `cfg` is the
    permissive candidate configuration and the learned score replaces the hand-tuned min_area / min_score rule."""
    if rescorer is None:
        return encode(instances(fg, border, cfg))
    masks, feats = instances(fg, border, cfg, want_feats=True)
    if not masks:
        return []
    s = rescorer["model"].predict(candidate_features(upsample(fg), masks, feats))
    return encode([m for m, v in zip(masks, s) if v >= rescorer["thr"]])
