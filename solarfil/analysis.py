"""Numbers of the report that Table 1 (`ablate.py`) does not cover: error analysis, threshold scan, mask thickness, ...

Every sub-command reads the candidate file written by `python -m solarfil.cands` in out-of-fold mode (the candidates of
the 707 annotated photographs with cross-fitted learned-filter scores), keeps the candidates with score >= tau and scores
them against every annotator copy exactly like the official metric (IoU > 0.5, pooled over annotator copies).

    python -m solarfil.analysis numbers --data_root D --cands cands_oof.pkl [--tau 0.20] [--boot 2000] [--out_dir report]
        PQ / SQ / DQ and the rubric measures with day-level bootstrap intervals, the overlap structure, PQ by number of
        annotator copies, the fate of every annotated filament (matched / dropped by the filter / error of extent /
        missed), PQ per fold, per-photograph PQ (report/per_photo.csv) and the photographs closest to the 25th / 50th /
        75th percentile among those with >= 8 annotated filaments per copy (the example figure)
    python -m solarfil.analysis annotators --data_root D
        agreement of two annotators of the same photograph (the last row of Table 1): PQ, SQ, DQ and semantic Dice
        over all ordered pairs of annotator copies (no candidate file needed)
    python -m solarfil.analysis tau    --data_root D --cands cands_oof.pkl [--taus 0 0.1 0.2 0.3]
        PQ, semantic Dice and pair statistics as a function of the filter threshold
    python -m solarfil.analysis hand   --data_root D --cands cands_oof.pkl
        the hand-tuned (min area, min mean probability) rule of Table 1 on a wider grid (is its optimum inside the grid?)
    python -m solarfil.analysis radius --data_root D --cands cands_oof.pkl [--workers 2]
        PQ after growing / shrinking every selected mask by r = -2 ... +2 pixels, pooled and per annotator, with the
        PQ-optimal (fractional) radius of every annotator: annotators differ in how thick they draw
    python -m solarfil.analysis pairs  --data_root D --cands cands_oof.pkl [--workers 2]
        area / skeleton length / width of matched prediction-annotation pairs (IoU > 0.5)
    python -m solarfil.analysis extent --data_root D --cands cands_oof.pkl
        errors of extent: area ratio, recall and precision of the best candidate of the annotated filaments that it
        overlaps without matching them, for small and large annotations
    python -m solarfil.analysis shift  --cands cands_oof.pkl --test_cands cands_test.pkl [--tau 0.20]
        candidates per photograph, share kept by the filter and learned-score quantiles on the out-of-fold and on the test
        photographs (a shift between the two would show up here; no ground truth needed)
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .ablate import pq as _pq3
from .ablate import stats
from .annotations import gt_frame, load_photos, photo_gt_rles
from .cands import to_frame
from .data import make_folds
from .evaluate import bootstrap, evaluate

SIZE = [2048, 2048]
RADII = (-2, -1, 0, 1, 2)
_G: dict = {}


def _init(g):
    _G.update(g)


def _load(a):
    cands = pickle.load(open(a.cands, "rb"))
    photos = load_photos(os.path.join(a.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    return cands, photos, sorted(cands["photos"])


def selected(cands: dict, stem: str, tau: float) -> list[dict]:
    """COCO RLEs of the primary candidates of one photograph whose learned score is >= tau."""
    c = cands["photos"][stem]
    return [{"size": SIZE, "counts": r.encode()} for r, lv, s in zip(c["rle"], c["level"], c["score"])
            if lv == cands["primary"] and s >= tau]


def iou_matrix(pred: list[dict], gt: list[dict]) -> np.ndarray:
    """(len(pred), len(gt)) IoU matrix of two lists of RLEs (empty lists allowed)."""
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)))
    return np.asarray(mask_util.iou(pred, gt, [0] * len(gt))).reshape(len(pred), len(gt))


def fates(iou: np.ndarray, sel: np.ndarray) -> Counter:
    """What happened to every annotated filament (column of `iou`, candidates x annotations).

    matched: a selected candidate has IoU > 0.5; dropped: only candidates that the filter dropped do; extent: candidates
    overlap it (IoU >= 0.1) but none reaches 0.5; missed: no candidate overlaps it (IoU < 0.1)."""
    out: Counter = Counter()
    for j in range(iou.shape[1]):
        col = iou[:, j]
        if (col[sel] > 0.5).any():
            out["matched"] += 1
        elif (col > 0.5).any():
            out["dropped"] += 1
        elif (col >= 0.1).any():
            out["extent"] += 1
        else:
            out["missed"] += 1
    return out


def extent_failures(iou: np.ndarray, a_pred: np.ndarray, a_gt: np.ndarray) -> np.ndarray:
    """Annotations (columns of `iou`, candidates x annotations) whose best candidate overlaps them (IoU >= 0.1) without
    reaching IoU 0.5: rows (annotation area, candidate area, recall, precision) of the best candidate, recall and
    precision of the pair being |A and B| / |annotation| and |A and B| / |candidate| (|A and B| follows from the IoU)."""
    out = []
    for j in range(iou.shape[1]):
        i = int(np.argmax(iou[:, j])) if iou.shape[0] else -1
        if i < 0 or not 0.1 <= iou[i, j] <= 0.5:
            continue
        inter = iou[i, j] * (a_pred[i] + a_gt[j]) / (1.0 + iou[i, j])
        out.append((a_gt[j], a_pred[i], inter / a_gt[j], inter / a_pred[i]))
    return np.array(out, float).reshape(-1, 4)


def shrink_grow(m: np.ndarray, r: int) -> np.ndarray:
    """Binary mask grown (r > 0) or shrunk (r < 0) by |r| pixels (elliptical kernel of (2|r|+1)^2 pixels).
    A mask that vanishes stays as a one-pixel mask, i.e. still counts as a false positive."""
    if r == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(r) + 1, 2 * abs(r) + 1))
    out = cv2.dilate(m, k) if r > 0 else cv2.erode(m, k)
    if not out.any():
        out = np.zeros_like(m)
        out[0, 0] = 1
    return out


def vertex(p, radii=RADII) -> float:
    """Fractional position of the maximum of the curve p(radii): vertex of the parabola through the maximum and its two
    neighbours (the maximum is moved inside the range if it sits at an end)."""
    p = np.asarray(p, float)
    i = min(max(int(np.argmax(p)), 1), len(p) - 2)
    d = p[i - 1] - 2 * p[i] + p[i + 1]
    return float(radii[i] + (0.5 * (p[i - 1] - p[i + 1]) / d if d < 0 else 0.0))


# ------------------------------------------------------------------------------------------------ numbers
def numbers(cands, photos, stems, tau, n_boot, out_dir=""):
    gt = gt_frame(photos, stems)
    pred = to_frame(cands, tau)
    pred["stem"] = pred.filament_id.str.rsplit("_", n=1).str[0]
    res = evaluate(gt, pred[["filament_id", "segmentation_rle"]])
    print(f"tau {tau:.2f}: {len(pred)} instances in {pred.stem.nunique()} of {len(stems)} photographs")
    print("PQ %.4f SQ %.4f DQ %.4f  TP %d FP %d FN %d" % tuple(res[k] for k in ("PQ", "SQ", "DQ", "TP", "FP", "FN")))
    for k in ("mean_dice_semantic", "mean_dice_overlapping_pairs", "mean_iou_overlapping_pairs"):
        print(f"{k}: {res[k]:.4f}")
    for k, (m, lo, hi) in bootstrap(res, n_boot).items():
        print(f"{k}: {m:.4f} [{lo:.4f}, {hi:.4f}]  (95 % percentile interval, days resampled)")
    d_pred, d_gt = res["pred_covering_n_gt"], res["gt_split_into_n_predictions"]
    n_p, n_g = sum(d_pred.values()), sum(d_gt.values())
    print("predictions (per annotator copy) overlapping exactly one annotation: %.1f %%, two or more: %.1f %%; "
          "annotations overlapped by two or more predictions: %.1f %%"
          % (100 * d_pred.get(1, 0) / n_p, 100 * sum(v for k, v in d_pred.items() if k >= 2) / n_p,
             100 * sum(v for k, v in d_gt.items() if k >= 2) / n_g))

    print("\nPQ by number of annotator copies of the photograph")
    for name, ok in (("all", lambda n: True), ("1 copy", lambda n: n == 1), ("2 copies", lambda n: n == 2),
                     ("3 copies", lambda n: n == 3), (">=2 copies", lambda n: n >= 2)):
        st = [s for s in stems if ok(len(photos[s].annotators))]
        r = evaluate(gt_frame(photos, st), pred[pred.stem.isin(st)][["filament_id", "segmentation_rle"]])
        print(f"{name:11s} photos {len(st):4d} annotated filaments {r['TP'] + r['FN']:5d}  PQ {r['PQ']:.4f} SQ {r['SQ']:.4f} "
              f"DQ {r['DQ']:.4f}  TP {r['TP']} FP {r['FP']} FN {r['FN']}  semantic Dice {r['mean_dice_semantic']:.4f}")

    cnt: Counter = Counter()
    for s in stems:
        c = cands["photos"][s]
        idx = [i for i, lv in enumerate(c["level"]) if lv == cands["primary"]]
        rl = [{"size": SIZE, "counts": c["rle"][i].encode()} for i in idx]
        sel = np.array([c["score"][i] >= tau for i in idx], bool)
        for g in photo_gt_rles(photos[s]).values():
            cnt += fates(iou_matrix(rl, g), sel)
    tot = sum(cnt.values())
    print("\nfate of the annotated filaments (pooled over annotator copies):",
          {k: f"{cnt[k]} ({cnt[k] / tot:.3f})" for k in ("matched", "dropped", "extent", "missed")}, tot)

    folds = make_folds(list(photos), 5, 42).set_index("stem")["fold"].to_dict()
    pf = []
    for f in range(5):
        st = [s for s in stems if folds[s] == f]
        pf.append(evaluate(gt_frame(photos, st), pred[pred.stem.isin(st)][["filament_id", "segmentation_rle"]])["PQ"])
    print("PQ per stage-1 fold:", np.round(pf, 4), "min", round(min(pf), 4), "max", round(max(pf), 4))

    rec = pd.DataFrame(res["_records"], columns=["stem", "iou", "tp", "fp", "fn", "dice"])
    g = rec.groupby("stem").sum()
    g["pq"] = g["iou"] / (g["tp"] + 0.5 * g["fp"] + 0.5 * g["fn"]).clip(lower=1e-9)
    g["copies"] = rec.groupby("stem").size()
    g["n_gt_per_copy"] = (g["tp"] + g["fn"]) / g["copies"]
    g["dice"] = rec.groupby("stem")["dice"].mean()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        g.to_csv(os.path.join(out_dir, "per_photo.csv"))
    big = g[g["n_gt_per_copy"] >= 8]
    print(f"\n{len(g)} photographs, {len(big)} with >= 8 annotated filaments per copy; the example figure shows the photographs "
          "nearest to the quantiles of the per-photograph PQ of these")
    for q in (0.25, 0.5, 0.75):
        v = big["pq"].quantile(q)
        near = (big["pq"] - v).abs().sort_values().index[:4]
        print(f"quantile {q}: PQ {v:.3f}")
        print(big.loc[near, ["pq", "tp", "fp", "fn", "copies", "n_gt_per_copy", "dice"]].round(3).to_string())


# ------------------------------------------------------------------------------------------------ annotators
def annotators(photos, stems):
    """Agreement of two annotators of the same photograph: every ordered pair of annotator copies (one plays the
    prediction, the other the ground truth), PQ pooled over the pairs like the official metric."""
    tot, dice, n_ph = np.zeros(4), [], 0
    for s in stems:
        G = {a: g for a, g in photo_gt_rles(photos[s]).items() if g}
        if len(G) < 2:
            continue
        n_ph += 1
        for a, b in itertools.permutations(G, 2):
            tot += stats([iou_matrix(G[a], G[b])], np.ones(len(G[a]), bool))
            ua, ub = mask_util.merge(G[a]), mask_util.merge(G[b])
            inter = mask_util.area(mask_util.merge([ua, ub], intersect=True))
            dice.append(2.0 * inter / max(mask_util.area(ua) + mask_util.area(ub), 1))
    p, sq, dq = _pq3(tot)
    print(f"{n_ph} photographs with >= 2 annotator copies, {len(dice)} ordered pairs: PQ {p:.4f} SQ {sq:.4f} DQ {dq:.4f}  "
          f"TP {int(tot[1])} FP {int(tot[2])} FN {int(tot[3])}; mean semantic Dice {np.mean(dice):.4f}")


# ------------------------------------------------------------------------------------------------ tau
def tau_scan(cands, photos, stems, taus):
    gt = gt_frame(photos, stems)
    print("tau   instances  PQ     SQ     DQ     TP    FP    FN   semantic-Dice pair-Dice pair-IoU")
    for tau in taus:
        pred = to_frame(cands, tau)
        r = evaluate(gt, pred)
        print(f"{tau:.2f} {len(pred):8d}  {r['PQ']:.4f} {r['SQ']:.4f} {r['DQ']:.4f} {r['TP']:5d} {r['FP']:5d} {r['FN']:5d}   "
              f"{r['mean_dice_semantic']:.4f}      {r['mean_dice_overlapping_pairs']:.4f}    {r['mean_iou_overlapping_pairs']:.4f}", flush=True)


# ------------------------------------------------------------------------------------------------ hand
def hand_grid(cands, photos, stems, areas=(100, 150, 200, 250, 300),
              scores=(0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)):
    """PQ of 'keep candidates with area >= A and mean foreground probability >= S' (no learned score) on all candidates."""
    rows = {}
    for s in stems:
        c = cands["photos"][s]
        n_p = int((c["level"] == cands["primary"]).sum())
        pr = [{"size": SIZE, "counts": r.encode()} for r in c["rle"][:n_p]]
        rows[s] = (c["X"][:n_p], [iou_matrix(pr, g) for g in photo_gt_rles(photos[s]).values() if g], c["score"][:n_p])
    tot = sum(stats(i, np.ones(len(x), bool)) for x, i, _ in rows.values())
    print("all candidates (no filter):             PQ %.4f  TP %d FP %d FN %d" % (_pq3(tot)[0], tot[1], tot[2], tot[3]))
    tot = sum(stats(i, sc >= 0.20) for _, i, sc in rows.values())
    print("learned filter, score >= 0.20:          PQ %.4f  TP %d FP %d FN %d" % (_pq3(tot)[0], tot[1], tot[2], tot[3]))
    print("hand rule, PQ by min area A (rows) and min mean probability S (columns)")
    print("A \\ S " + " ".join(f"{S:6.2f}" for S in scores))
    best = (-1.0, None)
    for A in areas:
        line = []
        for S in scores:
            t = sum(stats(i, (np.exp(x[:, 0]) >= A - 1e-6) & (x[:, 1] >= S)) for x, i, _ in rows.values())
            line.append(_pq3(t)[0])
            best = max(best, (line[-1], (A, S)))
        print(f"A={A:4d} " + " ".join(f"{v:.4f}" for v in line))
    print("best hand rule on this grid: PQ %.4f at (min area, min mean probability) = %s" % best)


# ------------------------------------------------------------------------------------------------ radius
def _radius_job(stem):
    rl = _G["sel"][stem]
    base = [mask_util.decode(r) for r in rl]
    out = {}
    for r in RADII:
        enc = [mask_util.encode(np.asfortranarray(shrink_grow(m, r))) for m in base]
        for cid, g in _G["gts"][stem].items():
            out[(cid, r)] = stats([iou_matrix(enc, g)], np.ones(len(enc), bool)) if g else np.zeros(4)
    return stem, out


def radius(cands, photos, stems, tau, workers, n_boot=1000):
    gts = {s: photo_gt_rles(photos[s]) for s in stems}
    sel = {s: selected(cands, s, tau) for s in stems}
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=({"gts": gts, "sel": sel},)) as ex:
        for k, (s, o) in enumerate(ex.map(_radius_job, stems, chunksize=4)):
            res.update({(s,) + key: v for key, v in o.items()})
            if k % 100 == 0:
                print(k, len(stems), f"{time.time() - t0:.0f}s", flush=True)

    def curve(arrs):
        t = np.sum(arrs, axis=0)
        return t[:, 0] / (t[:, 1] + 0.5 * t[:, 2] + 0.5 * t[:, 3])

    per = defaultdict(dict)                                    # annotator -> photograph -> (len(RADII), 4)
    for (stem, cid, r), v in res.items():
        per[cid.split("-")[0]].setdefault(stem, np.zeros((len(RADII), 4)))[RADII.index(r)] += v
    allp = curve([a for d in per.values() for a in d.values()])
    print("\nPQ of all annotator copies after growing (+) / shrinking (-) every mask by r pixels:")
    print("r      " + "  ".join(f"{r:+d}    " for r in RADII))
    print("PQ     " + "  ".join(f"{p:.4f}" for p in allp), f"  PQ-optimal radius {vertex(allp):+.2f}")
    rng = np.random.default_rng(0)
    out = []
    for a, d in per.items():
        if len(d) < 30:
            continue
        arrs = np.array(list(d.values()))
        p = curve(arrs)
        vs = [vertex(curve(arrs[rng.integers(0, len(arrs), len(arrs))])) for _ in range(n_boot)]
        out.append((vertex(p), a, len(arrs), p, *np.percentile(vs, [2.5, 97.5])))
    print("\nper annotator (>= 30 photographs): PQ-optimal radius [95 % interval over photographs]; "
          "PQ gain of shrinking / growing by one pixel")
    for v, a, n, p, lo, hi in sorted(out):
        print(f"{a}  photos {n:3d}  PQ(r=0) {p[2]:.4f}  optimum {v:+.2f} [{lo:+.2f}, {hi:+.2f}]  "
              f"PQ(-1)-PQ(0) {p[1] - p[2]:+.4f}  PQ(+1)-PQ(0) {p[3] - p[2]:+.4f}")
    vv = np.array([o[0] for o in out])
    print(f"{len(out)} annotators: optimum from {vv.min():+.2f} to {vv.max():+.2f} px (median {np.median(vv):+.2f}); "
          f"interval excludes 0 for {sum(1 for o in out if o[4] > 0 or o[5] < 0)}")


# ------------------------------------------------------------------------------------------------ pairs
def _pairs_job(stem):
    from .morphology import describe
    pr = _G["sel"][stem]
    rows, cache = [], {}
    for cid, g in _G["gts"][stem].items():
        iou = iou_matrix(pr, g)
        for i, j in zip(*np.nonzero(iou > 0.5)):
            if i not in cache:
                cache[i] = describe(mask_util.decode(pr[i]))
            rows.append((cache[i], describe(mask_util.decode(g[j]))))
    return rows


def pairs(cands, photos, stems, tau, workers):
    from .morphology import COLUMNS
    gts = {s: photo_gt_rles(photos[s]) for s in stems}
    sel = {s: selected(cands, s, tau) for s in stems}
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=({"gts": gts, "sel": sel},)) as ex:
        for r in ex.map(_pairs_job, stems, chunksize=4):
            rows += r
    P, G = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    print(f"{len(rows)} matched (annotator copy, prediction) pairs with IoU > 0.5")
    for name in ("area", "length", "width"):
        j = COLUMNS.index(name)
        ratio = P[:, j] / np.maximum(G[:, j], 1e-6)
        print(f"{name:7s} median: annotation {np.median(G[:, j]):7.1f}  prediction {np.median(P[:, j]):7.1f}   "
              f"median of prediction/annotation {np.median(ratio):.3f} (quartiles {np.percentile(ratio, 25):.3f}, "
              f"{np.percentile(ratio, 75):.3f}); prediction larger in {100 * np.mean(P[:, j] > G[:, j]):.0f} % of the pairs")


# ------------------------------------------------------------------------------------------------ extent
def extent(cands, photos, stems, tau, split=1200):
    """Errors of extent: what the best selected candidate of an annotation that it overlaps but does not match looks like."""
    rows = []
    n_ann = 0
    for s in stems:
        sel = selected(cands, s, tau)
        a_pred = np.asarray(mask_util.area(sel), float) if sel else np.zeros(0)
        for g in photo_gt_rles(photos[s]).values():
            n_ann += len(g)
            if g:
                rows.append(extent_failures(iou_matrix(sel, g), a_pred, np.asarray(mask_util.area(g), float)))
    d = np.concatenate(rows) if rows else np.zeros((0, 4))
    print(f"{len(d)} of {n_ann} annotated filaments ({100 * len(d) / n_ann:.1f} %) are overlapped by a selected candidate "
          f"(IoU >= 0.1) but never matched (IoU <= 0.5)")
    print(f"{'annotation':>16s} {'n':>5s} {'share':>6s} {'median candidate/annotation area':>34s} {'recall':>7s} {'precision':>10s}")
    for name, k in ((f"<= {split} px", d[:, 0] <= split), (f"> {split} px", d[:, 0] > split), ("all", d[:, 0] > 0)):
        if k.any():
            print(f"{name:>16s} {k.sum():5d} {100 * k.mean():5.1f}% {np.median(d[k, 1] / d[k, 0]):34.2f} "
                  f"{np.median(d[k, 2]):7.2f} {np.median(d[k, 3]):10.2f}")


# ------------------------------------------------------------------------------------------------ shift
def _primary_scores(cands: dict) -> np.ndarray:
    return np.concatenate([c["score"][c["level"] == cands["primary"]] for c in cands["photos"].values()])


def shift(oof: dict, test: dict, tau: float):
    """Learned scores of the primary candidates of the out-of-fold photographs and of the test photographs."""
    print(f"{'':>6s} {'photos':>7s} {'cand./photo':>12s} {'share >= tau':>13s} {'mean score':>11s}   score quantiles 10 / 25 / 50 / 75 / 90 %")
    for name, c in (("OOF", oof), ("test", test)):
        sc = _primary_scores(c)
        print(f"{name:>6s} {len(c['photos']):7d} {len(sc) / len(c['photos']):12.2f} {100 * np.mean(sc >= tau):12.1f}% "
              f"{sc.mean():11.3f}   " + " / ".join(f"{q:.3f}" for q in np.quantile(sc, [0.1, 0.25, 0.5, 0.75, 0.9])))


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("shift")
    p.add_argument("--cands", required=True, help="out-of-fold candidate file (python -m solarfil.cands)")
    p.add_argument("--test_cands", required=True, help="candidate file of the test photographs (python -m solarfil.cands --probs ...)")
    p.add_argument("--tau", type=float, default=0.20)
    for name in ("numbers", "annotators", "tau", "hand", "radius", "pairs", "extent"):
        p = sub.add_parser(name)
        p.add_argument("--data_root", required=True)
        p.add_argument("--cands", required=name != "annotators", default="",
                       help="out-of-fold candidate file (python -m solarfil.cands)")
        p.add_argument("--tau", type=float, default=0.20, help="score threshold of the learned filter")
        p.add_argument("--workers", type=int, default=2)
        if name == "numbers":
            p.add_argument("--boot", type=int, default=2000)
            p.add_argument("--out_dir", default="")
        if name == "tau":
            p.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.14, 0.18, 0.20, 0.22, 0.26, 0.30, 0.40])
    args = a.parse_args(argv)
    if args.cmd == "shift":
        return shift(pickle.load(open(args.cands, "rb")), pickle.load(open(args.test_cands, "rb")), args.tau)
    if args.cmd == "annotators":
        photos = load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
        return annotators(photos, sorted(photos))
    cands, photos, stems = _load(args)
    if args.cmd == "numbers":
        numbers(cands, photos, stems, args.tau, args.boot, args.out_dir)
    elif args.cmd == "tau":
        tau_scan(cands, photos, stems, args.taus)
    elif args.cmd == "hand":
        hand_grid(cands, photos, stems)
    elif args.cmd == "radius":
        radius(cands, photos, stems, args.tau, args.workers)
    elif args.cmd == "extent":
        extent(cands, photos, stems, args.tau)
    else:
        pairs(cands, photos, stems, args.tau, args.workers)


if __name__ == "__main__":
    main()
