"""Apparent morphology of filaments: physically interpretable descriptors of every instance.

A filament seen in H-alpha on the disk is a dark ribbon that follows a polarity-inversion line; its apparent shape tells
whether a segmentation is plausible even where an annotation is missing.  For every instance mask we measure

    area, length (skeleton), width (area / length), elongation, curvature (skeleton length / end-to-end extent),
    orientation of the principal axis against the image x axis, the acute angle between the axis and the radial
    direction (90 deg = tangential, parallel to the limb) and the radial position r / R_ANN of the centroid.

    python -m solarfil.morphology --gt gt.csv --pred submission.csv --out figures/morphology.png
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from pycocotools import mask as mask_util

NATIVE = 2048
R_ANN = 900.0            # annotated filaments never extend beyond ~900 px from the disk centre (disk radius is ~960-1000 px)
COLUMNS = ["area", "length", "width", "elongation", "curvature", "theta", "radial_angle", "r_norm"]


def describe(m: np.ndarray) -> list[float]:
    """Descriptors of one binary mask (2048 x 2048, any dtype)."""
    from skimage.morphology import skeletonize
    ys, xs = np.nonzero(m)
    area = float(len(ys))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    skel = skeletonize(m[max(y0 - 1, 0):y1 + 1, max(x0 - 1, 0):x1 + 1] > 0)
    length = max(float(skel.sum()), 1.0)
    cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
    ev, evec = np.linalg.eigh(cov)
    vx, vy = evec[:, -1]
    theta = math.degrees(math.atan2(vy, vx))
    theta = (theta + 90.0) % 180.0 - 90.0                                   # axis, not direction: [-90, 90)
    cx, cy = xs.mean() - NATIVE / 2, ys.mean() - NATIVE / 2
    rr = math.hypot(cx, cy)
    radial = abs(vx * cx + vy * cy) / (rr + 1e-6)                           # |cos| of the angle axis <-> radial direction
    proj = (xs - xs.mean()) * vx + (ys - ys.mean()) * vy
    extent = max(float(proj.max() - proj.min()), 1.0)
    return [area, length, area / length, math.sqrt(max(ev[-1], 1e-6) / max(ev[0], 1e-6)), max(length / extent, 1.0), theta,
            math.degrees(math.acos(min(radial, 1.0))), rr / R_ANN]


def table(rles: list) -> np.ndarray:
    """(N, len(COLUMNS)) descriptors of COCO RLEs (dict or counts string)."""
    out = []
    for r in rles:
        if not isinstance(r, dict):
            r = {"size": [NATIVE, NATIVE], "counts": r.encode() if isinstance(r, str) else r}
        out.append(describe(mask_util.decode(r)))
    return np.asarray(out, np.float32).reshape(-1, len(COLUMNS))


def plot(tables: dict[str, np.ndarray], path: str) -> None:
    """Histograms of the descriptors for several sets (e.g. annotations vs predictions), densities for comparability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = [("area", "area [px$^2$]", np.geomspace(100, 60000, 30), True), ("length", "skeleton length [px]", np.geomspace(10, 1500, 30), True),
            ("width", "mean width [px]", np.linspace(2, 60, 30), False), ("curvature", "curvature (length / extent)", np.linspace(1, 2.5, 30), False),
            ("theta", "axis angle to image x axis [deg]", np.linspace(-90, 90, 37), False), ("radial_angle", "angle to radial direction [deg]", np.linspace(0, 90, 19), False)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    for ax, (key, label, bins, logx) in zip(axes.ravel(), spec):
        j = COLUMNS.index(key)
        for name, t in tables.items():
            if len(t):
                ax.hist(t[:, j], bins=bins, density=True, histtype="step", linewidth=1.6, label=f"{name} (n={len(t)})")
        ax.set_xlabel(label)
        if logx:
            ax.set_xscale("log")
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary(t: np.ndarray) -> str:
    q = lambda k: np.percentile(t[:, COLUMNS.index(k)], [25, 50, 75]).round(1).tolist()
    tang = float((t[:, COLUMNS.index("radial_angle")] > 60).mean())
    return (f"n={len(t)}  area q25/50/75 {q('area')}  length {q('length')}  width {q('width')}  "
            f"share with axis within 30 deg of the tangent {tang:.2f}")


def main(argv=None):
    import pandas as pd
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--gt", default="", help="annotation csv (filament_id, segmentation_rle)")
    a.add_argument("--pred", required=True)
    a.add_argument("--out", default="morphology.png")
    a.add_argument("--max_n", type=int, default=0, help="debug: describe at most N instances per set")
    args = a.parse_args(argv)
    sets = {"prediction": pd.read_csv(args.pred)}
    if args.gt:
        sets = {"annotation": pd.read_csv(args.gt), **sets}
    tabs = {}
    for name, df in sets.items():
        rl = df["segmentation_rle"].tolist()[:args.max_n or None]
        tabs[name] = table(rl)
        print(name, summary(tabs[name]))
    plot(tabs, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
