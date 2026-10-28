"""H-alpha overlays of predicted filament masks (contours on the original image)."""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .preprocess import make_input, read_gray


def decode(rle: str, size: int = 2048) -> np.ndarray:
    """COCO RLE string -> uint8 mask (size x size)."""
    return mask_util.decode({"size": [size, size], "counts": rle.encode()})


def overlay(img: np.ndarray, rles: list[str], gt_rles: list[str] | None = None) -> np.ndarray:
    """Draw prediction contours (yellow, numbered) and optional GT contours (cyan) on a gray image."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for rs, col in ((gt_rles or [], (255, 255, 0)), (rles, (0, 220, 255))):
        for i, r in enumerate(rs, 1):
            cnts, _ = cv2.findContours(decode(r), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cnts, -1, col, 2)
            if col == (0, 220, 255) and cnts:
                x, y, _, _ = cv2.boundingRect(cnts[0])
                cv2.putText(out, str(i), (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return out


def render_submission(sub_csv: str, img_dir: str, out_dir: str, n: int = 12, size: int = 1024) -> list[str]:
    """Write overlay PNGs for the ``n`` test images with the most predicted filaments."""
    df = pd.read_csv(sub_csv)
    df["stem"] = df["filament_id"].str.split("_", n=1).str[0]
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for stem in df["stem"].value_counts().index[:n]:
        src = next(os.path.join(img_dir, f"{stem}{e}") for e in (".jpg", ".jpeg", ".png")
                   if os.path.exists(os.path.join(img_dir, f"{stem}{e}")))
        vis = overlay(read_gray(src), df.loc[df["stem"] == stem, "segmentation_rle"].tolist())
        path = os.path.join(out_dir, f"{stem}.png")
        cv2.imwrite(path, cv2.resize(vis, (size, size), interpolation=cv2.INTER_AREA))
        paths.append(path)
    return paths


def pipeline_figure(img_path: str, probs_path: str, rescorer_path: str, out: str, roi: int = 1024, dpi: int = 110) -> str:
    """Walk-through of the inference stages on one photograph (a ``roi`` x ``roi`` window with the most filament area):

    (1) H-alpha photograph, (2) flat-fielded network input, (3) stage-1 foreground probability, (4) candidate instances
    (threshold + fragment linking) coloured by the learned score g_hat, (5) kept (green, g_hat >= tau) and dropped (red)
    candidates, (6) the submitted instances.  ``probs_path`` is a map written by ``infer.py --save_probs``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figures import best_roi, rle_dict
    from .postprocess import PostCfg, candidate_features, instances, upsample
    from .rescore import load_rescorer

    img = read_gray(img_path)
    flat = make_input(img)[..., 1]
    p = np.load(probs_path)["p"].astype(np.float32) / 255.0
    resc = load_rescorer(rescorer_path)
    masks, feats = instances(p[0], p[1], PostCfg(**resc["cand_cfg"]), want_feats=True)
    fg = upsample(p[0])
    score = resc["model"].predict(candidate_features(fg, masks, feats)) if masks else np.zeros(0)
    rles = [mask_util.encode(np.asfortranarray(m)) for m in masks]
    y0, x0 = best_roi([rles], roi) if rles else (0, 0)
    sl = (slice(y0, y0 + roi), slice(x0, x0 + roi))
    rgb = lambda a: np.repeat(a[sl][..., None], 3, axis=2)

    def contours(sel, colors):
        canvas = rgb(img).copy()
        for i in sel:
            m = np.ascontiguousarray(masks[i][sl])
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, cnts, -1, colors(i), 3)
            if cnts:
                x, y, _, _ = cv2.boundingRect(cnts[0])
                for col, th in (((0, 0, 0), 5), (colors(i), 2)):              # dark halo keeps the label readable
                    cv2.putText(canvas, f"{score[i]:.2f}", (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th)
        return canvas

    inside = [i for i in range(len(masks)) if masks[i][sl].any()]
    viridis = matplotlib.colormaps["viridis"]
    by_score = lambda i: tuple(int(255 * c) for c in viridis(float(np.clip(score[i] / 0.6, 0, 1)))[:3])
    keep = [i for i in inside if score[i] >= resc["thr"]]
    drop = [i for i in inside if score[i] < resc["thr"]]
    panels = [
        ("1  H-alpha photograph", rgb(img)),
        ("2  flat-fielded network input", rgb(flat)),
        ("3  stage 1: foreground probability", (matplotlib.colormaps["magma"](fg[sl])[..., :3] * 255).astype(np.uint8)),
        (f"4  candidates ({len(inside)}), colour = learned score", contours(inside, by_score)),
        (f"5  filter: keep {len(keep)} (green), drop {len(drop)} (red), tau = {resc['thr']:.2f}",
         contours(inside, lambda i: (60, 230, 60) if score[i] >= resc["thr"] else (255, 60, 60))),
        ("6  submitted instances", contours(keep, lambda i: (60, 230, 60))),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.4))
    for ax, (title, im) in zip(axes.ravel(), panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
