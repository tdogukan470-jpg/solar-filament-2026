"""H-alpha overlays of predicted filament masks (contours on the original image)."""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .preprocess import read_gray


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
