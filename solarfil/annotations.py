"""MAGFiLO COCO annotations -> ground-truth RLEs and soft training targets.

Important dataset facts (verified on the 2026 Kaggle train split):
  * 707 unique photos, 1154 COCO `images`: a photo can be annotated by 1-3 annotators.
    Each annotator copy has its own image id "<annotator>-<photo stem>".
  * The official metric scores the prediction for a photo against EVERY annotator copy
    separately (see metric.py), so the best prediction is a consensus of annotators.
  * Every annotation is a single filled polygon.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from pycocotools import mask as mask_util

NATIVE = 2048


@dataclass
class Photo:
    stem: str
    file_name: str
    # annotator image id -> list of polygons (flat [x0, y0, x1, y1, ...])
    annotators: dict = field(default_factory=dict)
    # annotator image id -> MAGFiLO chirality class of each polygon (1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous)
    categories: dict = field(default_factory=dict)


def load_photos(json_path: str) -> dict[str, Photo]:
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    photos: dict[str, Photo] = {}
    for im in coco["images"]:
        stem = os.path.splitext(im["file_name"])[0]
        p = photos.setdefault(stem, Photo(stem=stem, file_name=im["file_name"]))
        polys, cats = [], []
        for a in anns.get(im["id"], []):
            seg = a["segmentation"]
            if isinstance(seg, list):
                keep = [s for s in seg if len(s) >= 6]
                polys.extend(keep)
                cats.extend([a.get("category_id", 0)] * len(keep))
        p.annotators[im["id"]] = polys
        p.categories[im["id"]] = cats
    return photos


def poly_to_rle(poly: list, h: int = NATIVE, w: int = NATIVE) -> dict:
    rles = mask_util.frPyObjects([poly], h, w)
    return mask_util.merge(rles)


def photo_gt_rles(photo: Photo, classes=None) -> dict[str, list[dict]]:
    """annotator id -> list of COCO RLE dicts (one per filament), exactly like annToMask.
    classes: optional set of MAGFiLO classes to keep (1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous)."""
    if classes is None:
        return {aid: [poly_to_rle(p) for p in polys] for aid, polys in photo.annotators.items()}
    return {aid: [poly_to_rle(p) for p, c in zip(polys, photo.categories[aid]) if c in classes]
            for aid, polys in photo.annotators.items()}


def soft_targets(photo: Photo, size: int, border_px: int = 5, classes=None) -> tuple[np.ndarray, np.ndarray]:
    """Soft foreground and instance-border targets at (size, size), averaged over annotators.
    classes: optional set of MAGFiLO classes that count as foreground (others are background); annotator copies
    without any such filament do not exist in the matching ground-truth table and are not averaged in.

    foreground: fraction of annotators (and of the downsampled pixel area) covered by a filament.
    border    : pixels where dilated masks of two different filaments of the same annotator meet.
                Teaches the network where touching filaments must be split.
    """
    fg = np.zeros((size, size), np.float32)
    bd = np.zeros((size, size), np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_px + 1, 2 * border_px + 1))
    copies = [polys if classes is None else [p for p, c in zip(polys, photo.categories[aid]) if c in classes]
              for aid, polys in photo.annotators.items()]
    n_ann = max(len(photo.annotators) if classes is None else sum(bool(c) for c in copies), 1)
    for polys in copies:
        union = np.zeros((NATIVE, NATIVE), np.uint8)
        count = np.zeros((NATIVE, NATIVE), np.uint8)
        for p in polys:
            m = mask_util.decode(poly_to_rle(p))
            union |= m
            if len(polys) > 1:
                count += cv2.dilate(m, k)
        fg += cv2.resize(union.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        if len(polys) > 1:
            b = (count >= 2).astype(np.float32)
            bd += cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA)
    return fg / n_ann, np.clip(bd / n_ann, 0, 1)


def gt_frame(photos: dict[str, Photo], stems: list[str] | None = None):
    """Ground truth in the official submission format (annotator-prefixed filament ids)."""
    import pandas as pd

    rows = []
    for stem in stems or list(photos):
        for aid, polys in photos[stem].annotators.items():
            for i, p in enumerate(polys, 1):
                rle = poly_to_rle(p)
                rows.append({"filament_id": f"{aid}_{i}", "segmentation_rle": rle["counts"].decode()})
    return pd.DataFrame(rows)


def main(argv=None):
    import argparse
    a = argparse.ArgumentParser(description="Write the training ground truth in the submission format "
                                            "(one row per annotated filament, ids '<annotator>-<photo>_<k>').")
    a.add_argument("--data_root", required=True)
    a.add_argument("--out", required=True)
    args = a.parse_args(argv)
    df = gt_frame(load_photos(os.path.join(args.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")))
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} ground-truth filaments")


if __name__ == "__main__":
    main()
