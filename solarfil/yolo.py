"""Instance-detector candidate source (EXP-026): YOLO11-seg trained on the MAGFiLO annotations.

Why: our stage 1 is a semantic network (probability map -> components). A detector of a different family proposes
filaments independently; its instances can be added as extra candidates in front of the learned filter. Only training
images and their annotations are used; inference uses only the image itself (same input channels as stage 1).

    python -m solarfil.yolo prep    --data_root DATA --out /kaggle/temp/yolo --fold 0
    python -m solarfil.yolo train   --out /kaggle/temp/yolo --work /kaggle/working --epochs 80
    python -m solarfil.yolo predict --out /kaggle/temp/yolo --weights W --split val --dst preds_val.pkl

Training labels: every annotator copy of a photograph is a separate training sample (same image, that copy's
polygons), so the detector sees the annotators' disagreement in proportion to the data. Folds are the day-grouped
folds of solarfil.data.make_folds (seed 42), identical to stage 1, so fold-k predictions are out of fold. --fold -1 trains on all 707 photographs (for the test predictions).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .annotations import NATIVE, load_photos
from .data import make_folds
from .preprocess import load_input


def _write_png(args):
    src, dst = args
    if not os.path.exists(dst):
        x = load_input(src, NATIVE)  # (2048, 2048, 3) uint8: percentile, flat-field, CLAHE
        cv2.imwrite(dst, x[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return dst


def _write_sample(out, split, name, src, stem, polys):
    img_link = os.path.join(out, split, "images", name + ".png")
    if not os.path.exists(img_link):
        os.symlink(os.path.join(src, stem + ".png"), img_link)
    with open(os.path.join(out, split, "labels", name + ".txt"), "w") as f:
        for poly in polys:
            xy = np.clip(np.asarray(poly, np.float64) / NATIVE, 0, 1)
            f.write("0 " + " ".join(f"{v:.6f}" for v in xy) + "\n")


def prep(a):
    tr_json = os.path.join(a.data_root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    img_dir = os.path.join(a.data_root, "train", "train_images")
    photos = load_photos(tr_json)
    photos = {k: v for k, v in photos.items() if os.path.exists(os.path.join(img_dir, v.file_name))}
    folds = make_folds(list(photos), 5, 42)
    fold_of = dict(zip(folds.stem, folds.fold))
    src = os.path.join(a.out, "src")
    os.makedirs(src, exist_ok=True)
    test_paths = sorted(glob.glob(os.path.join(a.data_root, "test", "test_images", "*.jp*g")))
    jobs = [(os.path.join(img_dir, p.file_name), os.path.join(src, s + ".png")) for s, p in photos.items()]
    jobs += [(t, os.path.join(src, "test__" + os.path.splitext(os.path.basename(t))[0] + ".png")) for t in test_paths]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(_write_png, jobs, chunksize=4)):
            if i % 100 == 0:
                print(f"png {i}/{len(jobs)}", flush=True)
    n_tr = n_lab = 0
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
    for s, p in photos.items():
        split = "val" if fold_of[s] == a.fold else "train"
        copies = list(p.annotators.items()) if split == "train" else list(p.annotators.items())[:1]
        if a.fold < 0 and fold_of[s] == 0:  # all-data model: Ultralytics still needs a val folder for its final
            _write_sample(a.out, "val", f"{s}__0", src, s, copies[0][1])  # report; those photos are ALSO trained on
        for k, (_aid, polys) in enumerate(copies):
            name = f"{s}__{k}"
            img_link = os.path.join(a.out, split, "images", name + ".png")
            if not os.path.exists(img_link):
                os.symlink(os.path.join(src, s + ".png"), img_link)
            with open(os.path.join(a.out, split, "labels", name + ".txt"), "w") as f:
                for poly in polys:
                    xy = np.clip(np.asarray(poly, np.float64) / NATIVE, 0, 1)
                    f.write("0 " + " ".join(f"{v:.6f}" for v in xy) + "\n")
                    n_lab += 1
            n_tr += split == "train"
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {a.out}\ntrain: train/images\nval: val/images\nnames:\n  0: filament\n")
    with open(os.path.join(a.out, "split.pkl"), "wb") as f:
        pickle.dump({"val": sorted(s for s in photos if fold_of[s] == a.fold),
                     "test": [os.path.splitext(os.path.basename(t))[0] for t in test_paths], "fold": a.fold}, f)
    print(f"fold {a.fold}: train samples={n_tr} polygons={n_lab} test={len(test_paths)}", flush=True)


def train(a):
    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=os.path.join(a.out, "data.yaml"), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch,
                workers=a.workers, device=a.device, seed=0, deterministic=False, cos_lr=True, close_mosaic=10,
                mosaic=0.5, mixup=0.0, flipud=0.5, fliplr=0.5, degrees=10.0, scale=0.3, translate=0.1,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, cls=a.cls, mask_ratio=a.mask_ratio, val=False, plots=False,
                project=a.work, name="yolo", exist_ok=True, amp=True, fraction=a.fraction, **({"time": a.time} if a.time > 0 else {}))


def predict(a):
    import torch
    from pycocotools import mask as mask_util
    from ultralytics import YOLO
    with open(os.path.join(a.out, "split.pkl"), "rb") as f:
        split = pickle.load(f)
    src = os.path.join(a.out, "src")
    names = split[a.split]
    paths = [os.path.join(src, ("test__" if a.split == "test" else "") + s + ".png") for s in names]
    model = YOLO(a.weights)
    out = {}
    for s, p in zip(names, paths):
        r = model.predict(p, imgsz=a.imgsz, conf=a.conf, iou=0.5, retina_masks=True, max_det=150,
                          verbose=False, half=torch.cuda.is_available())[0]
        inst = []
        if r.masks is not None:
            m = (r.masks.data > 0.5).to(torch.uint8).cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for mi, c in zip(m, conf):
                if mi.shape != (NATIVE, NATIVE):
                    mi = cv2.resize(mi, (NATIVE, NATIVE), interpolation=cv2.INTER_NEAREST)
                if mi.sum() == 0:
                    continue
                rle = mask_util.encode(np.asfortranarray(mi))
                inst.append({"counts": rle["counts"].decode(), "size": list(rle["size"]), "conf": float(c)})
        out[s] = inst
    with open(a.dst, "wb") as f:
        pickle.dump(out, f)
    n = sum(len(v) for v in out.values())
    print(f"predict {a.split}: {len(out)} photos, {n} instances (conf >= {a.conf}) -> {a.dst}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--data_root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, default=0); p.add_argument("--workers", type=int, default=4)
    t = sp.add_parser("train"); t.add_argument("--out", required=True); t.add_argument("--work", required=True)
    t.add_argument("--model", default="yolo11m-seg.pt"); t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--epochs", type=int, default=80); t.add_argument("--batch", type=int, default=4)
    t.add_argument("--workers", type=int, default=4); t.add_argument("--cls", type=float, default=1.5)
    t.add_argument("--mask_ratio", type=int, default=2); t.add_argument("--device", default="0")
    t.add_argument("--fraction", type=float, default=1.0); t.add_argument("--time", type=float, default=0.0)
    q = sp.add_parser("predict"); q.add_argument("--out", required=True); q.add_argument("--weights", required=True)
    q.add_argument("--split", choices=("val", "test"), required=True); q.add_argument("--dst", required=True)
    q.add_argument("--imgsz", type=int, default=1280); q.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args(argv)
    {"prep": prep, "train": train, "predict": predict}[a.cmd](a)


if __name__ == "__main__":
    main()
