"""Image preprocessing for GONG H-alpha full-disk observations.

Filaments are dark, elongated absorption features on the solar disk. Their contrast
is low and the disk brightness falls off towards the limb (limb darkening) and varies
between the six GONG sites. We therefore build a 3-channel input:

  ch0  global intensity (percentile-normalised)
  ch1  flattened image  = image / smooth background  (removes limb darkening / gradients)
  ch2  local contrast    = CLAHE of the flattened image

All channels are uint8 so they can be cached cheaply.
"""
from __future__ import annotations

import cv2
import numpy as np

NATIVE = 2048


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def disk_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of the solar disk (Otsu on a blurred image, largest component, filled)."""
    small = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        th = (lab == k).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(th)
    if cnts:
        cv2.drawContours(filled, cnts, -1, 255, -1)
    filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.resize(filled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def flatten(img: np.ndarray, mask: np.ndarray, sigma: float = 24.0) -> np.ndarray:
    """Divide by a smooth background estimated inside the disk (normalised convolution).

    The Gaussian is applied at 1/4 resolution with sigma = 2 * `sigma` / 4 pixels, i.e. its effective standard deviation
    is 2 * `sigma` = 48 native pixels for the default `sigma` = 24 (kept as is: the stored networks were trained with it)."""
    s = 4  # work at 1/4 resolution for speed; the background is smooth anyway
    h, w = img.shape
    small = cv2.resize(img.astype(np.float32), (w // s, h // s), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float32), (w // s, h // s), interpolation=cv2.INTER_AREA)
    # median filter first so dark filaments don't pull the background down
    med = cv2.medianBlur(np.clip(small, 0, 255).astype(np.uint8), 15).astype(np.float32)
    num = cv2.GaussianBlur(med * m, (0, 0), sigma / s * 2)
    den = cv2.GaussianBlur(m, (0, 0), sigma / s * 2)
    bg = num / np.maximum(den, 1e-3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    ratio = img.astype(np.float32) / np.maximum(bg, 1.0)
    ratio[~mask] = 1.0
    # map ratio 0.6..1.25 -> 0..255
    out = (ratio - 0.6) / (1.25 - 0.6) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def make_input(img: np.ndarray) -> np.ndarray:
    """Return (H, W, 3) uint8 network input at native resolution."""
    mask = disk_mask(img)
    vals = img[mask] if mask.any() else img.ravel()
    lo, hi = np.percentile(vals, [0.5, 99.5])
    ch0 = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    ch1 = flatten(img, mask)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    ch2 = clahe.apply(ch1)
    ch2[~mask] = 0
    return np.dstack([ch0, ch1, ch2])


def load_input(path: str, size: int) -> np.ndarray:
    """Read an observation and return the 3-channel input resized to (size, size)."""
    x = make_input(read_gray(path))
    if size != x.shape[0]:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    return x
