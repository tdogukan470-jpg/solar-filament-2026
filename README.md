# solarfil — Solar Filament Instance Segmentation (Filament Segmentation Challenge 2026)

End-to-end, reproducible pipeline for segmenting individual solar filaments in GONG H-alpha
full-disk images (MAGFiLO), optimised for the competition's Panoptic Quality (PQ) metric.

## Method

| Stage | What | Why |
|---|---|---|
| Preprocessing (`preprocess.py`) | 3 channels: percentile-normalised intensity, limb-darkening-flattened image (division by a median/Gaussian background estimated inside the solar disk), CLAHE of the flattened image | filaments are low-contrast dark features; the disk brightness gradient varies across the 6 GONG sites |
| Targets (`annotations.py`) | polygons rasterised with pycocotools at 2048², averaged over all annotators of a photo (soft consensus), area-downsampled; plus an *instance-border* map where dilated filaments of the same annotator touch | the official metric scores a prediction against **every** annotator copy, so the consensus is the PQ-optimal target; the border channel lets touching filaments be split |
| Model (`model.py`) | U-Net, EfficientNet-B4 encoder (ImageNet), scSE attention, 2 outputs (foreground, border) | |
| Training (`train.py`) | 1536² working resolution, 768² crops (70 % centred on filaments), dihedral + photometric augmentation, BCE+soft-Dice, AdamW + cosine, EMA weights, 5-fold CV grouped by observation date | high resolution preserves thin barbs; date grouping avoids same-day leakage between GONG sites |
| Inference (`infer.py`) | full-disk prediction, 8× dihedral TTA, fold ensemble, upsampling to 2048² | |
| Instances (`postprocess.py`) | threshold → speck removal → fragment linking (dilation) → border cut → nearest-instance reassignment → hole filling → area / confidence filters; masks are disjoint by construction | PQ punishes fragmentation (FP + FN) and over-merging |
| Evaluation (`metric.py`) | exact re-implementation of the organisers' self-evaluation notebook using RLE IoUs (verified identical to the dense official code, ~1500× faster) | all tuning is done on out-of-fold predictions with the real metric |
| Tuning (`tune.py`) | coordinate-descent over post-processing parameters on OOF predictions | |

## Reproduce

```bash
pip install -r requirements.txt
DATA=/path/to/MAGFiLO_1.0_Kaggle_2026
for f in 0 1 2 3 4; do python -m solarfil.train --data_root $DATA --work work --fold $f; done
python -m solarfil.tune  --data_root $DATA --oof work/oof_Unet_efficientnet-b4_s1536 --out work/postcfg.json
python -m solarfil.infer --data_root $DATA --ckpts work/Unet_*.pth --postcfg work/postcfg.json --out submission.csv
```

`solar_filament_v2.ipynb` runs the same pipeline on Kaggle (GPU) and is generated from the
package by `make_notebook.py`.

## Notes on the data

* 707 training photos, 1154 annotator copies (1–3 annotators per photo), 8199 filaments,
  each a single polygon; median filament area ≈ 1 200 px at 2048².
* Leave-one-annotator-out agreement between human annotators is low (PQ ≈ 0.33 on a
  multi-annotator subset), which is why a soft consensus target is used.

## Repository layout

| Path | Contents |
|---|---|
| `solarfil/` | the main package (v2/v3 pipeline used for the best submission) |
| `make_notebook.py` | builds a self-contained Kaggle notebook embedding `solarfil/` |
| `kaggle_push/` | the Kaggle notebooks actually run (training folds, 2048 px models, tuning, 10-model GPU inference `sf-v3-infer-ens10-gpu`) |
| `experiments/v4_annotator_targets/` | annotator-conditioned hard targets — tested, did not beat the soft-consensus baseline |
| `experiments/v5_v6/` | time-blocked folds, spine/offset decoder (v5, worse) and UNet++ variant (v6, small ensemble gain) |
| `kaggle_ops.py` | local helper that drives the Kaggle CLI; reads the API key from `~/.kaggle/kaggle.json`, which is never committed |

Data, model weights and submission files are not stored in this repository.
