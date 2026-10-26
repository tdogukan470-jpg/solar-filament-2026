# solarfil - Solar Filament Instance Segmentation (Filament Segmentation Challenge 2026)

End-to-end, reproducible pipeline for segmenting individual solar filaments in GONG H-alpha
full-disk images (MAGFiLO), optimised for the competition's Panoptic Quality (PQ) metric.

## Method

| Stage | What | Why |
|---|---|---|
| Preprocessing (`preprocess.py`) | 3 channels: percentile-normalised intensity, limb-darkening-flattened image (division by a median/Gaussian background estimated inside the solar disk, sigma=24), CLAHE of the flattened image | filaments are low-contrast dark features; the disk brightness gradient varies across the 6 GONG sites |
| Targets (`annotations.py`) | polygons rasterised at 2048^2, averaged over all annotators of a photo (soft consensus), area-downsampled; plus an *instance-border* map | the metric scores a prediction against **every** annotator copy, so the consensus is the PQ-optimal target; the border channel lets touching filaments be split |
| Model (`model.py`) | U-Net, EfficientNet-B4 encoder (ImageNet), scSE attention, 2 outputs (foreground, border) | |
| Training (`train.py`) | 1536^2 / 2048^2 working resolution, 768^2 crops (70 % centred on filaments), dihedral + photometric augmentation, BCE+soft-Dice, AdamW + cosine, EMA, 5-fold CV grouped by observation date | high resolution preserves thin barbs; date grouping avoids same-day leakage between GONG sites |
| Inference (`infer.py`) | full-disk prediction, 8x dihedral TTA, 10-model ensemble (5 folds x 2 resolutions), upsampling to 2048^2 | |
| Instances (`postprocess.py`) | threshold -> speck removal -> fragment linking -> hole filling -> area / confidence filters; masks are disjoint by construction | PQ punishes fragmentation (FP + FN) and over-merging |
| Evaluation (`metric.py`, `evaluate.py`) | exact re-implementation of the organisers' PQ; `evaluate.py` additionally reports mean Dice, SQ/DQ, Dice and IoU distributions and one-to-many / many-to-one relations | all tuning is done on out-of-fold predictions with the real metric |
| Tuning (`tune.py`) | coordinate descent over post-processing parameters on OOF predictions (saturated: held-out gain of any further variant < 0.001 PQ) | |
| Visualisation (`visualize.py`) | contour overlays of predicted filaments on the H-alpha image | |

## Reproduce

Open `pipeline.ipynb` and set `RUN_MODE`:

* `"REPRODUCE_SUBMISSION"` - downloads the 10 checkpoints listed in `configs/weights.json`
  (SHA-256 verified), predicts the 180 test photos and writes `submission.csv` + `figures/*.png`.
* `"TRAIN_FULL"` - retrains everything (5 folds x 2 resolutions, 20+ GPU hours), tunes the
  post-processing on out-of-fold predictions and predicts.

Command line equivalent:

```bash
pip install -r requirements.txt
DATA=/path/to/MAGFiLO_1.0_Kaggle_2026
python -m solarfil.weights --manifest configs/weights.json --dest weights
python -m solarfil.infer --data_root $DATA --ckpts weights/*.pth --postcfg configs/postcfg.json --out submission.csv
# full retraining
for f in 0 1 2 3 4; do python -m solarfil.train --data_root $DATA --work work --fold $f --size 1536 --epochs 36; done
for f in 0 1 2 3 4; do python -m solarfil.train --data_root $DATA --work work --fold $f --size 2048 --epochs 32; done
python -m solarfil.tune --data_root $DATA --oof work/oof_Unet_efficientnet-b4_s1536 --out work/postcfg.json
# rubric-style evaluation of any prediction file against any ground truth
python -m solarfil.evaluate --gt gt.csv --pred submission.csv --plots figures
```

## Results

Day-grouped 5-fold out-of-fold PQ (single 1536 model, 707 photos): 0.437 (SQ 0.674, DQ 0.650).
Human inter-annotator PQ on multi-annotator photos is only ~0.34 (oracle vs. all copies ~0.50), which bounds
achievable scores. Public leaderboard: 0.38.

## Notes on the data

* 707 training photos, 1154 annotator copies (1-3 annotators per photo), 8199 filaments,
  each a single polygon; median filament area ~1 200 px at 2048^2.
* Test inference uses only the single test image; no metadata, neighbouring observations or ground truth.
