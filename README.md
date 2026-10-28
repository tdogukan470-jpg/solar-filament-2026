# Solar filament instance segmentation - Filament Segmentation Challenge 2026

Pipeline for the Kaggle competition *Solar Filament Segmentation Challenge 2026*
([`filament-segmentation-2026`](https://www.kaggle.com/competitions/filament-segmentation-2026)): instance
segmentation of solar filaments in GONG H-alpha full-disk observations (MAGFiLO dataset), optimised for the competition's
Panoptic Quality (PQ) metric. Author: Doğukan Turan.

The method in one paragraph: ten U-Nets (EfficientNet-B4 encoder, five folds grouped by observation day, two input
resolutions) trained on the *soft consensus* of the annotators give a foreground and an instance-border probability map;
connected components of the thresholded map (with short gaps closed) are filament *candidates*; a gradient-boosted model,
fitted on out-of-fold candidates, predicts each candidate's expected contribution to PQ and the candidates above a single
threshold are submitted. The 4-page report is submitted separately through the competition form.

## Results

Every number is computed on **out-of-fold (OOF)** predictions: a photograph is only ever predicted by networks that did not
see it (its map is the average of the 1536^2 and the 2048^2 network of its held-out fold: two members, whereas the
deployed ensemble averages ten), and the learned filter is cross-fitted over the same folds: the filter scores of the
photographs of fold k come from a filter fitted on the candidates of the other four folds (Table 1 uses the 6429
candidates of the two-network maps; the deployed filter is fitted on 12,648 candidates, see "Learned instance filter"
below). The hand-tuned rule is the best cell of a two-parameter grid search on all candidates (optimistic for the rule;
its optimum, mean probability >= 0.90 with any area limit between 100 and 250 px, is an interior point of a wider grid:
`python -m solarfil.analysis hand`). 707 annotated
photographs, 8199 annotated filaments, five folds grouped by observation day. Intervals are 95 % percentile bootstrap
intervals that resample observation days (photographs of one day are near-duplicates).

| post-processing of the stage-1 maps | PQ [95 % CI] | SQ | DQ |
|---|---|---|---|
| threshold 0.5, connected components | 0.396 [0.387, 0.405] | 0.678 | 0.584 |
| + fragment linking (14 px) | 0.408 [0.399, 0.417] | 0.677 | 0.603 |
| + hand-tuned area / score filter (area >= 100 px, mean prob. >= 0.9) | 0.445 [0.435, 0.455] | 0.682 | 0.652 |
| **+ learned instance filter (final)** | **0.451 [0.441, 0.461]** | 0.680 | 0.663 |
| *annotator vs. annotator (same photograph)* | *0.340* | *0.634* | *0.536* |

* The learned filter improves on the best hand-tuned rule by +0.006 PQ, paired-bootstrap interval [+0.004, +0.009]. It
  halves the false positives (4825 -> 2466) for a loss of 6 % of the true positives (5624 -> 5294). PQ per fold:
  0.428 ... 0.469. `python -m solarfil.ablate` recomputes this table (see "Reproduce Table 1").
* Where the remaining error is (annotated filaments, pooled over annotator copies): 64.6 % are matched at IoU > 0.5; 4.0 %
  have a matching candidate that the filter dropped; 22.0 % overlap a candidate but never reach IoU 0.5 (errors of extent:
  PQ is extremely sensitive to a one-pixel boundary shift of a thin filament); 9.4 % have no candidate at all. The errors
  of extent are of two kinds (`python -m solarfil.analysis extent`; submitted candidates only, 1364 annotations, 16.6 %):
  43 % concern small annotations (<= 1200 px) whose best candidate is a median 2.2 x larger (recall 0.87, precision 0.40),
  57 % larger annotations of which the best candidate covers about half (area ratio 0.54, recall 0.46, precision 0.82).
* Rubric measures on the OOF predictions: semantic Dice of the union masks 0.676 [0.665, 0.686]; mean Dice / IoU of
  intersecting prediction-annotation pairs 0.73 / 0.59; 82.6 % of the predictions overlap exactly one annotation of an
  annotator copy and 3.4 % two or more, 2.8 % of the annotations are split between several predictions (one-to-many /
  many-to-one overlaps are rare, the errors are errors of extent, not of splitting or merging).
* The annotators' own disagreement caps what any single prediction can reach: two annotators of the same photograph reach
  PQ 0.34 and a semantic Dice of 0.61 against each other, whereas our consensus prediction reaches PQ 0.450 and Dice 0.68
  against each of the annotators of the 296 photographs that have several (the consensus target is closer to every
  annotator than the annotators are to each other).
* Public leaderboard: the previous version of this pipeline (hand-tuned filter instead of the learned one) scored 0.38,
  i.e. 0.85-0.87 x the OOF PQ of its hand-tuned rule (0.437-0.445 depending on the rule's parameters). The OOF PQ is
  optimistic (the best epoch of every network is selected on its held-out fold, and the post-processing parameters were
  chosen on the same OOF predictions) and pessimistic (two network members instead of ten) at the same time; we cannot
  separate these effects.
* Annotators differ in how thick they draw. Growing or shrinking every mask by one pixel costs 0.026 PQ on the pooled OOF
  set (optimum: radius 0), but the PQ-optimal uniform radius estimated per annotator ranges from about -0.7 to +1.0 px
  (`python -m solarfil.analysis radius`). On matched prediction-annotation pairs the prediction has about the area of the
  annotation (median ratio 1.05) but a shorter skeleton (117 vs 140 px) and a larger width (12.3 vs 9.8 px)
  (`python -m solarfil.analysis pairs`).

### Examples

![one test photograph, stage by stage](docs/pipeline_demo.jpg)

*One test photograph, stage by stage (`solarfil.visualize.pipeline_figure`): H-alpha image, flat-fielded network input,
stage-1 foreground probability, candidate instances coloured by the learned score g_hat, the filter's decision
(tau = 0.20) and the submitted instances.*

![out-of-fold examples](docs/examples_oof.jpg)

*Out-of-fold examples (`python -m solarfil.figures oof`): photographs with at least eight annotated filaments per copy at the median (PQ 0.46) and the 25th percentile (PQ 0.38) of per-photograph PQ. Columns:
photograph, annotator copies (cyan / orange / magenta), prediction (green = matched at IoU > 0.5, red = unmatched
prediction, yellow = annotation without a match).*

![test examples](docs/examples_test.jpg)

*Predictions of the submitted pipeline on three test photographs (`python -m solarfil.figures test`; no ground truth exists for them).*

![overlap statistics](docs/summary.png)

*Overlap and morphology statistics of 8199 annotated and 4772 predicted filaments.*

## Method

| Stage | Module | What and why |
|---|---|---|
| Pre-processing | `preprocess.py` | 3 channels: percentile-normalised intensity; *flat-field* image (intensity divided by a smooth background estimated inside the Otsu-detected disk from a median-filtered copy, Gaussian with sigma = 48 px); CLAHE of the flat-field image. Removes limb darkening and site-to-site gradients. |
| Targets | `annotations.py` | Polygons rasterised at 2048^2 and averaged over all annotator copies of a photograph (**soft consensus**, because the metric scores a prediction against *every* copy and the copies disagree); plus an instance-border map (pixels where the 5-px dilations of two different filaments of one annotator meet). |
| Stage 1 network | `model.py`, `train.py` | U-Net, ImageNet-pretrained EfficientNet-B4 encoder, scSE decoder attention (`segmentation_models_pytorch`), two outputs (foreground, border). BCE (positive weight 3) + soft Dice, plus 0.5 x (BCE + 0.5 Dice) on the border. 768^2 crops of the photograph resized to 1536^2 or 2048^2 (70 % centred on a filament), dihedral + photometric augmentation, AdamW (lr 6e-4, cosine), 36 / 32 epochs (1536^2 / 2048^2), batch 6, mixed precision, EMA 0.998 of the weights; the epoch with the best held-out PQ is kept. Five folds **grouped by observation day**, so near-duplicate photographs of one day never straddle folds. |
| Inference | `infer.py`, `model.py` | Whole photograph, 8 dihedral transforms, ensemble of the 10 networks (5 folds x 2 resolutions), probabilities averaged at 2048^2: 80 forward passes per photograph. |
| Candidates | `postprocess.py` | Foreground threshold 0.5, components < 40 px removed, fragments closer than 14 px linked (dilation of radius 7), at least 100 px; 9.1 candidates per photograph on average (more than the 7.1 annotated filaments per annotator copy). Masks are disjoint by construction, as the submission format requires. The border map is trained, but cutting along it did not improve OOF PQ and is switched off. |
| Learned instance filter | `rescore.py` | 36 features per candidate (probability statistics inside it, shape: area / skeleton length / width / solidity / orientation, position, contrast against a 12-px ring, context: neighbours, rank of its confidence, photograph-level activity). A histogram gradient-boosting regressor predicts the candidate's **expected PQ gain** g = mean over annotator copies of IoU x [IoU > 0.5]. Adding a candidate changes the PQ denominator TP + FP/2 + FN/2 by exactly +1/2 (match: TP+1, FN-1; no match: FP+1) and the numerator by its IoU, so it improves PQ iff g > PQ/2 ~ 0.22; we keep candidates with g_hat >= 0.20 (the OOF-PQ maximum; PQ stays within 0.002 of it for thresholds in [0.18, 0.26]). Fitted on OOF candidates only: the deployed filter on the candidates extracted from the maps of the 1536^2 network alone and from those of the 1536^2 + 2048^2 pair (6219 + 6429 = 12,648 candidates of 707 photographs; a view with more members is closer to the final ensemble), the cross-fitted scores of Table 1 on the 6429 candidates of the pair maps; the stored file holds the training table, not a pickled model, so it does not depend on the scikit-learn version. |
| Out-of-fold maps | `oof.py` | Regenerates the out-of-fold probability maps from the released checkpoints (each network predicts its held-out fold), so the filter can be refitted without retraining. |
| Evaluation | `metric.py`, `evaluate.py`, `morphology.py`, `analysis.py` | Re-implementation of the official PQ (pooled over annotator copies, IoU > 0.5) and of the rubric measures (Dice, IoU, one-to-many / many-to-one), day-level bootstrap intervals, skeleton length / width statistics; `analysis.py` recomputes the other numbers of the report (error analysis, threshold scan, annotator agreement, mask thickness per annotator, matched pairs). |
| Visualisation | `visualize.py`, `figures.py` | H-alpha overlays, a step-by-step figure of one photograph, report figures. |
| Stage 2 (negative result, **not** in the submitted pipeline) | `refine.py`, `cands.py` | Instance-conditioned re-segmentation of every selected candidate in its own rotated 128 x 384 crop (7 input channels: photograph, stage-1 probabilities, the candidate and its neighbours; U-Net with ResNet-34 encoder; five folds cross-fitted like stage 1). Starting from the candidates of an earlier version of the filter (4688 instances, OOF PQ 0.4497) the refined masks give 0.4495 / 0.4512 / 0.4501 for mask thresholds 0.4 / 0.5 / 0.6, i.e. no change beyond the bootstrap noise; the final filter (0.4512) was not combined with it and the submission does not use the stage. The code and the protocol are kept so that the experiment can be repeated (see the docstring of `refine.py`). |

Compliance: only the competition's training photographs and annotations are used to fit anything (ImageNet initialisation
of the encoders aside); the only input at test time is the test H-alpha photograph itself - no metadata, no neighbouring
frames, no external or leaked labels, no annotations of the public test set.

## Repository layout

```
pipeline.ipynb       notebook that runs the whole pipeline (reproduce the submission / retrain everything)
solarfil/            the package (modules above, one responsibility each; every module has a docstring, the ones that run from the shell a command line: `python -m solarfil.<module> -h`)
configs/
  weights.json       manifest of the 10 checkpoints (URL inside this repository + SHA-256)
  rescorer_all.npz   training table of the learned instance filter + threshold + candidate settings
  postcfg.json       settings of the earlier hand-tuned version (min_area 300, min_score 0.8): the baseline of `solarfil.infer` without `--rescorer` and the source of the candidate settings of the refit commands
weights/             the 10 trained U-Nets (82 MB each, ImageNet-pretrained EfficientNet-B4 U-Net, EMA weights)
tests/               unit tests (metric, post-processing, re-scorer, evaluation, analysis helpers, stage-2 geometry)
docs/                figures
requirements.txt     loose requirements (tested with the reference environment and a newer one)
requirements-reference.txt   exact versions of the reference run (Kaggle notebook image, Python 3.13)
```

## Reproduce the submission

Open `pipeline.ipynb` (GPU) and run all cells with `RUN_MODE = "REPRODUCE_SUBMISSION"` (the default). The weights are in
`weights/`; a missing or corrupted file is fetched from the URL of `configs/weights.json` and verified against its SHA-256.
Prediction of the 180 test photographs takes about one hour on a T4 / P100; set `QUICK = True` to process only six
photographs (a few minutes) as an environment check. The notebook writes `submission.csv` and figures.

Command line equivalent:

```bash
pip install -r requirements.txt
DATA=/path/to/MAGFiLO_1.0_Kaggle_2026            # contains train/ and test/
python -m solarfil.weights --manifest configs/weights.json --dest weights
python -m solarfil.infer --data_root $DATA --ckpts weights/Unet_*.pth --rescorer configs/rescorer_all.npz \
       --save_probs work/probs --out submission.csv
python -m solarfil.infer --from_probs work/probs --rescorer configs/rescorer_all.npz --out submission.csv   # CPU: only the post-processing
```

GPU arithmetic in half precision is not bit-wise deterministic: re-running may move a few boundary pixels, not the
filaments. The learned filter is re-fitted from `configs/rescorer_all.npz` in a few seconds on every run (it is
deterministic).

## Refit the learned filter without retraining the networks

The learned instance filter is fitted on out-of-fold probability maps. They can be regenerated from the released weights
(each checkpoint predicts the photographs of its held-out fold, 8-fold TTA, about one GPU hour for the ten networks) and the
filter refitted from them:

```bash
python -m solarfil.oof --data_root $DATA --ckpts weights/Unet_*.pth --out work
python -m solarfil.rescore --data_root $DATA --postcfg configs/postcfg.json --gt all=1,2,3,4 --thr 0.20 \
       --view work/oof_Unet_efficientnet-b4_s1536 \
       --view work/oof_Unet_efficientnet-b4_s1536,work/oof_Unet_efficientnet-b4_s2048 --out work/rescorer_{name}.npz
```

(two views: the maps of the 1536^2 network alone and the average of the 1536^2 and 2048^2 networks, i.e. the 12,648
candidates stored in `configs/rescorer_all.npz`; `pipeline.ipynb` runs the same commands with `RUN_MODE = "REFIT_FILTER"`.)

## Reproduce Table 1 (post-processing ablation with confidence intervals)

```bash
python -m solarfil.oof   --data_root $DATA --ckpts weights/Unet_*.pth --out work                 # out-of-fold maps, GPU, about 1 h
python -m solarfil.cands --data_root $DATA --postcfg configs/postcfg.json --out work/cands_oof.pkl \
       --view work/oof_Unet_efficientnet-b4_s1536,work/oof_Unet_efficientnet-b4_s2048              # candidates + cross-fitted scores
python -m solarfil.ablate --data_root $DATA --oof_root work --cands work/cands_oof.pkl --tau 0.20 # the four rows, about 5 min on CPU
python -m solarfil.cands --emit work/cands_oof.pkl --thr 0.20 --out work/pred_oof.csv            # the OOF predictions of the last row
python -m solarfil.annotations --data_root $DATA --out gt_train.csv
python -m solarfil.evaluate --gt gt_train.csv --pred work/pred_oof.csv --bootstrap 2000 --plots figures   # rubric measures
```

Row 4 uses the cross-fitted scores stored in the candidate file: the score of a candidate of fold k was predicted by a
filter fitted on the candidates of the other four folds, which is why the file can be recomputed without any test data.

## Reproduce the other numbers and figures of the report

All of them start from the out-of-fold candidate file `work/cands_oof.pkl` of the previous section (CPU only).

```bash
python -m solarfil.analysis numbers    --data_root $DATA --cands work/cands_oof.pkl --out_dir report   # PQ/SQ/DQ + rubric measures with intervals, overlap structure,
                                                                                                    # PQ by number of copies, fate of the annotated filaments, per-fold PQ,
                                                                                                    # per-photograph PQ and the photographs of the example figure (~2 min)
python -m solarfil.analysis annotators --data_root $DATA                                            # last row of Table 1: annotator vs annotator (296 photographs)
python -m solarfil.analysis tau        --data_root $DATA --cands work/cands_oof.pkl                 # PQ / Dice as a function of the filter threshold
python -m solarfil.analysis hand       --data_root $DATA --cands work/cands_oof.pkl                 # the hand-tuned rule on a wider grid
python -m solarfil.analysis radius     --data_root $DATA --cands work/cands_oof.pkl                 # mask thickness: pooled and per annotator (~15 min)
python -m solarfil.analysis pairs      --data_root $DATA --cands work/cands_oof.pkl                 # matched prediction-annotation pairs (~5 min)
python -m solarfil.analysis extent     --data_root $DATA --cands work/cands_oof.pkl                 # errors of extent: small annotations vs. long ones (~1 min)
python -m solarfil.morphology --gt gt_train.csv --pred work/pred_oof.csv --out figures/morphology.png   # apparent morphology: annotations vs predictions
# figures (the stems are the photographs printed by `analysis numbers` near the 50th / 25th percentile, and three test photographs)
python -m solarfil.figures oof     --data_root $DATA --cands work/cands_oof.pkl --stems 20160526011254Uh 20151119133734Ch --out examples_oof.png
python -m solarfil.figures test    --sub submission.csv --img_dir $DATA/test/test_images --stems 20161020145454Bh 20141222201814Mh 20130110085614Th --out examples_test.png
python -m solarfil.figures summary --gt gt_train.csv --pred work/pred_oof.csv --extra "prediction (test)=submission.csv" --out summary.png
```

## Retrain everything (`RUN_MODE = "TRAIN_FULL"`, 20+ GPU hours)

```bash
for f in 0 1 2 3 4; do python -m solarfil.train --data_root $DATA --work work --fold $f --size 1536 --epochs 36; done
for f in 0 1 2 3 4; do python -m solarfil.train --data_root $DATA --work work --fold $f --size 2048 --epochs 32; done
# out-of-fold maps are written to work/oof_Unet_efficientnet-b4_s{1536,2048}; fit the learned filter on them
python -m solarfil.rescore --data_root $DATA --postcfg configs/postcfg.json --gt all=1,2,3,4 --thr 0.20 \
       --view work/oof_Unet_efficientnet-b4_s1536 \
       --view work/oof_Unet_efficientnet-b4_s1536,work/oof_Unet_efficientnet-b4_s2048 --out work/rescorer_{name}.npz
python -m solarfil.infer --data_root $DATA --ckpts work/Unet_*.pth --rescorer work/rescorer_all.npz --out submission.csv
```

`--thr 0.20` fixes the filter threshold to the PQ-optimal point of the Table 1 protocol (without it the PQ-optimal point of
the cross-validation sweep over the pooled views is taken: 0.22 in our final refit, PQ 0.4500 against 0.4491 at 0.20;
the PQ plateau spans [0.18, 0.26]).

## Evaluate any prediction file

```bash
python -m solarfil.annotations --data_root $DATA --out gt_train.csv         # ground truth in the submission format
python -m solarfil.evaluate --gt gt_train.csv --pred my_predictions.csv --bootstrap 1000 --plots figures
```

prints PQ / SQ / DQ, the semantic Dice, mean Dice and IoU of overlapping pairs, the one-to-many and many-to-one
distributions and bootstrap intervals, and writes the Dice / IoU histograms. `gt_train.csv` ids are
`<annotator>-<photo>_<k>`; every annotator copy is scored separately, exactly like the competition metric.

## Notes on the data

* 707 training photographs (2011-2022, six GONG sites), 1154 annotator copies (1-3 per photograph), 8199 filaments, each
  a single polygon; median filament area about 1200 px at 2048^2. 180 test photographs.
* Annotators disagree strongly (PQ 0.34 between two annotators), which is why the learned consensus is a better target than
  any single annotator's labels.

## Tests

```bash
python -m pytest -q tests
```

## License

MIT, see `LICENSE`. The GONG data belong to NSO/AURA and are used under the competition's terms; they are not part of this
repository.
