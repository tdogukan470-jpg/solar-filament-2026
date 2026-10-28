"""solarfil: instance segmentation of solar filaments in GONG H-alpha images (Filament Segmentation Challenge 2026).

Pipeline (module in brackets; `python -m solarfil.<module> -h` prints the command line of the modules that have one)
--------------------------------------------------------------------------------------------------------------------
1. pre-processing : intensity, flat-field and CLAHE channels                                  (preprocess)
2. targets        : soft consensus of the annotator copies + instance-border map              (annotations)
3. stage 1        : U-Net / EfficientNet-B4 / scSE, 2 outputs (foreground, border), 5 folds
                    grouped by observation day, 1536^2 and 2048^2, soft Dice + BCE, EMA       (model, data, train)
4. inference      : whole photograph, 8 dihedral transforms, 10 networks averaged            (infer, weights)
5. candidates     : foreground > 0.5, specks removed, fragments linked, >= 100 px, disjoint   (postprocess)
6. learned filter : gradient-boosted estimate of every candidate's expected PQ gain; keep
                    candidates above the threshold (0.20 in the submission)                  (rescore)
7. evaluation     : re-implementation of the official PQ and of the rubric measures           (metric, evaluate)

Experiments behind the report: out-of-fold maps (oof), candidates with cross-fitted scores (cands), post-processing
ablation (ablate), report numbers (analysis), apparent morphology (morphology), figures (figures, visualize),
negative result: instance-conditioned second stage (refine), tuner of the hand-tuned baseline rule (tune).
"""
__version__ = "2.0.0"
