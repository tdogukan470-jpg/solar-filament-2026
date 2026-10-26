"""solarfil: instance segmentation of solar filaments in GONG H-alpha images.

Pipeline
--------
1. preprocess  : limb-darkening flattening + contrast channels (preprocess.py)
2. targets     : multi-annotator soft foreground + instance-border maps (annotations.py)
3. model       : U-Net (segmentation_models_pytorch) with 2 outputs: foreground, border (model.py)
4. training    : high-resolution crops, Dice+BCE, EMA, GroupKFold by photo (train.py)
5. inference   : full-disk prediction, dihedral TTA, fold ensemble (inference.py)
6. instances   : threshold -> fragment linking -> border splitting -> hole filling (postprocess.py)
7. evaluation  : exact re-implementation of the official Panoptic Quality metric (metric.py)
"""
__version__ = "2.0.0"
