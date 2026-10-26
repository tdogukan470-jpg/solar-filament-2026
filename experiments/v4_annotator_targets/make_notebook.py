"""Build a self-contained Kaggle notebook that embeds the solarfil package."""
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


cells = [md("""
# Solar Filament Segmentation 2026 — solarfil v2 (full pipeline)

**Settings (right panel):** Accelerator = GPU (T4 x2 or P100), Internet = ON, Input = competition data.
To continue from an earlier run, also add that run's *output* as input (checkpoints and OOF files are found automatically).

Pipeline: limb-darkening flattening → U-Net (EfficientNet-B4) at 1536² on 768² crops with
multi-annotator soft targets + an instance-border channel → 8x dihedral TTA + fold ensemble →
fragment linking / border splitting / filtering tuned with the **exact official PQ metric** on out-of-fold predictions.
"""),
         code("""
# ======================= SETTINGS =======================
FOLDS      = [0, 1]        # folds to train in THIS session (~1.5-2 h each on T4). 5 folds total: 0..4
EPOCHS     = 36
ENCODER    = "efficientnet-b4"
SIZE       = 1536
CROP       = 768
BATCH      = 6
RUN_TRAIN  = True
RUN_TUNE   = False         # needs out-of-fold files of ALL 5 folds (current + earlier outputs)
RUN_INFER  = False         # writes /kaggle/working/submission.csv using every checkpoint found
TUNE_PHOTOS = 300          # OOF photos used for post-processing tuning (CPU time ~ linear)
POSTCFG    = None          # dict -> written to postcfg.json (skips tuning), e.g. from an earlier tune
INFER_TTA  = 8
USE_RESCORER = False
TUNE_ROUNDS = 1
TARGET_MODE = "soft"      # "soft" = annotator average (v2); "annotator" = one annotator copy per sample + small-object weights
# ========================================================
"""),
         code("!pip install -q segmentation-models-pytorch==0.5.0 pycocotools 2>&1 | tail -1")]

for path in sorted(glob.glob(os.path.join(ROOT, "solarfil", "*.py"))):
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    src = open(path, encoding="utf-8").read()
    cells.append(code(f"%%writefile {rel}\n" + src.replace("\r", "")))

# make sure the package dir exists before the first %%writefile
cells.insert(3, code("import os; os.makedirs('solarfil', exist_ok=True)"))

cells.append(code("""
import glob, os, shutil, subprocess, sys, json
DATA = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)[0]))
WORK = '/kaggle/working'
print('data root:', DATA)
# bring in checkpoints / OOF from earlier sessions (attached as inputs)
for p in glob.glob('/kaggle/input/**/*.pth', recursive=True):
    if not os.path.exists(os.path.join(WORK, os.path.basename(p))):
        shutil.copy(p, WORK); print('reuse', os.path.basename(p))
for pat in ['rescorer*.pkl', 'postcfg*.json']:
    for p in glob.glob('/kaggle/input/**/' + pat, recursive=True):
        shutil.copy(p, WORK); print('reuse', os.path.basename(p))
for d in glob.glob('/kaggle/input/**/oof_*', recursive=True):
    if os.path.isdir(d):
        dst = os.path.join(WORK, os.path.basename(d)); os.makedirs(dst, exist_ok=True)
        for f in glob.glob(d + '/*.npz'):
            if not os.path.exists(os.path.join(dst, os.path.basename(f))): shutil.copy(f, dst)
def run(cmd):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    assert r.returncode == 0
"""))
cells.append(code("""
if RUN_TRAIN:
    for f in FOLDS:
        run(f"{sys.executable} -m solarfil.train --data_root {DATA} --work {WORK} --fold {f} --epochs {EPOCHS} "
            f"--encoder {ENCODER} --size {SIZE} --crop {CROP} --bs {BATCH} --workers 4 --cache_root /tmp --target_mode {TARGET_MODE}")
"""))
cells.append(code("""
OOF = f"{WORK}/oof_Unet_{ENCODER}_s{SIZE}"
if RUN_TUNE:
    print(len(glob.glob(OOF + '/*.npz')), 'OOF photos available')
    init = ''
    if POSTCFG:
        json.dump({"cfg": POSTCFG}, open(f"{WORK}/postcfg_init.json", "w")); init = f"--init {WORK}/postcfg_init.json"
    run(f"{sys.executable} -m solarfil.tune --data_root {DATA} --oof {OOF} --max_photos {TUNE_PHOTOS} --workers 4 "
        f"--rounds {TUNE_ROUNDS} {init} --out {WORK}/postcfg.json")
"""))
cells.append(code("""
if POSTCFG and not RUN_TUNE:
    json.dump({"cfg": POSTCFG}, open(f"{WORK}/postcfg.json", "w"))
if RUN_INFER:
    ck = ' '.join(sorted(glob.glob(f'{WORK}/*_f[0-9].pth')))
    pc = f'--postcfg {WORK}/postcfg.json' if os.path.exists(f'{WORK}/postcfg.json') else ''
    if USE_RESCORER and os.path.exists(f'{WORK}/rescorer.pkl'):
        pc += f' --rescorer {WORK}/rescorer.pkl'
    run(f"{sys.executable} -m solarfil.infer --data_root {DATA} --ckpts {ck} {pc} --work {WORK} --tta {INFER_TTA} --out {WORK}/submission.csv")
# keep the output small: drop caches
for d in glob.glob(f'{WORK}/cache_*') + glob.glob(f'{WORK}/test_cache_*'):
    shutil.rmtree(d, ignore_errors=True)
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "solar_filament_v2.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out, len(cells), "cells")
