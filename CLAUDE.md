# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
source .venv/bin/activate       # Python 3.11.14, managed via uv
uv pip install -r requirements.txt
```

## Common Commands

### Training
```bash
python train.py --data_dir appa-real-release --tensorboard tf_log
# Override config via CLI (YACS style):
python train.py --data_dir appa-real-release --tensorboard tf_log MODEL.ARCH se_resnet50 TRAIN.OPT sgd TRAIN.LR 0.1
# Monitor training:
tensorboard --logdir=tf_log
```

### Evaluation
```bash
python test.py --data_dir appa-real-release --resume checkpoint/epoch{N}_{loss}_{mae}.pth
```

### Inference / Demo
```bash
python demo.py                                              # webcam
python demo.py --img_dir /path/to/images                   # image directory
python demo.py --img_dir /path/to/images --output_dir out/ # save results
```

## Architecture Overview

This project frames **age estimation as 101-class classification** (ages 0–100). At inference time, a weighted average over the softmax probabilities produces a continuous age prediction.

### Data flow
```
APPA-REAL CSV + images → FaceDataset (dataset.py) → DataLoader
    → Model forward pass → CrossEntropyLoss
    → Validation: softmax → weighted sum → MAE
    → Checkpoint saved to checkpoint/
```

### Key modules

| File | Role |
|------|------|
| `model.py` | Loads pretrained model (default: `se_resnext50_32x4d`) via `pretrainedmodels`; replaces final linear layer with 101-class head |
| `dataset.py` | `FaceDataset` — reads APPA-REAL CSVs, applies `imgaug` augmentation pipeline, normalizes images |
| `train.py` | Training loop with validation, LR scheduling (step decay), checkpoint saving (best MAE) |
| `test.py` | Runs `validate()` on the test split; reuses the same function from train.py |
| `demo.py` | Loads a checkpoint, uses **dlib** face detector, predicts age on webcam or image files |
| `defaults.py` | YACS config with all hyperparameters — override any key on the CLI without editing the file |

### Configuration (defaults.py)

All hyperparameters live in a YACS config. Key defaults:

```
MODEL.ARCH          se_resnext50_32x4d
MODEL.IMG_SIZE      224
TRAIN.OPT           adam
TRAIN.LR            0.001
TRAIN.LR_DECAY_STEP 20
TRAIN.LR_DECAY_RATE 0.2
TRAIN.BATCH_SIZE    128
TRAIN.EPOCHS        80
TRAIN.AGE_STDDEV    1.0   # label noise during training
```

### Dataset (APPA-REAL)

- Located in `appa-real-release/` (train/valid/test subdirs + CSV files)
- CSVs contain: `file_name`, `apparent_age_avg`, `apparent_age_std`, `real_age`
- `ignore_list.csv` lists 121 images excluded from training
- ~7,591 total face-cropped JPEG images with crowdsourced age annotations

### Model output

Checkpoints are saved to `checkpoint/` as `epoch{N}_{loss}_{mae}.pth`. The best checkpoint (lowest validation MAE) is tracked separately.

## Project Ideas (from README)

The README documents many research extension directions: regression vs. classification comparison, DEX/Residual DEX, label smoothing, Gaussian/Laplace loss, imbalanced data handling, multi-task learning, domain adaptation, MC-Dropout uncertainty, GAN augmentation, and TTA ensembling.
