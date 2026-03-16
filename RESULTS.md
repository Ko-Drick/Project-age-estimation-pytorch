# Résultats

## Modèles entraînés

| Modèle | Checkpoint | Epochs | Val MAE |
|---|---|---|---|
| Classification (DEX) | `checkpoint/epoch078_0.02346_3.9613.pth` | 80 | 3.961 |
| Regression (L1) | `checkpoint/regression/epoch079_0.03157_3.9553.pth` | 80 | 3.955 |
| Gaussian NLL | `checkpoint/gaussian/epoch078_0.02517_4.0750.pth` | 78 | 4.075 |
| Label Smoothing (σ=2) | `checkpoint/label_smoothing/epoch046_0.02439_3.9597.pth` | 80 | 3.960 |
| DEX + Weight Decay (1e-4) | `checkpoint/classification/epoch077_0.02320_3.8761.pth` | 80 | 3.876 |
| DEX + Balanced Sampling | `checkpoint/epoch062_0.02368_4.1170_BALANCED_SAMPLING.pth` | 80 | 4.117 |
| DEX + Uncertainty Weighting | `checkpoint/uncertainty_weighting/epoch072_0.02352_3.9670.pth` | 80 | 3.967 |
| Residual DEX | `checkpoint/epoch078_0.03407_4.2666_Residual_DEX.pth` | 78 | 4.267 |

## Test MAE (APPA-REAL test set)

| Modèle | Sans TTA | Avec TTA |
|---|---|---|
| Baseline (repo original) | 4.800 | - |
| Classification (DEX) | 4.639 | 4.459 |
| Regression (L1) | 4.647 | 4.442 |
| Gaussian NLL | 4.712 | 4.501 |
| Label Smoothing (σ=2) | 4.708 | 4.561 |
| DEX + Weight Decay (1e-4) | 4.722 | 4.600 |
| DEX + Balanced Sampling | 4.995 | 4.772 |
| DEX + Uncertainty Weighting | 5.006 | 4.843 |
| Residual DEX | 4.968 | 4.700 |
| **Ensemble (3 modèles)** | **4.428** | **4.298** |

TTA : 4 transforms (original, hflip, crop 90%, hflip+crop).
Ensemble : moyenne des prédictions Classification + Regression + Gaussian.

## Analyse de l'overfitting (gap val/test)

| Modèle | Val MAE | Test MAE | Gap |
|---|---|---|---|
| Classification (DEX) | 3.961 | 4.639 | 0.678 |
| Regression (L1) | 3.955 | 4.647 | 0.692 |
| Gaussian NLL | 4.075 | 4.712 | 0.637 |
| Label Smoothing (σ=2) | 3.960 | 4.708 | 0.748 |
| DEX + Weight Decay (1e-4) | 3.876 | 4.722 | 0.846 |
| DEX + Balanced Sampling | 4.117 | 4.772 | 0.655 |
| DEX + Uncertainty Weighting | 3.967 | 5.006 | 1.039 |
| Residual DEX | 4.267 | 4.968 | 0.701 |
