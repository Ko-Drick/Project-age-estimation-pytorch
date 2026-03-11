# Résultats

## Modèles entraînés

| Modèle | Checkpoint | Epochs | Val MAE |
|---|---|---|---|
| Classification (DEX) | `checkpoint/epoch078_0.02346_3.9613.pth` | 80 | 3.961 |
| Regression (L1) | `checkpoint/regression/epoch079_0.03157_3.9553.pth` | 80 | 3.955 |
| Gaussian NLL | `checkpoint/gaussian/epoch078_0.02517_4.0750.pth` | 78 | 4.075 |

## Test MAE (APPA-REAL test set)

| Modèle | Sans TTA | Avec TTA |
|---|---|---|
| Baseline (repo original) | 4.800 | - |
| Classification (DEX) | 4.639 | 4.459 |
| Regression (L1) | 4.647 | 4.442 |
| Gaussian NLL | 4.712 | 4.501 |

TTA : 4 transforms (original, hflip, crop 90%, hflip+crop).

## DIY complétés

- [x] **Régression vs Classification** — la classification DEX est déjà implémentée dans le repo de base. Régression L1 entraînée depuis le backbone classification (~80 epochs). Résultats quasi identiques (Δ MAE < 0.01).
- [x] **Gaussian NLL loss** — le modèle prédit un âge ET une incertitude. Incertitude moyenne sur le test set : ±5.6 ans. MAE légèrement supérieur à la régression simple.
- [x] **Test-Time Augmentation (TTA)** — gain de ~0.2 MAE sur tous les modèles sans re-training.
