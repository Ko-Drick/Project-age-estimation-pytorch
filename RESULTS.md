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

## Test MAE (APPA-REAL test set)

| Modèle | Sans TTA | Avec TTA |
|---|---|---|
| Baseline (repo original) | 4.800 | - |
| Classification (DEX) | 4.639 | 4.459 |
| Regression (L1) | 4.647 | 4.442 |
| Gaussian NLL | 4.712 | 4.501 |
| Label Smoothing (σ=2) | 4.708 | 4.561 |
| DEX + Weight Decay (1e-4) | 4.722 | TBD |
| DEX + Balanced Sampling | 4.995 | 4.772 |
| DEX + Uncertainty Weighting | 5.006 | 4.843 |
| Residual DEX | TBD | TBD |

TTA : 4 transforms (original, hflip, crop 90%, hflip+crop).

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

Le gap val/test est constant (~0.7 MAE) et ne diminue pas avec la régularisation (weight decay, label smoothing). Cela suggère un **décalage de distribution** entre les splits val et test d'APPA-REAL plutôt qu'un overfitting classique. L'optimisation du checkpoint sur le val set n'est pas un bon proxy pour la performance test.

### Cause identifiée : distribution d'âges déséquilibrée

Le test set contient significativement plus de personnes âgées et d'enfants que le val set :

| Tranche d'âge | Train | Val | Test |
|---|---|---|---|
| 0-10 | 8.9% | 8.3% | 11.2% |
| 10-20 | 10.8% | 12.5% | 7.5% |
| 20-30 | **37.1%** | **35.5%** | **29.3%** |
| 30-40 | 21.7% | 21.1% | 23.4% |
| 40-50 | 9.9% | 11.5% | 12.3% |
| 50-60 | 7.9% | 7.1% | 7.4% |
| 60-70 | 2.7% | 3.1% | **5.3%** |
| 70-80 | 0.7% | 0.7% | **2.4%** |
| 80-100 | 0.2% | 0.2% | **1.2%** |

Le modèle optimise principalement sur la tranche 20-30 ans (dominante dans train/val) et généralise moins bien sur les extrêmes (enfants, personnes âgées), qui sont plus présents dans le test set. C'est la cause principale du gap val/test, pas un overfitting classique des poids.

## DIY complétés

- [x] **Régression vs Classification** — la classification DEX est déjà implémentée dans le repo de base. Régression L1 entraînée depuis le backbone classification (~80 epochs). Résultats quasi identiques (Δ MAE < 0.01).
- [x] **Gaussian NLL loss** — le modèle prédit un âge ET une incertitude. Incertitude moyenne sur le test set : ±5.6 ans. MAE légèrement supérieur à la régression simple.
- [x] **Test-Time Augmentation (TTA)** — gain de ~0.2 MAE sur tous les modèles sans re-training.
- [x] **Label Smoothing gaussien (σ=2 ans)** — soft targets gaussiens centrés sur l'âge cible au lieu d'un one-hot. Val MAE quasi identique au DEX de base (3.960 vs 3.961). Test MAE légèrement supérieur (4.708 sans TTA, 4.561 avec TTA vs 4.459 pour DEX), probablement dû à l'initialisation depuis le checkpoint regression.
- [x] **Weight Decay (1e-4)** — meilleur val MAE (3.876 vs 3.961) mais test MAE dégradé (4.722 vs 4.639). Le gap val/test augmente (0.846 vs 0.678), confirmant que le problème n'est pas un overfitting classique mais un décalage de distribution val/test dans APPA-REAL.
- [x] **Balanced Sampling** — `WeightedRandomSampler` avec bins de 10 ans pour suréchantillonner les âges rares. Val MAE dégradé (4.117 vs 3.961) et test MAE+TTA dégradé (4.772 vs 4.459). Le rééquilibrage force le modèle à sur-représenter les âges extrêmes (peu d'exemples, haute variance) au détriment de la tranche majoritaire 20-40 ans. Le gap val/test (0.655) est néanmoins le plus bas de tous les modèles, ce qui confirme que le balanced sampling réduit le biais de distribution — mais au prix d'une MAE globale plus élevée. Trade-off classique entre équité par tranche d'âge et performance moyenne.
- [x] **Uncertainty Weighting** — pondération de la loss par `1/(human_std + 1)` pour down-weight les images ambiguës (fort désaccord humain). Val MAE légèrement meilleur (3.967 vs 3.961) mais **pire test MAE** (5.006 vs 4.639) et **plus grand gap val/test** (1.039). Explication : les images à fort std humain sont souvent les âges extrêmes, sur-représentés dans le test set. En les down-weightant, le modèle se spécialise encore plus sur les 20-40 ans et perd en généralisation. Résultat négatif mais instructif.
- [ ] **Residual DEX** — en cours d'entraînement.
