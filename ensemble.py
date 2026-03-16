"""
Ensemble predictions from multiple models (classification, regression, gaussian).

Usage:
    python ensemble.py --data_dir appa-real-release --tta
    python ensemble.py --data_dir appa-real-release  # without TTA
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model, get_regression_model, get_gaussian_model
from dataset import FaceDataset
from defaults import _C as cfg
from tta import TTAWrapper


DEFAULT_CHECKPOINTS = {
    "classification": "checkpoint/epoch078_0.02346_3.9613.pth",
    "regression": "checkpoint/regression/epoch079_0.03157_3.9553.pth",
    "gaussian": "checkpoint/gaussian/epoch078_0.02517_4.0750.pth",
}


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classif_checkpoint", type=str, default=DEFAULT_CHECKPOINTS["classification"])
    parser.add_argument("--regress_checkpoint", type=str, default=DEFAULT_CHECKPOINTS["regression"])
    parser.add_argument("--gaussian_checkpoint", type=str, default=DEFAULT_CHECKPOINTS["gaussian"])
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def get_all_predictions(predict_fn, loader, device, label="Model"):
    """Returns per-sample age predictions as a numpy array."""
    preds = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=label):
            x = x.to(device)
            preds.append(predict_fn(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def get_ground_truth(loader):
    """Returns ground truth ages from loader."""
    gt = []
    for _, y in loader:
        gt.append(y.numpy())
    return np.concatenate(gt, axis=0)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tta_suffix = " + TTA" if args.tta else ""
    ages = torch.arange(0, 101, dtype=torch.float32).to(device)

    # --- Load models ---
    arch = cfg.MODEL.ARCH

    # Classification
    print("=> Loading classification model")
    classif_ckpt = torch.load(args.classif_checkpoint, map_location="cpu")
    arch = classif_ckpt.get("arch", arch)
    classif_model = get_model(model_name=arch, pretrained=None)
    classif_model.load_state_dict(classif_ckpt["state_dict"])
    classif_model = classif_model.to(device).eval()

    if args.tta:
        tta_classif = TTAWrapper(classif_model, mode="classification")
        classif_fn = lambda x: (tta_classif.predict(x) * ages).sum(dim=-1)
    else:
        classif_fn = lambda x: (F.softmax(classif_model(x), dim=-1) * ages).sum(dim=-1)

    # Regression
    print("=> Loading regression model")
    regress_ckpt = torch.load(args.regress_checkpoint, map_location="cpu")
    regress_model = get_regression_model(model_name=arch, pretrained=None)
    regress_model.load_state_dict(regress_ckpt["state_dict"])
    regress_model = regress_model.to(device).eval()

    if args.tta:
        tta_regress = TTAWrapper(regress_model, mode="regression")
        regress_fn = lambda x: tta_regress.predict(x)
    else:
        regress_fn = lambda x: regress_model(x).squeeze(1)

    # Gaussian
    print("=> Loading gaussian model")
    gaussian_ckpt = torch.load(args.gaussian_checkpoint, map_location="cpu")
    gaussian_model = get_gaussian_model(model_name=arch, pretrained=None)
    gaussian_model.load_state_dict(gaussian_ckpt["state_dict"])
    gaussian_model = gaussian_model.to(device).eval()

    if args.tta:
        tta_gaussian = TTAWrapper(gaussian_model, mode="gaussian")
        gaussian_fn = lambda x: tta_gaussian.predict(x)
    else:
        gaussian_fn = lambda x: gaussian_model(x)[:, 0]

    # --- Dataset (use classification mode for labels — apparent_age_avg rounded) ---
    dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False, mode="regression")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # --- Get predictions from each model ---
    print(f"\n=> Running ensemble{tta_suffix}")
    classif_preds = get_all_predictions(classif_fn, loader, device, f"Classification{tta_suffix}")
    regress_preds = get_all_predictions(regress_fn, loader, device, f"Regression{tta_suffix}")
    gaussian_preds = get_all_predictions(gaussian_fn, loader, device, f"Gaussian{tta_suffix}")

    gt = get_ground_truth(loader)

    # --- Individual MAEs ---
    classif_mae = np.abs(classif_preds - gt).mean()
    regress_mae = np.abs(regress_preds - gt).mean()
    gaussian_mae = np.abs(gaussian_preds - gt).mean()

    # --- Ensemble: simple average ---
    ensemble_preds = (classif_preds + regress_preds + gaussian_preds) / 3.0
    ensemble_mae = np.abs(ensemble_preds - gt).mean()

    # --- Results ---
    print("\n" + "=" * 50)
    print(f"{'Model':<30} {'Test MAE':>10}")
    print("-" * 50)
    print(f"{'Classification' + tta_suffix:<30} {classif_mae:>10.4f}")
    print(f"{'Regression' + tta_suffix:<30} {regress_mae:>10.4f}")
    print(f"{'Gaussian' + tta_suffix:<30} {gaussian_mae:>10.4f}")
    print("-" * 50)
    print(f"{'ENSEMBLE (avg)' + tta_suffix:<30} {ensemble_mae:>10.4f}")
    print("=" * 50)

    best_individual = min(classif_mae, regress_mae, gaussian_mae)
    gain = best_individual - ensemble_mae
    print(f"=> Gain vs best individual: {gain:.4f} MAE")


if __name__ == "__main__":
    main()
