"""
Compare classification, regression and gaussian models on the test set.

Usage:
    python compare.py \
        --data_dir appa-real-release \
        --classif_checkpoint checkpoint/epoch078_0.02346_3.9613.pth \
        --regress_checkpoint checkpoint/regression/epochXXX_*.pth \
        --gaussian_checkpoint checkpoint/gaussian/epochXXX_*.pth  # optional
        --tta  # optional: enable Test-Time Augmentation
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model, get_regression_model, get_gaussian_model, get_residual_dex_model
from dataset import FaceDataset
from defaults import _C as cfg
from tta import TTAWrapper


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classif_checkpoint", type=str, default=None)
    parser.add_argument("--regress_checkpoint", type=str, default=None)
    parser.add_argument("--gaussian_checkpoint", type=str, default=None)
    parser.add_argument("--label_smoothing_checkpoint", type=str, default=None)
    parser.add_argument("--residual_checkpoint", type=str, default=None)
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def evaluate_classification(predict_fn, loader, device, label="Classification"):
    """predict_fn(x) -> softmax probs (B x 101)"""
    preds, gt = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=label):
            x = x.to(device)
            probs = predict_fn(x)
            preds.append(probs.cpu().numpy())
            gt.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    return np.abs(ave_preds - gt).mean()


def evaluate_regression(predict_fn, loader, device, label="Regression"):
    """predict_fn(x) -> predicted ages (B,)"""
    preds, gt = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=label):
            x = x.to(device)
            preds.append(predict_fn(x).cpu().numpy())
            gt.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    return np.abs(preds - gt).mean()


def evaluate_gaussian(predict_fn, loader, device, label="Gaussian"):
    """predict_fn(x) -> predicted means (B,)"""
    preds, gt = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=label):
            x = x.to(device)
            preds.append(predict_fn(x).cpu().numpy())
            gt.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    return np.abs(preds - gt).mean()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tta_suffix = " + TTA" if args.tta else ""

    arch = cfg.MODEL.ARCH

    # --- Classification (optional) ---
    classif_mae = None
    if args.classif_checkpoint:
        classif_ckpt = torch.load(args.classif_checkpoint, map_location="cpu")
        arch = classif_ckpt.get("arch", cfg.MODEL.ARCH)

        classif_model = get_model(model_name=arch, pretrained=None)
        classif_model.load_state_dict(classif_ckpt["state_dict"])
        classif_model = classif_model.to(device).eval()

        if args.tta:
            tta_classif = TTAWrapper(classif_model, mode="classification")
            classif_predict = lambda x: tta_classif.predict(x)
        else:
            classif_predict = lambda x: F.softmax(classif_model(x), dim=-1)

        classif_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE,
                                      augment=False, mode="classification")
        classif_loader = DataLoader(classif_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)
        classif_mae = evaluate_classification(classif_predict, classif_loader, device,
                                              label=f"Classification{tta_suffix}")

    # --- Regression (optional) ---
    regress_mae = None
    if args.regress_checkpoint:
        regress_ckpt = torch.load(args.regress_checkpoint, map_location="cpu")

        regress_model = get_regression_model(model_name=arch, pretrained=None)
        regress_model.load_state_dict(regress_ckpt["state_dict"])
        regress_model = regress_model.to(device).eval()

        if args.tta:
            tta_regress = TTAWrapper(regress_model, mode="regression")
            regress_predict = lambda x: tta_regress.predict(x)
        else:
            regress_predict = lambda x: regress_model(x).squeeze(1)

        regress_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE,
                                      augment=False, mode="regression")
        regress_loader = DataLoader(regress_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)
        regress_mae = evaluate_regression(regress_predict, regress_loader, device,
                                          label=f"Regression{tta_suffix}")

    # --- Gaussian (optional) ---
    gaussian_mae = None
    if args.gaussian_checkpoint:
        gaussian_ckpt = torch.load(args.gaussian_checkpoint, map_location="cpu")

        gaussian_model = get_gaussian_model(model_name=arch, pretrained=None)
        gaussian_model.load_state_dict(gaussian_ckpt["state_dict"])
        gaussian_model = gaussian_model.to(device).eval()

        if args.tta:
            tta_gaussian = TTAWrapper(gaussian_model, mode="gaussian")
            gaussian_predict = lambda x: tta_gaussian.predict(x)
        else:
            gaussian_predict = lambda x: gaussian_model(x)[:, 0]

        gaussian_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE,
                                       augment=False, mode="gaussian")
        gaussian_loader = DataLoader(gaussian_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)
        gaussian_mae = evaluate_gaussian(gaussian_predict, gaussian_loader, device,
                                         label=f"Gaussian{tta_suffix}")

    # --- Label Smoothing (optional) ---
    label_smoothing_mae = None
    if args.label_smoothing_checkpoint:
        ls_ckpt = torch.load(args.label_smoothing_checkpoint, map_location="cpu")

        ls_model = get_model(model_name=arch, pretrained=None)
        ls_model.load_state_dict(ls_ckpt["state_dict"])
        ls_model = ls_model.to(device).eval()

        if args.tta:
            tta_ls = TTAWrapper(ls_model, mode="classification")
            ls_predict = lambda x: tta_ls.predict(x)
        else:
            ls_predict = lambda x: F.softmax(ls_model(x), dim=-1)

        ls_loader = DataLoader(
            FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False, mode="classification"),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        label_smoothing_mae = evaluate_classification(ls_predict, ls_loader, device,
                                                      label=f"Label Smoothing{tta_suffix}")

    # --- Residual DEX (optional) ---
    residual_mae = None
    if args.residual_checkpoint:
        residual_ckpt = torch.load(args.residual_checkpoint, map_location="cpu")

        residual_model = get_residual_dex_model(model_name=arch, pretrained=None)
        residual_model.load_state_dict(residual_ckpt["state_dict"])
        residual_model = residual_model.to(device).eval()

        if args.tta:
            tta_residual = TTAWrapper(residual_model, mode="residual_dex")
            residual_predict = lambda x: tta_residual.predict(x)
        else:
            ages = torch.arange(0, 101, dtype=torch.float32).to(device)
            residual_predict = lambda x: (lambda lg, res: (torch.softmax(lg, dim=1) * (ages + res)).sum(dim=1))(*residual_model(x))

        residual_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE,
                                       augment=False, mode="residual_dex")
        residual_loader = DataLoader(residual_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)
        residual_mae = evaluate_regression(residual_predict, residual_loader, device,
                                           label=f"Residual DEX{tta_suffix}")

    # --- Results ---
    results = {}
    if classif_mae is not None:
        results["Classification"] = classif_mae
    if regress_mae is not None:
        results["Regression"] = regress_mae
    if gaussian_mae is not None:
        results["Gaussian NLL"] = gaussian_mae
    if label_smoothing_mae is not None:
        results["Label Smoothing"] = label_smoothing_mae
    if residual_mae is not None:
        results["Residual DEX"] = residual_mae

    print("\n" + "=" * 45)
    print(f"{'Model':<25} {'Test MAE':>10}  TTA")
    print("-" * 45)
    for name, mae in results.items():
        print(f"{name:<25} {mae:>10.4f}  {'yes' if args.tta else 'no'}")
    print("=" * 45)

    if len(results) >= 2:
        winner = min(results, key=results.get)
        best = results[winner]
        second = sorted(results.values())[1]
        print(f"=> Best model: {winner} (Δ MAE = {abs(best - second):.4f})")
    elif len(results) == 1:
        name, mae = next(iter(results.items()))
        print(f"=> {name}: MAE = {mae:.4f}")


if __name__ == "__main__":
    main()
