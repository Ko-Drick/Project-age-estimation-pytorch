"""
Compare classification, regression and gaussian models on the test set.

Usage:
    python compare.py \
        --data_dir appa-real-release \
        --classif_checkpoint checkpoint/classification/epochXXX_*.pth \
        --regress_checkpoint checkpoint/regression/epochXXX_*.pth \
        --gaussian_checkpoint checkpoint/gaussian/epochXXX_*.pth  # optional
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


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--classif_checkpoint", type=str, required=True)
    parser.add_argument("--regress_checkpoint", type=str, required=True)
    parser.add_argument("--gaussian_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def evaluate_classification(model, loader, device):
    model.eval()
    preds, gt = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Classification"):
            x = x.to(device)
            outputs = model(x)
            preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
            gt.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    mae = np.abs(ave_preds - gt).mean()
    return mae


def evaluate_regression(model, loader, device):
    model.eval()
    preds, gt = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Regression"):
            x = x.to(device)
            outputs = model(x).squeeze(1)
            preds.append(outputs.cpu().numpy())
            gt.append(y.numpy())
    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    mae = np.abs(preds - gt).mean()
    return mae


def evaluate_gaussian(model, loader, device):
    model.eval()
    means, stds, gt = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Gaussian"):
            x = x.to(device)
            outputs = model(x)
            means.append(outputs[:, 0].cpu().numpy())
            stds.append(torch.exp(0.5 * outputs[:, 1]).cpu().numpy())
            gt.append(y.numpy())
    means = np.concatenate(means, axis=0)
    stds = np.concatenate(stds, axis=0)
    gt = np.concatenate(gt, axis=0)
    mae = np.abs(means - gt).mean()
    avg_std = stds.mean()
    return mae, avg_std


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Classification ---
    classif_ckpt = torch.load(args.classif_checkpoint, map_location="cpu")
    arch = classif_ckpt.get("arch", cfg.MODEL.ARCH)

    classif_model = get_model(model_name=arch, pretrained=None)
    classif_model.load_state_dict(classif_ckpt["state_dict"])
    classif_model = classif_model.to(device)

    classif_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE,
                                  augment=False, mode="classification")
    classif_loader = DataLoader(classif_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
    classif_mae = evaluate_classification(classif_model, classif_loader, device)

    # --- Regression ---
    regress_ckpt = torch.load(args.regress_checkpoint, map_location="cpu")

    regress_model = get_regression_model(model_name=arch, pretrained=None)
    regress_model.load_state_dict(regress_ckpt["state_dict"])
    regress_model = regress_model.to(device)

    regress_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE,
                                  augment=False, mode="regression")
    regress_loader = DataLoader(regress_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
    regress_mae = evaluate_regression(regress_model, regress_loader, device)

    # --- Gaussian (optional) ---
    gaussian_mae = None
    gaussian_std = None
    if args.gaussian_checkpoint:
        gaussian_ckpt = torch.load(args.gaussian_checkpoint, map_location="cpu")

        gaussian_model = get_gaussian_model(model_name=arch, pretrained=None)
        gaussian_model.load_state_dict(gaussian_ckpt["state_dict"])
        gaussian_model = gaussian_model.to(device)

        gaussian_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE,
                                       augment=False, mode="gaussian")
        gaussian_loader = DataLoader(gaussian_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)
        gaussian_mae, gaussian_std = evaluate_gaussian(gaussian_model, gaussian_loader, device)

    # --- Results ---
    print("\n" + "=" * 50)
    print(f"{'Model':<20} {'Test MAE':>10} {'Avg Std':>10}")
    print("-" * 50)
    print(f"{'Classification':<20} {classif_mae:>10.4f} {'N/A':>10}")
    print(f"{'Regression':<20} {regress_mae:>10.4f} {'N/A':>10}")
    if gaussian_mae is not None:
        print(f"{'Gaussian NLL':<20} {gaussian_mae:>10.4f} {gaussian_std:>10.4f}")
    print("=" * 50)

    results = {"Classification": classif_mae, "Regression": regress_mae}
    if gaussian_mae is not None:
        results["Gaussian NLL"] = gaussian_mae
    winner = min(results, key=results.get)
    best = results[winner]
    second = sorted(results.values())[1]
    print(f"=> Best model: {winner} (Δ MAE = {abs(best - second):.4f})")


if __name__ == "__main__":
    main()
