"""
Extract backbone weights from a classification checkpoint.
Useful to warm-start regression training without the classification head.

Usage:
    python extract_backbone.py --checkpoint checkpoint/classification/epochXXX_*.pth
    python extract_backbone.py --checkpoint checkpoint/classification/epochXXX_*.pth --output checkpoint/backbone.pth
"""
import argparse
import torch
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to classification checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: same dir as checkpoint)")
    return parser.parse_args()


def main():
    args = get_args()
    src = Path(args.checkpoint)

    checkpoint = torch.load(str(src), map_location="cpu")
    state_dict = checkpoint["state_dict"]

    backbone_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("last_linear")}

    output_path = Path(args.output) if args.output else src.parent / "backbone_only.pth"
    torch.save(
        {
            "epoch": checkpoint["epoch"],
            "arch": checkpoint["arch"],
            "mode": "backbone_only",
            "state_dict": backbone_state_dict,
        },
        str(output_path),
    )

    total = len(state_dict)
    kept = len(backbone_state_dict)
    print(f"=> saved backbone ({kept}/{total} layers) to '{output_path}'")


if __name__ == "__main__":
    main()
