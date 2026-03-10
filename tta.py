"""
Test-Time Augmentation (TTA) wrapper.

Applies a set of transforms at inference and averages the predictions.
Works with all three modes: classification, regression, gaussian.

Transforms applied by default:
  1. Original
  2. Horizontal flip
  3. Center crop 90% + resize
  4. Horizontal flip + center crop 90% + resize
"""
import torch
import torch.nn.functional as F


def _hflip(x):
    return torch.flip(x, dims=[3])


def _center_crop(x, scale=0.9):
    """Crop the center region and resize back to original size."""
    _, _, h, w = x.shape
    ch, cw = int(h * scale), int(w * scale)
    top = (h - ch) // 2
    left = (w - cw) // 2
    cropped = x[:, :, top:top + ch, left:left + cw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


DEFAULT_TRANSFORMS = [
    ("original",     lambda x: x),
    ("hflip",        _hflip),
    ("center_crop",  _center_crop),
    ("hflip+crop",   lambda x: _center_crop(_hflip(x))),
]


class TTAWrapper:
    """Wraps a model and applies TTA at inference time.

    Usage:
        tta = TTAWrapper(model, mode="classification")
        probs = tta.predict(x)   # averaged over all transforms
    """

    def __init__(self, model, mode, transforms=None):
        """
        Args:
            model:      trained PyTorch model
            mode:       "classification", "regression" or "gaussian"
            transforms: list of (name, fn) pairs — defaults to DEFAULT_TRANSFORMS
        """
        assert mode in ("classification", "regression", "gaussian"), f"Unknown mode: {mode}"
        self.model = model
        self.mode = mode
        self.transforms = transforms if transforms is not None else DEFAULT_TRANSFORMS

    def predict(self, x):
        """
        Run all TTA transforms and return averaged prediction.

        Returns:
            classification -> averaged softmax probabilities  (B x 101)
            regression     -> averaged age predictions        (B,)
            gaussian       -> averaged mean predictions       (B,)
        """
        all_outputs = []

        for _, transform in self.transforms:
            out = self.model(transform(x))
            all_outputs.append(out)

        if self.mode == "classification":
            probs = [F.softmax(o, dim=-1) for o in all_outputs]
            return torch.stack(probs).mean(0)
        elif self.mode == "regression":
            preds = [o.squeeze(1) for o in all_outputs]
            return torch.stack(preds).mean(0)
        else:  # gaussian
            means = [o[:, 0] for o in all_outputs]
            return torch.stack(means).mean(0)
