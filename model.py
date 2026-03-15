import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils


def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def get_regression_model(model_name="se_resnext50_32x4d", pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, 1)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def get_gaussian_model(model_name="se_resnext50_32x4d", pretrained="imagenet"):
    """Outputs 2 values per sample: predicted age (mean) + log variance."""
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, 2)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


class ResidualDEXModel(nn.Module):
    """Residual DEX: predicts class probabilities + per-class residuals.

    Age = Σ softmax(logits)[c] * (c + tanh(residuals[c]) * 0.5)

    The residuals allow the model to correct each bin by up to ±0.5 years,
    giving continuous sub-integer predictions while keeping the soft-classification
    structure of DEX.
    """
    def __init__(self, backbone, num_classes=101):
        super().__init__()
        dim_feats = backbone.last_linear.in_features
        backbone.last_linear = nn.Identity()
        self.backbone = backbone
        self.class_head = nn.Linear(dim_feats, num_classes)
        self.residual_head = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.class_head(feats)
        residuals = torch.tanh(self.residual_head(feats)) * 0.5  # in (-0.5, 0.5)
        return logits, residuals


def get_residual_dex_model(model_name="se_resnext50_32x4d", pretrained="imagenet"):
    backbone = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    backbone.avg_pool = nn.AdaptiveAvgPool2d(1)
    return ResidualDEXModel(backbone)


def main():
    model = get_model()
    print(model)


if __name__ == '__main__':
    main()
