import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model, get_regression_model, get_gaussian_model, get_residual_dex_model
from dataset import FaceDataset
from defaults import _C as cfg


class GaussianLabelSmoothingLoss(nn.Module):
    """Cross-entropy with Gaussian soft targets centered on the true age class.

    Instead of a one-hot target, each sample gets a Gaussian distribution over
    the 101 age classes (σ = label_smoothing in years), which respects the
    ordinal structure of age labels.
    """
    def __init__(self, num_classes=101, sigma=2.0):
        super().__init__()
        self.sigma = sigma
        ages = torch.arange(num_classes).float()
        self.register_buffer('ages', ages)

    def forward(self, logits, targets):
        ages = self.ages.unsqueeze(0)                      # (1, 101)
        mu = targets.float().unsqueeze(1)                  # (B, 1)
        soft_targets = torch.exp(-0.5 * ((ages - mu) / self.sigma) ** 2)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train_classification(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            cur_loss = loss.item()

            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()

            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, accuracy_monitor.avg


def validate_classification(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())

                if criterion is not None:
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def train_regression(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device).float()

            outputs = model(x).squeeze(1)
            loss = criterion(outputs, y)
            cur_loss = loss.item()

            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              sample_num=sample_num)

    return loss_monitor.avg


def validate_regression(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device).float()

                outputs = model(x).squeeze(1)
                preds.append(outputs.cpu().numpy())
                gt.append(y.cpu().numpy())

                if criterion is not None:
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    mae = np.abs(preds - gt).mean()

    return loss_monitor.avg, mae


def gaussian_nll_loss(outputs, targets):
    """Negative log-likelihood of a Gaussian: penalizes both error and overconfidence."""
    mean = outputs[:, 0]
    log_var = outputs[:, 1]
    loss = 0.5 * torch.exp(-log_var) * (mean - targets) ** 2 + 0.5 * log_var
    return loss.mean()


def train_gaussian(train_loader, model, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device).float()

            outputs = model(x)
            loss = gaussian_nll_loss(outputs, y)
            cur_loss = loss.item()

            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              sample_num=sample_num)

    return loss_monitor.avg


def validate_gaussian(validate_loader, model, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    means = []
    stds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device).float()

                outputs = model(x)
                means.append(outputs[:, 0].cpu().numpy())
                stds.append(torch.exp(0.5 * outputs[:, 1]).cpu().numpy())  # std = exp(0.5 * log_var)
                gt.append(y.cpu().numpy())

                loss = gaussian_nll_loss(outputs, y)
                cur_loss = loss.item()

                sample_num = x.size(0)
                loss_monitor.update(cur_loss, sample_num)
                _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                  sample_num=sample_num)

    means = np.concatenate(means, axis=0)
    stds = np.concatenate(stds, axis=0)
    gt = np.concatenate(gt, axis=0)
    mae = np.abs(means - gt).mean()
    avg_std = stds.mean()

    return loss_monitor.avg, mae, avg_std


def train_residual_dex(train_loader, model, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    ages_tensor = torch.arange(0, 101, dtype=torch.float32).to(device)

    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device).float()

            logits, residuals = model(x)
            probs = F.softmax(logits, dim=1)
            predicted_age = (probs * (ages_tensor + residuals)).sum(dim=1)
            loss = F.l1_loss(predicted_age, y)

            loss_monitor.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              sample_num=x.size(0))

    return loss_monitor.avg


def validate_residual_dex(validate_loader, model, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    preds = []
    gt = []
    ages_tensor = torch.arange(0, 101, dtype=torch.float32).to(device)

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device).float()

                logits, residuals = model(x)
                probs = F.softmax(logits, dim=1)
                predicted_age = (probs * (ages_tensor + residuals)).sum(dim=1)

                preds.append(predicted_age.cpu().numpy())
                gt.append(y.cpu().numpy())

                loss = F.l1_loss(predicted_age, y)
                loss_monitor.update(loss.item(), x.size(0))
                _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                  sample_num=x.size(0))

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    mae = np.abs(preds - gt).mean()
    return loss_monitor.avg, mae


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mode = cfg.MODEL.MODE
    assert mode in ("classification", "regression", "gaussian", "residual_dex"), f"Unknown mode: {mode}"
    print("=> mode: {}".format(mode))

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    if mode == "classification":
        model = get_model(model_name=cfg.MODEL.ARCH)
    elif mode == "regression":
        model = get_regression_model(model_name=cfg.MODEL.ARCH)
    elif mode == "gaussian":
        model = get_gaussian_model(model_name=cfg.MODEL.ARCH)
    else:
        model = get_residual_dex_model(model_name=cfg.MODEL.ARCH)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            model_sd = model.state_dict()
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k in model_sd and v.shape == model_sd[k].shape:
                    state_dict[k] = v
                elif f"backbone.{k}" in model_sd and v.shape == model_sd[f"backbone.{k}"].shape:
                    state_dict[f"backbone.{k}"] = v
            model.load_state_dict(state_dict, strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            if checkpoint.get('mode', mode) == mode:
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    # datasets — labels differ between modes
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV, mode=mode)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False, mode=mode)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    # criterion (gaussian and residual_dex use their own loss, no criterion object needed)
    if mode == "classification":
        if cfg.TRAIN.LABEL_SMOOTHING > 0:
            print(f"=> using Gaussian label smoothing (sigma={cfg.TRAIN.LABEL_SMOOTHING})")
            criterion = GaussianLabelSmoothingLoss(num_classes=101, sigma=cfg.TRAIN.LABEL_SMOOTHING).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
    elif mode == "regression":
        criterion = nn.L1Loss().to(device)
    else:
        criterion = None

    # checkpoints saved in a subfolder named after the mode
    checkpoint_dir = checkpoint_dir / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + f"/{mode}/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + f"/{mode}/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        if mode == "classification":
            train_loss, train_acc = train_classification(train_loader, model, criterion, optimizer, epoch, device)
            val_loss, val_acc, val_mae = validate_classification(val_loader, model, criterion, epoch, device)
        elif mode == "regression":
            train_loss = train_regression(train_loader, model, criterion, optimizer, epoch, device)
            val_loss, val_mae = validate_regression(val_loader, model, criterion, epoch, device)
        elif mode == "gaussian":
            train_loss = train_gaussian(train_loader, model, optimizer, epoch, device)
            val_loss, val_mae, val_std = validate_gaussian(val_loader, model, epoch, device)
            print(f"=> [epoch {epoch:03d}] avg predicted std: {val_std:.3f}")
        else:  # residual_dex
            train_loss = train_residual_dex(train_loader, model, optimizer, epoch, device)
            val_loss, val_mae = validate_residual_dex(val_loader, model, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)
            if mode == "classification":
                train_writer.add_scalar("acc", train_acc, epoch)
                val_writer.add_scalar("acc", val_acc, epoch)
            elif mode == "gaussian":
                val_writer.add_scalar("avg_std", val_std, epoch)

        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'mode': mode,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    main()
