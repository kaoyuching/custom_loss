from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn


__all__ = ["DiceLoss", "DiceBCELoss", "DiceCELoss"]


class DiceLoss(nn.Module):
    r"""
    Compute dice loss
    """
    def __init__(logit: bool = True, task: str = "binary", smooth: float = 1.):
        super(DiceLoss, self).__init__()
        self.logit = logit
        if task.lower() not in ["binary", "multilabel", "multiclass"]:
            raise ValueError("task should be one of 'binary', 'multilabel', or 'multiclass'")
        self.task = task.lower()
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor, class_weight: Optional[Union[list, Tensor]] = None):
        device = inputs.device
        b, c, *_ = inputs.size()
        if class_weight is None:
            class_weight = [1]*c
        class_weight = torch.tensor(class_weight, device=device)

        if self.logit and self.task in ["binary", "multilabel"]:
            inputs = torch.sigmoid(inputs)
        if self.logit and self.task == "multiclass":
            inputs = inputs.argmax(dim=1)
        _inputs = inputs.view(b, c, -1)
        _targets = targets.view(b, c, -1)

        intersection = torch.sum(_inputs * _targets, dim=2)
        denominator = torch.sum(_inputs + _targets, dim=2)
        _dice_score = (2. * intersection + self.smooth)/(denominator + self.smooth)  # shape: (B, C)
        dice_score = torch.mean(_dice_score, dim=0)
        dice_loss = 1 - torch.mean(dice_score * class_weight)
        return dice_loss


class DiceBCELoss(nn.Module):
    r"""
    Compute dice + bce loss for binary and multilabel tasks.
    """
    def __init__(
        self,
        class_weight: Optional[Union[list, Tensor]] = None,
        weight: Optional[Union[list, Tensor]] = None,
        logit: bool = True,
        reduction: str = "mean",
        task: str = "binary",
        smooth: float = 1.,
    ):
        super(DiceBCELoss, self).__init__()
        self.class_weight = class_weight
        self.bce_weight = weight
        self.logit = logit
        self.reduction = reduction
        self.dice_loss_fn = DiceLoss(logit=False, task=task.lower(), smooth=smooth)

    def forward(self, inputs: Tensor, targets: Tensor):
        device = inputs.device
        b, c, *_ = inputs.size()

        if self.class_weight is None:
            self.class_weight = [1]*c
        self.class_weight = torch.tensor(self.class_weight, device=device)

        if self.logit:
            inputs = torch.sigmoid(inputs)

        # compute bce loss
        if self.bce_weight is not None:
            self.bce_weight = torch.tensor(self.bce_weight, device=device)
        bce_loss_fn = nn.BCELoss(self.bce_weight, reduction=self.reduction)
        bce = 0
        for i, _weight in enumerate(self.class_weight):
            _inputs = inputs[:, i, :, :].view(-1)
            _targets = targets[:, i, :, :].contiguous().view(-1)
            _bce = bce_loss_fn(_inputs, _targets)*_weight
            bce += _bce

        # compute dice loss
        dice_loss = self.dice_loss_fn(inputs, targets, class_weight=self.class_weight)

        # return bce loss + dice loss
        return bce + dice_loss


class DiceCELoss(nn.Module):
    r"""
    Compute dice + bce loss for binary and multilabel tasks.
    """
    def __init__(
        self,
        class_weight: Optional[Union[list, Tensor]] = None,
        logit: bool = True,
        reduction: str = "mean",
        smooth: float = 1.,
    ):
        super(DiceCELoss, self).__init__()
        self.class_weight = class_weight
        self.logit = logit
        self.reduction = reduction
        self.dice_loss_fn = DiceLoss(logit=False, task="multiclass", smooth=smooth)

    def forward(self, inputs: Tensor, targets: Tensor):
        device = inputs.device
        b, c, *_ = inputs.size()

        if self.class_weight is None:
            self.class_weight = [1]*c
        self.class_weight = torch.tensor(self.class_weight, device=device)

        # compute ce loss
        ce_loss_fn = nn.CrossEntropyLoss(self.class_weight, reduction=self.reduction)
        ce_targets = torch.argmax(targets, dim=1)
        ce = ce_loss_fn(inputs, ce_targets)

        # compute dice loss
        if self.logit:
            inputs = inputs.softmax(dim=1)

        dice_loss = self.dice_loss_fn(inputs, targets, class_weight=self.class_weight)

        # return ce loss + dice loss
        return ce + dice_loss
