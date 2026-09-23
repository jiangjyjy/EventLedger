from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_huber_loss(pred: Tensor, target: Tensor, mask: Tensor, delta: float = 1.0) -> Tensor:
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], target[mask], delta=delta)


def bradley_terry_loss(preferred: Tensor, rejected: Tensor) -> Tensor:
    if preferred.numel() == 0:
        return (preferred.sum() + rejected.sum()) * 0.0
    return F.softplus(-(preferred - rejected)).mean()
