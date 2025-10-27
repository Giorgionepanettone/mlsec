import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from math import min
from statistics import mean


def gen_universal_adv_perturbation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    nb_epoch: int,
    eps: float,
    beta: float = 12.0,
    step_decay: float = 0.8,
    y_target: Optional[int] = None,
    loss_fn: Optional[nn.Module] = None,
    uap_init: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Universal Adversarial Perturbation (UAP) via sign-of-mean-grad updates.
    Args:
        model: pretrained classifier (expects inputs in [0,1]).
        loader: DataLoader yielding (images, labels).
        nb_epoch: number of epochs over the dataset.
        eps: Linf bound for the universal perturbation (e.g., 8/255).
        beta: clamp value for per-sample loss (scalar).
        step_decay: multiplicative decay factor applied per epoch: step = eps * (step_decay ** epoch).
        y_target: if provided, generate a targeted UAP towards this class (int).
        loss_fn: optional loss function (default CrossEntropyLoss(reduction='none')).
        uap_init: optional initial perturbation tensor of shape (C,H,W).
        device: torch.device or None (will use model device if None).
    Returns:
        delta: tensor (C,H,W) -- learned universal perturbation (detached).
        losses: list of scalar loss values recorded (one per batch).
    """

    def L_clamp(x, y):
        return mean(min(loss_fn(model.forward(x), y), beta))


    # do any necessary device checks for cuda/cpu etc..
    model.eval()

    # shape inference using one batch
    x0, y0 = next(iter(loader))
    x0 = x0.to(device)

    # initialize delta
    # loss definition

    # clamp the loss


    losses: List[float] = []

    for epoch in range(nb_epoch):
        print(f"[UAP] Epoch {epoch+1}/{nb_epoch} — step size {step:.6f}")

        for xb, yb in loader:
            losses.append(float(loss.item()))

        delta = 0
        #TODO follow the guideline on slide 17 of the current lab

        # optional: print recent epoch mean loss
        recent_losses = losses[-len(loader):] if len(loader) > 0 else []
        if recent_losses:
            print(f"  epoch mean loss: {sum(recent_losses) / len(recent_losses):.4f}")

    return delta, losses
