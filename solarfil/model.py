"""Network, losses and (TTA) inference."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FilamentNet(nn.Module):
    """U-Net with 2 output channels: 0 = filament foreground, 1 = instance border."""

    def __init__(self, arch="Unet", encoder="efficientnet-b4", weights="imagenet"):
        super().__init__()
        kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=2)
        if arch in ("Unet", "UnetPlusPlus"):
            kw["decoder_attention_type"] = "scse"
        self.net = getattr(smp, arch)(**kw)
        self.register_buffer("mean", MEAN.clone(), persistent=False)
        self.register_buffer("std", STD.clone(), persistent=False)

    def forward(self, x):  # x in [0,1]
        return self.net((x - self.mean) / self.std)


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2))
    den = p.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def soft_skeleton(p: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable morphological skeleton of a soft mask (B,H,W) in [0,1] (Shit et al., clDice, CVPR 2021):
    repeated soft erosion (min-pooling) and opening, keeping what each opening removes."""
    def erode(x):
        return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))

    def dilate(x):
        return F.max_pool2d(x, 3, 1, 1)

    x = p.unsqueeze(1)
    skel = F.relu(x - dilate(erode(x)))
    for _ in range(iters):
        x = erode(x)
        delta = F.relu(x - dilate(erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel.squeeze(1)


def soft_cldice_loss(logits, target, iters: int = 10, eps: float = 1.0):
    """1 - clDice: harmonic mean of the share of the predicted skeleton inside the target and of the target skeleton
    inside the prediction. Rewards thin, elongated structures (filament tails, barbs) that area losses barely see."""
    p = torch.sigmoid(logits)
    sp, st = soft_skeleton(p, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum(dim=(1, 2)) + eps) / (sp.sum(dim=(1, 2)) + eps)
    tsens = ((st * p).sum(dim=(1, 2)) + eps) / (st.sum(dim=(1, 2)) + eps)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


class FilamentLoss(nn.Module):
    def __init__(self, border_weight=0.5, pos_weight=3.0, cldice_weight=0.0):
        super().__init__()
        self.bw = border_weight
        self.pw = pos_weight
        self.cw = cldice_weight

    def forward(self, logits, target):
        fg_l, bd_l = logits[:, 0], logits[:, 1]
        fg_t, bd_t = target[:, 0], target[:, 1]
        pw = torch.tensor(self.pw, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(fg_l, fg_t, pos_weight=pw) + soft_dice_loss(fg_l, fg_t)
        if self.bw > 0:
            loss = loss + self.bw * (F.binary_cross_entropy_with_logits(bd_l, bd_t, pos_weight=pw * 3)
                                     + 0.5 * soft_dice_loss(bd_l, bd_t))
        if self.cw > 0:
            loss = loss + self.cw * soft_cldice_loss(fg_l, fg_t)
        return loss


@torch.no_grad()
def predict_full(models, x: torch.Tensor, tta: int = 8, amp: bool = True) -> torch.Tensor:
    """x: (3,H,W) in [0,1]. Returns averaged sigmoid probs (2,H,W) on CPU.

    tta: number of dihedral transforms (1, 2, 4 or 8)."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    dev = next(models[0].parameters()).device
    xb = x.unsqueeze(0).to(dev)
    acc = None
    n = 0
    for k in range(tta):
        r, f = k % 4, k >= 4
        xt = torch.rot90(xb, r, dims=(2, 3))
        if f:
            xt = torch.flip(xt, dims=(3,))
        for m in models:
            with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
                p = torch.sigmoid(m(xt).float())
            if f:
                p = torch.flip(p, dims=(3,))
            p = torch.rot90(p, -r, dims=(2, 3))
            acc = p if acc is None else acc + p
            n += 1
    return (acc / n)[0].float().cpu()


class EMA:
    def __init__(self, model, decay=0.999):
        import copy
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])
