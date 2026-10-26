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


class FilamentLoss(nn.Module):
    def __init__(self, border_weight=0.5, pos_weight=3.0):
        super().__init__()
        self.bw = border_weight
        self.pw = pos_weight

    def forward(self, logits, target):
        fg_l, bd_l = logits[:, 0], logits[:, 1]
        fg_t, bd_t = target[:, 0], target[:, 1]
        pw = torch.tensor(self.pw, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(fg_l, fg_t, pos_weight=pw) + soft_dice_loss(fg_l, fg_t)
        if self.bw > 0:
            loss = loss + self.bw * (F.binary_cross_entropy_with_logits(bd_l, bd_t, pos_weight=pw * 3)
                                     + 0.5 * soft_dice_loss(bd_l, bd_t))
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
