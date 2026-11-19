"""What the training of a slice model does on the GPU: window, augmentation, learning rate and prediction (04_slice_models)."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from pe.data.dicom import HU_MIN


def to_window(stored: torch.Tensor, level: float, width: float) -> torch.Tensor:
    """Stored values (HU + 1024) to floats in [0, 1]: 0 up to level - width / 2, 1 from level + width / 2."""
    return ((stored.float() + HU_MIN - (level - width / 2)) / width).clamp(0, 1)


def to_windows(stored: torch.Tensor, windows: list[tuple[float, float]]) -> torch.Tensor:
    """Every stored channel through every window (level, width): slices x windows channels."""
    return torch.cat([to_window(stored, level, width) for level, width in windows], dim=1)


def augment(x: torch.Tensor, fill=0.0, rotate: float = 15, zoom: float = 0.1, shift: float = 0.05) -> torch.Tensor:
    """A random rotation (degrees), zoom and shift (shares of the image) per item. Never a horizontal flip, which would
    swap left and right (02). The corners that come in get `fill`: air after the window, per channel."""
    n = len(x)
    draw = lambda limit: (2 * torch.rand(n, device=x.device) - 1) * limit
    angle, scale = draw(math.radians(rotate)), 1 + draw(zoom)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([cos, -sin, 2 * draw(shift), sin, cos, 2 * draw(shift)], dim=1).view(n, 2, 3)
    return F.grid_sample(x - fill, F.affine_grid(theta, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False) + fill


def learning_rate(step: int, steps: int, warmup: float = 0.05) -> float:
    """The share of the full learning rate at a step: linear warm-up over the first 5%, then a cosine down to 0."""
    n_warmup = max(1, round(warmup * steps))
    if step < n_warmup:
        return (step + 1) / n_warmup
    return 0.5 * (1 + math.cos(math.pi * (step - n_warmup) / max(1, steps - n_warmup)))


def slice_loss(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """Mean weighted binary cross-entropy. With `gamma`, the focal loss: each item is scaled by (1 - p of its true class) ** gamma."""
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if gamma:
        loss = loss * (1 - torch.exp(-loss)) ** gamma
    return (weight * loss).mean()


@torch.no_grad()
def predict(model, loader, windows: list[tuple[float, float]]) -> np.ndarray:
    """Logits of a loader that keeps its order."""
    model.eval()
    logits = []
    for stored, _ in loader:
        x = to_windows(stored.cuda(non_blocking=True), windows).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits.append(model(x)[0].float().cpu())
    return torch.cat(logits).numpy()
