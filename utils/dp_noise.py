import math
from typing import Optional
import torch
import torch.nn as nn


def compute_sigma(
    epsilon: float,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    batch_size: int = 1,
    noise_multiplier: Optional[float] = None,
) -> float:
    safe_bs = max(int(batch_size), 1)
    if noise_multiplier is not None:
        return float(noise_multiplier) * float(max_grad_norm) / math.sqrt(safe_bs)
    safe_eps = max(float(epsilon), 1e-6)
    safe_delta = min(max(float(delta), 1e-12), 1.0 - 1e-12)
    base_sigma = (float(max_grad_norm) * math.sqrt(2.0 * math.log(1.25 / safe_delta))) / safe_eps
    return base_sigma / math.sqrt(safe_bs)


def clip_and_add_dp_noise(
    model: nn.Module,
    epsilon: float,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    enabled: bool = True,
    batch_size: int = 1,
    noise_multiplier: Optional[float] = None,
) -> float:
    """
    Clips parameter gradients to `max_grad_norm` and injects Gaussian noise
    scaled to (epsilon, delta)-Differential Privacy when `enabled=True`.
    Returns the total gradient L2 norm prior to clipping.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    total_norm = float(torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads)).item())

    if not enabled:
        return total_norm

    clip_coef = min(1.0, float(max_grad_norm) / (total_norm + 1e-6))
    sigma = compute_sigma(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
    )

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)
                noise = torch.normal(
                    mean=0.0,
                    std=sigma,
                    size=p.grad.shape,
                    device=p.grad.device,
                    dtype=p.grad.dtype,
                )
                p.grad.add_(noise)

    return total_norm
