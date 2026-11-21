import math
import torch
import torch.nn as nn


def clip_and_add_dp_noise(
    model: nn.Module,
    epsilon: float,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    enabled: bool = True,
) -> float:
    """
    Clips parameter gradients to `max_grad_norm` and injects Gaussian noise
    scaled to (epsilon, delta)-Differential Privacy when `enabled=True`.
    Returns the total gradient L2 norm prior to clipping.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    total_norm = torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads)).item()

    if not enabled:
        return total_norm

    clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
    # Calibrated noise scale factor for (epsilon, delta)-Gaussian mechanism
    sigma = (max_grad_norm * math.sqrt(2.0 * math.log(1.25 / delta))) / max(epsilon, 1e-6)

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
