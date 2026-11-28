import torch
import torch.nn as nn
from utils.dp_noise import clip_and_add_dp_noise, compute_sigma


def test_dp_noise_disabled_preserves_exact_gradients():
    torch.manual_seed(42)
    layer = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    loss = layer(x).sum()
    loss.backward()
    before = layer.weight.grad.clone()

    norm = clip_and_add_dp_noise(layer, epsilon=2.0, enabled=False)
    assert norm > 0.0
    assert torch.allclose(layer.weight.grad, before)


def test_dp_sigma_decreases_as_epsilon_increases():
    s_tight = compute_sigma(epsilon=2.0, delta=1e-5, max_grad_norm=1.0, batch_size=2)
    s_loose = compute_sigma(epsilon=10.0, delta=1e-5, max_grad_norm=1.0, batch_size=2)
    assert s_tight > s_loose > 0.0
