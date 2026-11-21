import unittest
import torch
from models.unet_model import MedicalUNet
from utils.dp_noise import clip_and_add_dp_noise


class TestFederatedPipeline(unittest.TestCase):
    def test_unet_and_dp_noise(self):
        model = MedicalUNet(in_channels=2, out_channels=2)
        x = torch.randn(1, 2, 16, 16, 16)
        target = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)

        out = model(x)
        self.assertEqual(out.shape, (1, 2, 16, 16, 16))

        loss = torch.nn.functional.cross_entropy(out, target.squeeze(1))
        loss.backward()
        norm = clip_and_add_dp_noise(model, epsilon=2.0, delta=1e-5, max_grad_norm=1.0, enabled=True)
        self.assertGreaterEqual(norm, 0.0)


if __name__ == "__main__":
    unittest.main()
