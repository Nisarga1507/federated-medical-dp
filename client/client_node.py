from pathlib import Path
from typing import Dict, List, Tuple
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from client.dataset import build_dataloaders
from models.unet_model import MedicalUNet
from utils.dp_noise import clip_and_add_dp_noise


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_binary_dice(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1, keepdim=True)
    pred_fg = (preds == 1).float()
    true_fg = (targets == 1).float()
    intersection = (pred_fg * true_fg).sum().item()
    denom = pred_fg.sum().item() + true_fg.sum().item()
    if denom == 0.0:
        return 1.0
    return (2.0 * intersection) / denom


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, cfg: dict):
        self.client_id = client_id
        self.cfg = cfg
        self.device = select_device()

        data_cfg = cfg["data"]
        train_cfg = cfg["training"]
        priv_cfg = cfg["privacy"]

        self.model = MedicalUNet(
            in_channels=int(data_cfg["in_channels"]),
            out_channels=int(data_cfg["out_channels"]),
        ).to(self.device)

        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            client_id=client_id,
            splits_counts=data_cfg["samples_per_client"],
            in_channels=int(data_cfg["in_channels"]),
            spatial_size=tuple(data_cfg["spatial_size"]),
            batch_size=int(train_cfg["batch_size"]),
            data_root=Path("data"),
        )

        self.epochs = int(train_cfg["epochs"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
        )
        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True)

        self.dp_enabled = bool(priv_cfg.get("enabled", True))
        self.epsilon = float(priv_cfg.get("epsilon", 2.0))
        self.delta = float(priv_cfg.get("delta", 1e-5))
        self.max_grad_norm = float(priv_cfg.get("max_grad_norm", 1.0))

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        keys = list(self.model.state_dict().keys())
        state_dict = {
            k: torch.tensor(v, device=self.device, dtype=self.model.state_dict()[k].dtype)
            for k, v in zip(keys, parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.model.train()

        last_loss = 0.0
        last_grad_norm = 0.0

        for _ in range(self.epochs):
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                last_grad_norm = clip_and_add_dp_noise(
                    model=self.model,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    max_grad_norm=self.max_grad_norm,
                    enabled=self.dp_enabled,
                )
                self.optimizer.step()
                last_loss = float(loss.item())

        metrics = {
            f"client_{self.client_id}_train_loss": last_loss,
            f"client_{self.client_id}_grad_norm": float(last_grad_norm),
            "epsilon": self.epsilon if self.dp_enabled else -1.0,
        }
        return self.get_parameters(config), len(self.train_loader.dataset), metrics

    def _evaluate_loader(self, loader) -> Tuple[float, float, int]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                dice = compute_binary_dice(outputs, labels)

                bs = images.shape[0]
                total_loss += float(loss.item()) * bs
                total_dice += dice * bs
                count += bs

        return total_loss / max(count, 1), total_dice / max(count, 1), count

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        val_loss, val_dice, val_n = self._evaluate_loader(self.val_loader)
        test_loss, test_dice, _ = self._evaluate_loader(self.test_loader)

        metrics = {
            f"client_{self.client_id}_val_loss": float(val_loss),
            f"client_{self.client_id}_val_dice": float(val_dice),
            f"client_{self.client_id}_test_loss": float(test_loss),
            f"client_{self.client_id}_test_dice": float(test_dice),
        }
        return float(val_loss), val_n, metrics
