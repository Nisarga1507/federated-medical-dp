import json
from pathlib import Path
from typing import Dict, List


def load_run_summary(metrics_path: Path = Path("artifacts/federated_metrics.json")) -> Dict:
    if not metrics_path.exists():
        return {
            "rounds_completed": 0,
            "epsilon": 2.0,
            "delta": 1e-5,
            "train_samples_per_client": 12,
            "val_samples_per_client": 4,
            "test_samples_per_client": 4,
            "latest_test_dice": None,
            "latest_val_loss": None,
        }

    raw = json.loads(metrics_path.read_text())
    cfg = raw.get("config", {})
    rounds = raw.get("rounds", [])

    splits = cfg.get("data", {}).get("samples_per_client", {})
    priv = cfg.get("privacy", {})

    latest_test_dice = None
    latest_val_loss = None
    for item in reversed(rounds):
        eval_metrics = item.get("eval", {}).get("metrics", {})
        if latest_val_loss is None and item.get("eval", {}).get("loss") is not None:
            latest_val_loss = float(item["eval"]["loss"])
        test_dices = [
            float(v) for k, v in eval_metrics.items() if k.endswith("_test_dice")
        ]
        if test_dices:
            latest_test_dice = sum(test_dices) / len(test_dices)
            break

    return {
        "rounds_completed": len(rounds),
        "epsilon": float(priv.get("epsilon", 2.0)),
        "delta": float(priv.get("delta", 1e-5)),
        "train_samples_per_client": int(splits.get("train", 12)),
        "val_samples_per_client": int(splits.get("val", 4)),
        "test_samples_per_client": int(splits.get("test", 4)),
        "latest_test_dice": latest_test_dice,
        "latest_val_loss": latest_val_loss,
    }


def generate_cv_bullets(summary: Dict) -> List[str]:
    rounds = summary["rounds_completed"]
    eps = summary["epsilon"]
    delta = summary["delta"]
    tr = summary["train_samples_per_client"]
    va = summary["val_samples_per_client"]
    te = summary["test_samples_per_client"]
    dice = summary.get("latest_test_dice")

    dice_clause = (
        f", achieving a mean held-out test Dice score of {dice:.3f} across isolated client test partitions"
        if dice is not None
        else " with per-node train/validation/test evaluation splits"
    )

    return [
        (
            f"Architected a privacy-preserving Federated Learning 3D medical segmentation pipeline in Python, PyTorch, "
            f"MONAI, and Flower across decentralized hospital nodes ({tr} train / {va} val / {te} test scans per node), "
            f"keeping raw patient imaging local to support GDPR and HIPAA compliance."
        ),
        (
            f"Implemented Federated Averaging (FedAvg) over {rounds} communication rounds with local gradient clipping and "
            f"calibrated Gaussian noise injection for (epsilon={eps}, delta={delta:.0e})-Differential Privacy to mitigate "
            f"patient re-identification and membership inference risks."
        ),
        (
            f"Trained a multi-channel 3D MONAI UNet using combined Dice and Cross-Entropy loss{dice_clause}, and built an "
            f"interactive inspection dashboard for volume inference, privacy-utility tracking, and automated run summaries."
        ),
    ]
