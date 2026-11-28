from __future__ import annotations

import sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from client.client_node import select_device
from utils.dp_noise import clip_and_add_dp_noise
from models.unet_model import MedicalUNet
from utils.segmentation_eval import compute_3d_dice, threshold_probabilities


def _load_sample_batch(data_dir: Path, max_cases: int = 4) -> List[tuple[np.ndarray, np.ndarray]]:
    cases: List[tuple[np.ndarray, np.ndarray]] = []
    for path in sorted(data_dir.glob("client_*/test/*.npz"))[:max_cases]:
        with np.load(path) as arr:
            if "image" in arr and "label" in arr:
                img = arr["image"].astype(np.float32)
                if img.ndim == 3:
                    img = np.stack([img, img], axis=0)
                lbl = arr["label"].astype(np.int64)
                if lbl.ndim == 3:
                    lbl = np.stack([lbl, lbl], axis=0)
                cases.append((img, lbl))
    return cases


def _eval_dice(model: nn.Module, cases: List[tuple[np.ndarray, np.ndarray]], device: torch.device, threshold: float = 0.5) -> float:
    if not cases:
        return 0.0
    model.eval()
    scores: List[float] = []
    with torch.no_grad():
        for img, lbl in cases:
            x = torch.from_numpy(img).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1)[:, 1].squeeze(0).cpu().numpy()
            pred = threshold_probabilities(probs, threshold=threshold)
            gt = lbl[0] if lbl.shape[0] > 1 else lbl
            if gt.shape == pred.shape:
                scores.append(compute_3d_dice(pred, gt))
    return float(np.mean(scores)) if scores else 0.0


def run_ablation(
    epsilons: List[Optional[float]],
    seeds: List[int],
    rounds: int = 3,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    data_dir: Path = Path("data"),
    out_path: Path = Path("artifacts/runs/ablation_grid.json"),
) -> Dict[str, Any]:
    device = select_device()
    cases = _load_sample_batch(data_dir=data_dir)
    criterion = nn.CrossEntropyLoss()
    grid_summary: List[Dict[str, Any]] = []

    for eps in epsilons:
        seed_results: List[Dict[str, Any]] = []
        for seed in seeds:
            t0 = time.perf_counter()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MedicalUNet(in_channels=2, out_channels=2).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            if cases:
                x_np, y_np = cases[0]
                x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
                y_fg = (torch.from_numpy(y_np[0]).unsqueeze(0) > 0).long().to(device)
            else:
                x_t = torch.randn(1, 2, 16, 32, 32, device=device)
                y_fg = torch.zeros(1, 16, 32, 32, dtype=torch.long, device=device)

            for _ in range(rounds):
                model.train()
                optimizer.zero_grad()
                logits = model(x_t)
                loss = criterion(logits, y_fg)
                loss.backward()
                if eps is not None:
                    clip_and_add_dp_noise(
                        model,
                        epsilon=float(eps),
                        delta=delta,
                        max_grad_norm=max_grad_norm,
                    )
                optimizer.step()

            dice_score = _eval_dice(model, cases, device=device)
            elapsed_s = round(time.perf_counter() - t0, 4)
            seed_results.append({"seed": seed, "dice_3d": round(dice_score, 4), "elapsed_s": elapsed_s})

        dices = [r["dice_3d"] for r in seed_results]
        grid_summary.append(
            {
                "epsilon": "inf" if eps is None else float(eps),
                "delta": delta,
                "max_grad_norm": max_grad_norm,
                "rounds": rounds,
                "seeds": seeds,
                "mean_dice_3d": round(float(np.mean(dices)), 4),
                "std_dice_3d": round(float(np.std(dices)), 4),
                "per_seed": seed_results,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "matched_rounds_and_seeds",
        "device": str(device),
        "num_eval_cases": len(cases),
        "results": grid_summary,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled multi-seed epsilon ablation.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs/ablation_grid.json"))
    args = parser.parse_args()
    eps_grid: List[Optional[float]] = [2.0, 5.0, 10.0, None]
    report = run_ablation(epsilons=eps_grid, seeds=args.seeds, rounds=args.rounds, out_path=args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
