from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from client.client_node import select_device
from models.unet_model import MedicalUNet
from utils.segmentation_eval import compute_3d_dice, threshold_probabilities


def _find_checkpoints() -> List[Dict[str, Any]]:
    candidates: List[Tuple[str, str, Optional[float], Path]] = [
        ("Active Unarchived: artifacts/global_unet.pt", "unarchived", None, Path("artifacts/global_unet.pt")),
        ("High Utility (ε=10.0): eps_10.0", "dp_eps_10.0", 10.0, Path("artifacts/runs/eps_10.0")),
        ("Strict Privacy (ε=2.0): latest_eps2", "dp_eps_2.0", 2.0, Path("artifacts/runs/latest_eps2")),
    ]
    found: List[Dict[str, Any]] = []
    for label, tag, eps, path in candidates:
        pt_path: Optional[Path] = None
        if path.is_file() and path.suffix == ".pt":
            pt_path = path
        elif path.is_dir():
            pts = sorted(path.rglob("*.pt"))
            if pts:
                pt_path = pts[0]
        if pt_path is not None and pt_path.exists():
            found.append({"label": label, "tag": tag, "epsilon": eps, "checkpoint": str(pt_path)})
    return found


def _extract_state_dict(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return state


def _load_model(ckpt_path: Path, device: torch.device) -> Tuple[MedicalUNet, int, int]:
    state = _extract_state_dict(ckpt_path, device)
    in_ch = 2
    out_ch = 2
    first_w = state.get("model.model.0.conv.unit0.conv.weight")
    if isinstance(first_w, torch.Tensor) and first_w.ndim >= 2:
        in_ch = int(first_w.shape[1])
    for k, v in reversed(list(state.items())):
        if k.endswith(".conv.bias") and isinstance(v, torch.Tensor) and v.ndim == 1:
            out_ch = int(v.shape[0])
            break
    model = MedicalUNet(in_channels=in_ch, out_channels=out_ch).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, in_ch, out_ch


def _collect_client_cases(data_root: Path = Path("data")) -> Dict[str, List[Path]]:
    clients: Dict[str, List[Path]] = {}
    for client_dir in sorted(data_root.glob("client_*")):
        if not client_dir.is_dir():
            continue
        cases = sorted(client_dir.rglob("*.npz"))
        if cases:
            clients[client_dir.name] = cases
    return clients


def _extract_volume_and_mask(npz_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    with np.load(npz_path) as arr:
        keys = list(arr.files)
        x_key = next((k for k in ("image", "scan", "x", "vol") if k in keys), keys[0] if keys else None)
        y_key = next((k for k in ("label", "mask", "gt", "y", "target") if k in keys), keys[1] if len(keys) > 1 else None)
        if x_key is None or y_key is None:
            return None
        x = np.asarray(arr[x_key], dtype=np.float32)
        y = np.asarray(arr[y_key], dtype=np.float32)
        return x, y


def _prepare_input(vol: np.ndarray, expected_in_ch: int, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(vol).to(device)
    if t.ndim == 3:
        t = t.unsqueeze(0)  # [1, D, H, W]
    if t.shape[0] != expected_in_ch:
        if expected_in_ch == 2 and t.shape[0] == 1:
            t = torch.cat([t, t], dim=0)
        elif expected_in_ch == 1 and t.shape[0] > 1:
            t = t[:1]
    return t.unsqueeze(0)  # [1, C, D, H, W]


def _logits_to_fg_prob(logits: torch.Tensor) -> np.ndarray:
    if logits.shape[1] >= 2:
        probs = torch.softmax(logits, dim=1)[:, 1]
    else:
        probs = torch.sigmoid(logits)[:, 0]
    return probs.squeeze(0).cpu().numpy()


def run_ablation(seeds: Optional[List[int]] = None, threshold: float = 0.5) -> Dict[str, Any]:
    seeds = seeds or [42, 123, 777]
    device = select_device()
    checkpoints = _find_checkpoints()
    client_cases = _collect_client_cases()

    results: List[Dict[str, Any]] = []
    for ckpt in checkpoints:
        model, in_ch, out_ch = _load_model(Path(ckpt["checkpoint"]), device)
        per_client_summary: Dict[str, Any] = {}
        all_case_dices: List[float] = []
        latencies_ms: List[float] = []

        for client_name, cases in client_cases.items():
            client_dices: List[float] = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                for case_path in cases:
                    pair = _extract_volume_and_mask(case_path)
                    if pair is None:
                        continue
                    vol, mask = pair
                    tensor_in = _prepare_input(vol, in_ch, device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        logits = model(tensor_in)
                        probs = _logits_to_fg_prob(logits)
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                    pred_mask = threshold_probabilities(probs, threshold=threshold)
                    gt_bin = np.squeeze(mask > 0.5)
                    if pred_mask.shape != gt_bin.shape:
                        continue
                    dice_val = float(compute_3d_dice(pred_mask, gt_bin))
                    client_dices.append(dice_val)
                    all_case_dices.append(dice_val)

            if client_dices:
                per_client_summary[client_name] = {
                    "n_evaluations": len(client_dices),
                    "mean_dice_3d": round(float(np.mean(client_dices)), 4),
                    "std_dice_3d": round(float(np.std(client_dices)), 4),
                }

        results.append(
            {
                "label": ckpt["label"],
                "tag": ckpt["tag"],
                "epsilon": ckpt["epsilon"],
                "checkpoint": ckpt["checkpoint"],
                "in_channels": in_ch,
                "out_channels": out_ch,
                "seeds": seeds,
                "threshold": threshold,
                "mean_dice_3d": round(float(np.mean(all_case_dices)), 4) if all_case_dices else None,
                "std_dice_3d": round(float(np.std(all_case_dices)), 4) if all_case_dices else None,
                "p95_inference_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else None,
                "per_client": per_client_summary,
            }
        )

    return {
        "device": str(device),
        "seeds": seeds,
        "clients_evaluated": sorted(client_cases.keys()),
        "results": results,
    }


def main() -> None:
    out_path = Path("artifacts/runs/ablation_grid.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.copy2(out_path, out_path.with_suffix(".json.bak"))
    payload = run_ablation()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(payload['results'])} checkpoint evaluations.")


if __name__ == "__main__":
    main()
