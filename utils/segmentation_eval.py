from __future__ import annotations

import numpy as np
from PIL import Image


def threshold_probabilities(probs: np.ndarray, threshold: float = 0.60) -> np.ndarray:
    """Converts foreground probability map into a binary segmentation mask."""
    return (np.asarray(probs, dtype=np.float32) >= float(threshold)).astype(np.int64)


def compute_3d_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    """
    Computes binary 3D Dice score for foreground class == 1.
    Returns 1.0 when both prediction and ground truth have zero foreground voxels.
    """
    p_fg = (np.asarray(pred_mask) == 1)
    t_fg = (np.asarray(target_mask) == 1)
    denom = int(p_fg.sum() + t_fg.sum())
    if denom == 0:
        return 1.0
    inter = int((p_fg & t_fg).sum())
    return float((2.0 * inter) / denom)


def to_match_error_image(
    pred_2d: np.ndarray,
    gt_2d: np.ndarray,
    display_size: int = 200,
) -> Image.Image:
    """
    Renders a 2D slice comparison map:
      - Green  (34, 197, 94)  : True Positive (prediction & ground truth agree on lesion)
      - Red    (239, 68, 68)  : False Positive (over-predicted foreground)
      - Blue   (59, 130, 246) : False Negative (missed ground-truth lesion)
      - Black  (15, 23, 42)   : True Negative (background agreement)
    """
    p = np.asarray(pred_2d) > 0
    g = np.asarray(gt_2d) > 0
    rgb = np.zeros((*p.shape, 3), dtype=np.uint8)
    rgb[~p & ~g] = [15, 23, 42]
    rgb[p & g] = [34, 197, 94]
    rgb[p & ~g] = [239, 68, 68]
    rgb[~p & g] = [59, 130, 246]
    return Image.fromarray(rgb).resize(
        (display_size, display_size), resample=Image.Resampling.NEAREST
    )

def build_batch_report_payload(rows: list[dict]) -> dict:
    total = len(rows)
    fg_vals = [float(r["predicted_fg_pct"]) for r in rows if r.get("predicted_fg_pct") is not None]
    dice_vals = [float(r["volume_dice"]) for r in rows if r.get("volume_dice") is not None]
    return {
        "case_count": total,
        "mean_predicted_fg_pct": round(sum(fg_vals) / len(fg_vals), 4) if fg_vals else None,
        "mean_volume_dice": round(sum(dice_vals) / len(dice_vals), 4) if dice_vals else None,
        "cases": rows,
    }


def rows_to_csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    csv_lines = [",".join(headers)]
    for r in rows:
        csv_lines.append(",".join(str(r.get(h, "")).replace(",", ";") for h in headers))
    return ("\n").join(csv_lines) + "\n"
