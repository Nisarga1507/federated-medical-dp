from __future__ import annotations

from typing import Any, Dict, List


def summarize_operational_health(
    paired_sessions: List[Dict[str, Any]],
    dice_scores: List[float],
    latencies_ms: List[float],
    dice_alert_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes lightweight operational telemetry across RBAC sessions, segmentation
    3D Dice quality drift, and inference latency.
    """
    active_count = sum(
        1 for s in paired_sessions if str(s.get("logout_time_utc", "")).startswith("ACTIVE")
    )
    mean_dice = round(sum(dice_scores) / len(dice_scores), 4) if dice_scores else 0.0
    low_dice_alerts = sum(1 for d in dice_scores if d < dice_alert_threshold)
    p95_latency_ms = 0.0
    if latencies_ms:
        ordered = sorted(latencies_ms)
        idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
        p95_latency_ms = round(float(ordered[idx]), 2)

    return {
        "total_sessions": len(paired_sessions),
        "active_sessions": active_count,
        "evaluated_scans": len(dice_scores),
        "mean_dice_3d": mean_dice,
        "low_dice_alerts": low_dice_alerts,
        "p95_inference_ms": p95_latency_ms,
        "status": "DEGRADED" if low_dice_alerts > 0 else "HEALTHY",
    }
