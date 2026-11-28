from utils.monitoring import summarize_operational_health


def test_summarize_operational_health_healthy_and_degraded() -> None:
    sessions = [
        {"employee_name": "Nisha", "logout_time_utc": "ACTIVE (Still Signed In)"},
        {"employee_name": "Saye", "logout_time_utc": "2026-11-28T16:00:00+00:00"},
    ]
    healthy = summarize_operational_health(sessions, [0.7766, 0.7695], [18.2, 22.4, 25.0])
    assert healthy["total_sessions"] == 2
    assert healthy["active_sessions"] == 1
    assert healthy["status"] == "HEALTHY"
    assert healthy["p95_inference_ms"] >= 22.4

    degraded = summarize_operational_health(sessions, [0.77, 0.15], [19.0], dice_alert_threshold=0.50)
    assert degraded["low_dice_alerts"] == 1
    assert degraded["status"] == "DEGRADED"
