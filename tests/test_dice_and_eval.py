import numpy as np
from utils.segmentation_eval import compute_3d_dice, threshold_probabilities, to_match_error_image


def test_dice_identical_and_disjoint_masks():
    gt = np.zeros((16, 16, 16), dtype=np.int64)
    gt[4:12, 4:12, 4:12] = 1

    assert compute_3d_dice(gt, gt) == 1.0

    disjoint = np.zeros((16, 16, 16), dtype=np.int64)
    disjoint[0:3, 0:3, 0:3] = 1
    assert compute_3d_dice(disjoint, gt) == 0.0


def test_noisy_oversegmented_mask_has_low_dice_and_higher_threshold_reduces_fg():
    rng = np.random.default_rng(7)
    gt = np.zeros((16, 16, 16), dtype=np.int64)
    gt[6:10, 6:10, 6:10] = 1

    # Simulate noisy probabilities where background has scattered ~0.55 values
    probs = rng.uniform(0.0, 0.62, size=(16, 16, 16)).astype(np.float32)
    probs[6:10, 6:10, 6:10] = 0.92

    low_thresh_mask = threshold_probabilities(probs, threshold=0.50)
    high_thresh_mask = threshold_probabilities(probs, threshold=0.75)

    assert low_thresh_mask.mean() > high_thresh_mask.mean()
    assert compute_3d_dice(low_thresh_mask, gt) < 0.35
    assert compute_3d_dice(high_thresh_mask, gt) > compute_3d_dice(low_thresh_mask, gt)


def test_match_error_image_renders_rgb():
    pred = np.array([[1, 1], [0, 0]], dtype=np.int64)
    gt = np.array([[1, 0], [1, 0]], dtype=np.int64)
    img = to_match_error_image(pred, gt, display_size=64)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_batch_report_csv_and_json_helpers():
    from utils.segmentation_eval import build_batch_report_payload, rows_to_csv_text

    rows = [
        {"patient_name": "Case A", "predicted_fg_pct": 2.5, "volume_dice": 0.75},
        {"patient_name": "Case B", "predicted_fg_pct": 1.5, "volume_dice": 0.65},
    ]
    payload = build_batch_report_payload(rows)
    assert payload["case_count"] == 2
    assert abs(payload["mean_predicted_fg_pct"] - 2.0) < 1e-4
    assert abs(payload["mean_volume_dice"] - 0.7) < 1e-4

    csv_out = rows_to_csv_text(rows)
    assert "patient_name,predicted_fg_pct,volume_dice" in csv_out
    assert "Case A,2.5,0.75" in csv_out
