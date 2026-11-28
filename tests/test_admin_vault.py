import cv2
import numpy as np
from utils.admin_vault import (
    cosine_similarity,
    extract_face_signature,
    load_patient_records,
    load_staff_audit_events,
    login_staff_member,
    logout_staff_session,
    save_patient_record,
    verify_admin_face,
)


def _make_sample_jpg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    frame = rng.integers(40, 220, size=(240, 320, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", frame)
    return encoded.tobytes()


def test_unauthorized_user_cannot_save_or_load_vault(tmp_path):
    record = {
        "patient_name": "Test Patient",
        "disease": "Glioma",
        "dob": "1995-04-12",
        "source": "case_000.npz",
        "dice": 0.7695,
    }
    ok, msg = save_patient_record(record, is_admin_unlocked=False, vault_dir=tmp_path)
    assert ok is False
    assert "sign in" in msg.lower()
    assert load_patient_records(is_admin_unlocked=False, vault_dir=tmp_path) == []

    ok_admin, _ = save_patient_record(record, is_admin_unlocked=True, vault_dir=tmp_path)
    assert ok_admin is True
    loaded = load_patient_records(is_admin_unlocked=True, vault_dir=tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["patient_name"] == "Test Patient"


def test_verify_admin_rejects_when_unenrolled_or_blank_frame(tmp_path):
    blank = np.zeros((120, 120, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)
    verified, score, _ = verify_admin_face(encoded.tobytes(), vault_dir=tmp_path)
    assert verified is False
    assert score == 0.0


def test_cosine_similarity_bounds():
    v = np.ones(64, dtype=np.float32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5


def test_extract_face_signature_on_textured_frame():
    sig, _ = extract_face_signature(_make_sample_jpg(123))
    assert sig is not None
    assert abs(float(np.linalg.norm(sig)) - 1.0) < 1e-4


def test_staff_shift_ampm_login_capture_patient_result_and_admin_complete_view(tmp_path):
    # Shift without AM/PM is rejected
    bad_ok, _, bad_msg = login_staff_member("Staff 1", "Asha", "asha@clinic.org", "09:00 - 17:00", vault_dir=tmp_path)
    assert bad_ok is False
    assert "am/pm" in bad_msg.lower()

    cap_bytes = _make_sample_jpg(55)
    ok, staff_sess, _ = login_staff_member(
        "Staff 1",
        "Asha Rao",
        "asha.rao@clinic.org",
        "09:00 AM - 05:00 PM",
        capture_image_bytes=cap_bytes,
        vault_dir=tmp_path,
    )
    assert ok is True
    assert staff_sess is not None
    assert staff_sess["pose_digest_sha256_16"]

    save_patient_record(
        {
            "patient_name": "Rohan Sharma",
            "disease": "Glioma",
            "dob": "1995-04-12",
            "source": "uploaded_scan.nii.gz",
            "predicted_fg_pct": 2.4,
            "mean_lesion_prob_pct": 8.1,
            "volume_dice": 0.77,
        },
        is_admin_unlocked=False,
        vault_dir=tmp_path,
        actor_info=staff_sess,
    )
    logout_staff_session(
        staff_email=staff_sess["email"],
        staff_name=staff_sess["name"],
        staff_role=staff_sess["role_label"],
        shift_time=staff_sess["shift_time"],
        capture_path=f"sha256:{staff_sess['pose_digest_sha256_16']}",
        vault_dir=tmp_path,
    )

    assert load_patient_records(is_admin_unlocked=False, vault_dir=tmp_path) == []
    admin_records = load_patient_records(is_admin_unlocked=True, vault_dir=tmp_path)
    assert len(admin_records) == 1
    assert admin_records[0]["staff_name"] == "Asha Rao"
    assert admin_records[0]["staff_shift_time"] == "09:00 AM - 05:00 PM"
    assert admin_records[0]["patient_name"] == "Rohan Sharma"

    staff_events = load_staff_audit_events(is_admin_unlocked=True, vault_dir=tmp_path)
    assert [e["event"] for e in staff_events] == ["login", "logout"]
    assert staff_events[0]["arrival_capture_path"].startswith("sha256:")


def test_named_hand_pose_and_paired_session_duration(tmp_path) -> None:
    from pathlib import Path
    from utils.admin_vault import (
        build_paired_staff_sessions,
        load_staff_audit_events,
        login_staff_member,
        logout_staff_session,
    )

    ok, sess, _ = login_staff_member(
        staff_slot="Staff 1",
        staff_name="Nisha Nair",
        staff_email="nisha@clinic.org",
        shift_time="06:00 AM - 12:00 PM",
        capture_image_bytes=_make_sample_jpg(64),
        vault_dir=tmp_path,
        branch_name="Branch 1 - North Imaging Hub",
        pose_name="Thumbs Up",
    )
    assert ok and sess is not None
    assert sess["pose_name"] == "Thumbs Up"

    logout_staff_session(
        staff_email="nisha@clinic.org",
        staff_name="Nisha Nair",
        staff_role="Staff 1",
        shift_time="Branch 1 - North Imaging Hub | 06:00 AM - 12:00 PM",
        capture_path="sha256:abcdef1234567890",
        vault_dir=tmp_path,
    )

    events = load_staff_audit_events(is_admin_unlocked=True, vault_dir=tmp_path)
    assert events[0]["pose_name"] == "Thumbs Up"
    assert events[0]["branch"] == "Branch 1 - North Imaging Hub"

    paired = build_paired_staff_sessions(events)
    assert len(paired) == 1
    assert paired[0]["hand_pose_name"] == "Thumbs Up"
    assert paired[0]["branch"] == "Branch 1 - North Imaging Hub"
    assert paired[0]["duration_minutes"] is not None
    assert paired[0]["duration_minutes"] >= 0.0
