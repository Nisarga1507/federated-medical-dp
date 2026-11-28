from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import hashlib
import json
import os
from pathlib import Path
import secrets
import smtplib
import string
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


DEFAULT_VAULT_DIR = Path("artifacts/admin_vault")
PRIMARY_ADMIN_EMAIL = "nisargahsn15@gmail.com"
MAX_ADMINS = 2
MAX_STAFF = 2

SIX_HOUR_SHIFTS = [
    "06:00 AM - 12:00 PM (6h)",
    "12:00 PM - 06:00 PM (6h)",
    "06:00 PM - 12:00 AM (6h)",
    "12:00 AM - 06:00 AM (6h)",
]

SIX_BRANCHES = [
    "Branch 1 - North Imaging Hub",
    "Branch 2 - Central Diagnostics",
    "Branch 3 - Neuro Radiology Unit",
    "Branch 4 - South Clinical Center",
    "Branch 5 - West Triage Wing",
    "Branch 6 - East Screening Lab",
]



def extract_face_signature(image_bytes: bytes) -> Tuple[Optional[np.ndarray], str]:
    """
    Extracts a normalized 64x64 grayscale face feature vector from camera image bytes.
    Uses Haar CascadeClassifier when exposed by cv2, and falls back cleanly to a
    centered portrait region on OpenCV builds without CascadeClassifier.
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, "Unable to decode camera frame."

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 8.0:
        return None, "Camera frame is too dark or covered. Ensure your face is clearly lit."

    h_img, w_img = gray.shape
    crop = gray
    cascade_cls = getattr(cv2, "CascadeClassifier", None)
    if cascade_cls is not None and hasattr(cv2, "data"):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cascade_cls(cascade_path)
        if not detector.empty():
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
                crop = gray[y : y + h, x : x + w]
    else:
        y0, y1 = int(h_img * 0.15), int(h_img * 0.85)
        x0, x1 = int(w_img * 0.25), int(w_img * 0.75)
        if y1 > y0 and x1 > x0:
            crop = gray[y0:y1, x0:x1]

    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    resized = cv2.equalizeHist(resized).astype(np.float32)

    vec = resized.flatten()
    vec -= vec.mean()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None, "Face region lacked sufficient contrast."
    return vec / norm, "Face captured."


def extract_pose_signature(image_bytes: bytes) -> Tuple[Optional[np.ndarray], str]:
    """
    Extracts a hand-focused spatial pose signature from the lower/hand region of the
    camera frame for Admin sign-in verification.
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, "Unable to decode pose camera frame."

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 8.0:
        return None, "Pose camera frame is too dark or blank."

    h_img, w_img = gray.shape
    y0, y1 = int(h_img * 0.45), int(h_img * 0.95)
    x0, x1 = int(w_img * 0.15), int(w_img * 0.85)
    hand_crop = gray[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else gray

    resized = cv2.resize(hand_crop, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    combined = np.concatenate([resized.flatten(), grad_mag.flatten()]).astype(np.float32)
    combined -= combined.mean()
    norm = float(np.linalg.norm(combined))
    if norm < 1e-6:
        return None, "Hand pose frame lacked sufficient structure."
    return combined / norm, "Hand pose signature captured."


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32).flatten()
    b = np.asarray(vec_b, dtype=np.float32).flatten()
    if a.shape != b.shape or a.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_admins_meta(vault_dir: Path) -> List[Dict[str, Any]]:
    meta_path = vault_dir / "admins.json"
    if not meta_path.exists():
        return []
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _save_admins_meta(vault_dir: Path, admins: List[Dict[str, Any]]) -> None:
    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / "admins.json").write_text(json.dumps(admins, indent=2), encoding="utf-8")


def _load_staff_meta(vault_dir: Path) -> List[Dict[str, Any]]:
    meta_path = vault_dir / "staff_accounts.json"
    if not meta_path.exists():
        return []
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _save_staff_meta(vault_dir: Path, staff_list: List[Dict[str, Any]]) -> None:
    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / "staff_accounts.json").write_text(json.dumps(staff_list, indent=2), encoding="utf-8")


def get_active_admin_session(vault_dir: Path = DEFAULT_VAULT_DIR) -> Optional[Dict[str, Any]]:
    sess_path = vault_dir / "active_session.json"
    if not sess_path.exists():
        return None
    data = json.loads(sess_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) and data.get("email") else None


def _set_active_admin_session(vault_dir: Path, session_info: Optional[Dict[str, Any]]) -> None:
    sess_path = vault_dir / "active_session.json"
    if session_info is None:
        if sess_path.exists():
            sess_path.unlink()
        return
    vault_dir.mkdir(parents=True, exist_ok=True)
    sess_path.write_text(json.dumps(session_info, indent=2), encoding="utf-8")


def get_active_staff_session(vault_dir: Path = DEFAULT_VAULT_DIR) -> Optional[Dict[str, Any]]:
    sess_path = vault_dir / "staff_session.json"
    if not sess_path.exists():
        return None
    data = json.loads(sess_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) and data.get("email") else None


def _set_active_staff_session(vault_dir: Path, session_info: Optional[Dict[str, Any]]) -> None:
    sess_path = vault_dir / "staff_session.json"
    if session_info is None:
        if sess_path.exists():
            sess_path.unlink()
        return
    vault_dir.mkdir(parents=True, exist_ok=True)
    sess_path.write_text(json.dumps(session_info, indent=2), encoding="utf-8")


def is_admin_enrolled(vault_dir: Path = DEFAULT_VAULT_DIR) -> bool:
    return len(_load_admins_meta(vault_dir)) > 0 and (vault_dir / "admin_signatures.npz").exists()


def enrolled_admin_count(vault_dir: Path = DEFAULT_VAULT_DIR) -> int:
    return len(_load_admins_meta(vault_dir))


def list_enrolled_admins(vault_dir: Path = DEFAULT_VAULT_DIR) -> List[Dict[str, Any]]:
    return _load_admins_meta(vault_dir)


def _record_staff_audit(
    staff_member,
    event_type,
    timestamp_iso,
    shift_time="",
    capture_path=None,
    vault_dir=DEFAULT_VAULT_DIR,
    branch_name=None,
    pose_name=None,
):
    vault_dir.mkdir(parents=True, exist_ok=True)
    audit_path = vault_dir / "staff_audit_events.json"
    events = []
    if audit_path.exists():
        import json
        events = json.loads(audit_path.read_text(encoding="utf-8"))
    resolved_branch = (branch_name or staff_member.get("branch") or "").strip()
    resolved_pose = (pose_name or staff_member.get("pose_name") or "Captured Pose").strip()
    events.append(
        {
            "event": event_type,
            "staff_role": staff_member.get("role_label", "Staff"),
            "staff_name": staff_member.get("name", ""),
            "staff_email": staff_member.get("email", ""),
            "branch": resolved_branch,
            "shift_time": shift_time or staff_member.get("shift_time", ""),
            "pose_name": resolved_pose,
            "timestamp_utc": timestamp_iso,
            "arrival_capture_path": capture_path or f"sha256:{staff_member.get('pose_digest_sha256_16', '')}",
        }
    )
    import json
    audit_path.write_text(json.dumps(events, indent=2), encoding="utf-8")

def login_staff_member(
    staff_slot: str,
    staff_name: str,
    staff_email: str,
    shift_time: str,
    capture_image_bytes: Optional[bytes] = None,
    vault_dir: Path = DEFAULT_VAULT_DIR,
    branch_name: str = "Branch 1 - North Imaging Hub",
    pose_name: str = "Hand Pose Capture",
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Signs in an Employee (Staff 1 / Staff 2) with Name, Mail ID, 6-hour Shift (AM/PM),
    one of 6 Branches, and a live Hand Pose capture. Stores only a SHA-256 verification
    digest in audit logs—no raw camera frames are persisted to disk.
    """
    normalized_slot = "Staff 1" if str(staff_slot).strip() in ("1", "Staff 1", "staff_1") else "Staff 2"
    staff_id = "staff_1" if normalized_slot == "Staff 1" else "staff_2"

    clean_name = (staff_name or "").strip()
    clean_email = (staff_email or "").strip().lower()
    clean_shift = (shift_time or "").strip()
    clean_branch = (branch_name or "").strip()
    clean_pose_name = (pose_name or "Hand Pose Capture").strip() or "Hand Pose Capture"

    if not clean_name:
        return False, None, "Enter Employee Name first."
    if not clean_email or "@" not in clean_email:
        return False, None, "Enter a valid Employee Mail ID."
    if clean_email == PRIMARY_ADMIN_EMAIL:
        return False, None, f"Employee Mail ID must differ from Admin ({PRIMARY_ADMIN_EMAIL})."
    if not clean_shift or ("AM" not in clean_shift.upper() and "PM" not in clean_shift.upper()):
        return False, None, "Select a 6-hour shift time with AM/PM."
    if not clean_branch:
        return False, None, "Select one of the 6 branches."
    if capture_image_bytes is None:
        return False, None, "Capture your Hand Pose photo to sign in."

    pose_vec, pose_msg = extract_pose_signature(capture_image_bytes)
    if pose_vec is None:
        return False, None, f"Hand pose capture check failed: {pose_msg}"
    pose_digest = hashlib.sha256(capture_image_bytes).hexdigest()[:16]

    staff_list = _load_staff_meta(vault_dir)
    for existing in staff_list:
        if existing["email"].strip().lower() == clean_email and existing.get("id") != staff_id:
            return False, None, f"Email '{clean_email}' is already assigned to {existing.get('role_label')}."

    now = datetime.now(timezone.utc)
    staff_record = {
        "id": staff_id,
        "role_label": normalized_slot,
        "name": clean_name,
        "email": clean_email,
        "branch": clean_branch,
        "pose_name": clean_pose_name,
        "shift_time": clean_shift,
        "pose_verified": True,
        "pose_digest_sha256_16": pose_digest,
        "login_time_utc": now.isoformat(),
    }

    updated = [s for s in staff_list if s.get("id") != staff_id]
    updated.append(
        {
            "id": staff_id,
            "role_label": normalized_slot,
            "name": clean_name,
            "email": clean_email,
            "branch": clean_branch,
            "shift_time": clean_shift,
            "pose_verified": True,
            "pose_digest_sha256_16": pose_digest,
            "last_login_utc": now.isoformat(),
        }
    )
    updated.sort(key=lambda item: item["id"])
    _save_staff_meta(vault_dir, updated)

    _set_active_staff_session(vault_dir, staff_record)
    _record_staff_audit(
        staff_record,
        event_type="login",
        timestamp_iso=now.isoformat(),
        shift_time=f"{clean_branch} | {clean_shift}",
        capture_path=f"sha256:{pose_digest}",
        vault_dir=vault_dir,
        branch_name=clean_branch,
        pose_name=clean_pose_name,
    )
    return (
        True,
        staff_record,
        f"Signed in: {clean_name} ({clean_email}) • {clean_branch} • {clean_shift} • Hand Pose verified.",
    )

def logout_staff_session(
    staff_email: str,
    staff_name: str,
    staff_role: str = "Staff",
    shift_time: str = "",
    capture_path: str = "",
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> str:
    """Clears active staff session and records exact logout timestamp for Admin visibility."""
    logout_ts = datetime.now(timezone.utc).isoformat()
    _record_staff_audit(
        {
            "role_label": staff_role,
            "name": staff_name,
            "email": staff_email,
            "shift_time": shift_time,
            "arrival_capture_path": capture_path,
        },
        event_type="logout",
        timestamp_iso=logout_ts,
        shift_time=shift_time,
        capture_path=capture_path,
        vault_dir=vault_dir,
    )
    _set_active_staff_session(vault_dir, None)
    return logout_ts


def load_staff_audit_events(
    is_admin_unlocked: bool,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> List[Dict[str, Any]]:
    """Returns Staff 1 / Staff 2 login, shift, capture, and logout details strictly to unlocked Admin."""
    if not is_admin_unlocked:
        return []
    audit_path = vault_dir / "staff_audit_events.json"
    if not audit_path.exists():
        return []
    return json.loads(audit_path.read_text(encoding="utf-8"))


def enroll_admin_with_pose(
    face_image_bytes: bytes,
    pose_image_bytes: bytes,
    admin_label: str,
    admin_email: str,
    pose_note: str = "Hand gesture in lower frame",
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> Tuple[bool, str]:
    """Enrolls one of up to 2 permanent admins with face signature, pose signature, and email."""
    admins = _load_admins_meta(vault_dir)
    if len(admins) >= MAX_ADMINS:
        return False, f"Maximum of {MAX_ADMINS} permanent admins already enrolled."

    clean_email = admin_email.strip().lower()
    if not clean_email or "@" not in clean_email:
        return False, "Please provide a valid admin email address."

    if len(admins) == 0 and clean_email != PRIMARY_ADMIN_EMAIL:
        return False, f"Primary Admin #1 must be enrolled with {PRIMARY_ADMIN_EMAIL} first."

    if any(a["email"].strip().lower() == clean_email for a in admins):
        return False, f"Email {clean_email} is already enrolled."

    face_sig, face_msg = extract_face_signature(face_image_bytes)
    if face_sig is None:
        return False, f"Face check failed: {face_msg}"

    pose_sig, pose_msg = extract_pose_signature(pose_image_bytes)
    if pose_sig is None:
        return False, f"Pose check failed: {pose_msg}"

    slot = len(admins) + 1
    clean_name = admin_label.strip() or f"Admin {slot}"

    sig_path = vault_dir / "admin_signatures.npz"
    vault_dir.mkdir(parents=True, exist_ok=True)
    face_map: Dict[str, np.ndarray] = {}
    pose_map: Dict[str, np.ndarray] = {}
    if sig_path.exists():
        with np.load(sig_path) as existing:
            for k in existing.files:
                if k.startswith("face_"):
                    face_map[k] = existing[k].astype(np.float32)
                elif k.startswith("pose_"):
                    pose_map[k] = existing[k].astype(np.float32)

    key_suffix = f"admin_{slot}"
    face_map[f"face_{key_suffix}"] = face_sig
    pose_map[f"pose_{key_suffix}"] = pose_sig

    save_payload: Dict[str, np.ndarray] = {}
    save_payload.update(face_map)
    save_payload.update(pose_map)
    np.savez(sig_path, **save_payload)

    admins.append(
        {
            "id": key_suffix,
            "name": clean_name,
            "email": clean_email,
            "pose_note": pose_note.strip() or "Hand gesture in lower frame",
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save_admins_meta(vault_dir, admins)
    return True, f"Enrolled permanent Admin {slot}/{MAX_ADMINS}: {clean_name} ({clean_email})."


def _send_email_or_log(
    to_email: str,
    subject: str,
    body: str,
) -> str:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_user).strip()

    if smtp_host and smtp_from and to_email:
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = smtp_from
            msg["To"] = to_email
            msg.set_content(body)
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            return "sent_via_smtp"
        except Exception as exc:
            return f"smtp_error_local_fallback: {exc}"
    return "local_otp_console"


def generate_and_send_login_otp(
    login_email: str,
    ttl_minutes: int = 5,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> Tuple[bool, Optional[str], str]:
    """
    Generates a 5-character alphanumeric OTP and sends it directly to the enrolled
    admin email via SMTP. Never exposes the OTP code in the UI return message.
    """
    admins = _load_admins_meta(vault_dir)
    clean_email = (login_email or "").strip().lower()
    if not clean_email:
        return False, None, "Enter your admin email ID first (e.g., nisargahsn15@gmail.com)."

    matched_admins = [a for a in admins if a["email"].strip().lower() == clean_email]
    if not matched_admins:
        return False, None, f"Email '{clean_email}' is not an enrolled permanent admin."

    active_sess = get_active_admin_session(vault_dir=vault_dir)
    if active_sess is not None and active_sess["email"].strip().lower() != clean_email:
        return (
            False,
            None,
            f"Another admin ({active_sess['name']} • {active_sess['email']}) is currently logged in. They must logout first.",
        )

    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_from = os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "")).strip()
    if not smtp_host or not smtp_from:
        return (
            False,
            None,
            "SMTP email delivery is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, and SMTP_FROM to send the OTP directly to your email.",
        )

    alphabet = string.ascii_uppercase + string.digits
    code_5 = "".join(secrets.choice(alphabet) for _ in range(5))
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ttl_minutes)
    admin_info = matched_admins[0]

    delivery = _send_email_or_log(
        to_email=clean_email,
        subject="[Federated Medical DP Vault] Your 5-Digit Sign-In OTP",
        body=(
            f"Hello {admin_info['name']},\n\n"
            f"Your 5-character sign-in OTP is: {code_5}\n"
            f"Valid until (UTC): {expires_at.isoformat()}\n"
        ),
    )
    if delivery != "sent_via_smtp":
        return False, None, f"Failed to send OTP email to {clean_email} ({delivery}). Check SMTP credentials and try again."

    vault_dir.mkdir(parents=True, exist_ok=True)
    otp_path = vault_dir / "otp_challenge.json"
    otp_path.write_text(
        json.dumps(
            {
                "email": clean_email,
                "admin_name": admin_info["name"],
                "code": code_5,
                "issued_at_utc": now.isoformat(),
                "expires_at_utc": expires_at.isoformat(),
                "delivery": delivery,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return True, None, f"5-character OTP sent directly to {clean_email}. Check your inbox and enter the code below."


def verify_login_otp(
    login_email: str,
    entered_otp: str,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> Tuple[bool, str]:
    clean_email = (login_email or "").strip().lower()
    clean_code = (entered_otp or "").strip().upper()
    otp_path = vault_dir / "otp_challenge.json"
    if not otp_path.exists():
        return False, "No active OTP found. Click 'Send 5-Digit OTP' first."

    payload = json.loads(otp_path.read_text(encoding="utf-8"))
    if payload.get("email", "").strip().lower() != clean_email:
        return False, f"OTP was issued for {payload.get('email')}, not {clean_email}."

    expires_at = datetime.fromisoformat(payload["expires_at_utc"])
    if datetime.now(timezone.utc) > expires_at:
        return False, "OTP expired. Click 'Send 5-Digit OTP' to request a fresh code."

    if clean_code != payload.get("code", "").strip().upper():
        return False, "Invalid 5-character OTP. Check the code and try again."

    return True, f"5-character OTP verified for {clean_email}. Now complete Face & Hand Gesture security."


def send_admin_session_email(
    admin_email: str,
    admin_name: str,
    event_type: str,
    timestamp_iso: str,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> Tuple[bool, str]:
    vault_dir.mkdir(parents=True, exist_ok=True)
    audit_path = vault_dir / "audit_events.json"
    events: List[Dict[str, Any]] = []
    if audit_path.exists():
        events = json.loads(audit_path.read_text(encoding="utf-8"))

    subject = f"[Federated Medical DP Vault] Admin {event_type.upper()} — {admin_name}"
    body = (
        f"Hello {admin_name},\n\n"
        f"Admin {event_type.upper()} recorded for the Federated Medical DP Admin Vault.\n"
        f"Timestamp (UTC): {timestamp_iso}\n"
        f"Admin Email: {admin_email}\n"
    )
    delivery_status = _send_email_or_log(to_email=admin_email, subject=subject, body=body)

    events.append(
        {
            "event": event_type,
            "admin_name": admin_name,
            "admin_email": admin_email,
            "timestamp_utc": timestamp_iso,
            "delivery": delivery_status,
        }
    )
    audit_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    return True, f"Recorded {event_type} at {timestamp_iso} ({delivery_status})."


def verify_admin_face_and_pose(
    face_image_bytes: bytes,
    pose_image_bytes: bytes,
    login_email: Optional[str] = None,
    face_threshold: float = 0.65,
    pose_threshold: float = 0.40,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    admins = _load_admins_meta(vault_dir)
    sig_path = vault_dir / "admin_signatures.npz"
    if not admins or not sig_path.exists():
        return False, None, "No permanent admin enrolled yet."

    target_admins = admins
    if login_email is not None:
        clean_login_email = login_email.strip().lower()
        if not clean_login_email:
            return False, None, "Enter your authorized admin email address first."
        target_admins = [a for a in admins if a["email"].strip().lower() == clean_login_email]
        if not target_admins:
            return False, None, f"Email '{clean_login_email}' is not an enrolled permanent admin."

        active_sess = get_active_admin_session(vault_dir=vault_dir)
        if active_sess is not None and active_sess["email"].strip().lower() != clean_login_email:
            return (
                False,
                None,
                f"Another admin ({active_sess['name']} • {active_sess['email']}) is currently logged in. They must logout before you can sign in.",
            )

    face_sig, face_msg = extract_face_signature(face_image_bytes)
    if face_sig is None:
        return False, None, f"Face verification failed: {face_msg}"

    pose_sig, pose_msg = extract_pose_signature(pose_image_bytes)
    if pose_sig is None:
        return False, None, f"Pose verification failed: {pose_msg}"

    with np.load(sig_path) as data:
        best_match: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for admin in target_admins:
            aid = admin["id"]
            f_key = f"face_{aid}"
            p_key = f"pose_{aid}"
            if f_key not in data or p_key not in data:
                continue
            f_sim = cosine_similarity(face_sig, data[f_key])
            p_sim = cosine_similarity(pose_sig, data[p_key])
            combined = 0.6 * f_sim + 0.4 * p_sim
            if f_sim >= face_threshold and p_sim >= pose_threshold and combined > best_score:
                best_score = combined
                best_match = {
                    **admin,
                    "face_similarity": round(f_sim, 4),
                    "pose_similarity": round(p_sim, 4),
                }

    if best_match is not None:
        login_ts = datetime.now(timezone.utc).isoformat()
        send_admin_session_email(
            admin_email=best_match["email"],
            admin_name=best_match["name"],
            event_type="login",
            timestamp_iso=login_ts,
            vault_dir=vault_dir,
        )
        otp_path = vault_dir / "otp_challenge.json"
        if otp_path.exists():
            otp_path.unlink()
        best_match["login_time_utc"] = login_ts
        _set_active_admin_session(
            vault_dir,
            {
                "id": best_match["id"],
                "name": best_match["name"],
                "email": best_match["email"],
                "login_time_utc": login_ts,
            },
        )
        return (
            True,
            best_match,
            f"Verified {best_match['name']} ({best_match['email']}) • face={best_match['face_similarity']:.3f}, pose={best_match['pose_similarity']:.3f}. Login email alert recorded at {login_ts}.",
        )

    return False, None, "Face or hand pose signature did not match the entered admin email."


def logout_admin_session(
    admin_email: str,
    admin_name: str,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> str:
    logout_ts = datetime.now(timezone.utc).isoformat()
    send_admin_session_email(
        admin_email=admin_email,
        admin_name=admin_name,
        event_type="logout",
        timestamp_iso=logout_ts,
        vault_dir=vault_dir,
    )
    _set_active_admin_session(vault_dir, None)
    return logout_ts


def verify_admin_face(
    image_bytes: bytes,
    similarity_threshold: float = 0.75,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> Tuple[bool, float, str]:
    admins = _load_admins_meta(vault_dir)
    sig_path = vault_dir / "admin_signatures.npz"
    if not admins or not sig_path.exists():
        return False, 0.0, "No admin face enrolled yet."

    sig, msg = extract_face_signature(image_bytes)
    if sig is None:
        return False, 0.0, msg

    best_score = 0.0
    with np.load(sig_path) as data:
        for admin in admins:
            key = f"face_{admin['id']}"
            if key in data:
                best_score = max(best_score, cosine_similarity(sig, data[key]))

    if best_score >= similarity_threshold:
        return True, best_score, f"Admin verified (similarity={best_score:.3f})."
    return False, best_score, f"Face mismatch (similarity={best_score:.3f} < {similarity_threshold:.2f})."


def save_patient_record(
    record: Dict[str, Any],
    is_admin_unlocked: bool,
    vault_dir: Path = DEFAULT_VAULT_DIR,
    actor_info: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Persists complete staff login + uploaded scan + patient detail + segmentation result for Admin."""
    if not is_admin_unlocked and actor_info is None:
        return False, "Sign in as Admin or Staff 1 / Staff 2 before saving a diagnostic report."

    vault_dir.mkdir(parents=True, exist_ok=True)
    records_path = vault_dir / "client_records.json"
    existing: List[Dict[str, Any]] = []
    if records_path.exists():
        existing = json.loads(records_path.read_text(encoding="utf-8"))

    saved_by = (
        f"{actor_info.get('role_label', 'Staff')} — {actor_info.get('name', 'Staff')} ({actor_info.get('email', '')})"
        if actor_info
        else "Admin Vault"
    )
    entry: Dict[str, Any] = {
        **record,
        "saved_by": saved_by,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if actor_info:
        entry["staff_role"] = actor_info.get("role_label", "")
        entry["staff_name"] = actor_info.get("name", "")
        entry["staff_email"] = actor_info.get("email", "")
        entry["staff_shift_time"] = actor_info.get("shift_time", "")
        entry["staff_login_time_utc"] = actor_info.get("login_time_utc", "")
        entry["staff_arrival_capture_path"] = actor_info.get("arrival_capture_path", "")

    existing.append(entry)
    records_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return True, f"Submitted report for {record.get('patient_name', 'Unknown')} ({saved_by})."


def load_patient_records(
    is_admin_unlocked: bool,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> List[Dict[str, Any]]:
    """Previously stored patient, staff, and segmentation data is readable ONLY by an unlocked Admin."""
    if not is_admin_unlocked:
        return []
    records_path = vault_dir / "client_records.json"
    if not records_path.exists():
        return []
    return json.loads(records_path.read_text(encoding="utf-8"))


def load_audit_events(
    is_admin_unlocked: bool,
    vault_dir: Path = DEFAULT_VAULT_DIR,
) -> List[Dict[str, Any]]:
    if not is_admin_unlocked:
        return []
    audit_path = vault_dir / "audit_events.json"
    if not audit_path.exists():
        return []
    return json.loads(audit_path.read_text(encoding="utf-8"))


def build_paired_staff_sessions(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pairs employee login and logout events into per-session rows with branch,
    shift window, named hand pose, timestamps, and computed session duration.
    """
    from datetime import datetime

    def _parse_iso(ts: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

    def _infer_branch(ev: Dict[str, Any]) -> str:
        if ev.get("branch"):
            return str(ev["branch"])
        st_val = str(ev.get("shift_time", ""))
        if "|" in st_val:
            return st_val.split("|", 1)[0].strip()
        return "Unassigned Branch"

    def _infer_shift(ev: Dict[str, Any]) -> str:
        st_val = str(ev.get("shift_time", ""))
        if "|" in st_val:
            return st_val.split("|", 1)[1].strip()
        return st_val

    open_by_key: Dict[str, Dict[str, Any]] = {}
    paired_rows: List[Dict[str, Any]] = []

    for ev in events or []:
        etype = str(ev.get("event", "")).lower()
        key = str(ev.get("staff_email", "")).strip().lower() or str(ev.get("staff_name", ""))
        ts_raw = str(ev.get("timestamp_utc", ""))
        dt_obj = _parse_iso(ts_raw)
        branch_val = _infer_branch(ev)
        shift_val = _infer_shift(ev)
        pose_val = str(ev.get("pose_name") or "Captured Pose")

        if etype == "login":
            row: Dict[str, Any] = {
                "employee_name": ev.get("staff_name", "Staff"),
                "employee_email": ev.get("staff_email", ""),
                "branch": branch_val,
                "shift_window": shift_val,
                "hand_pose_name": pose_val,
                "login_time_utc": ts_raw,
                "logout_time_utc": "ACTIVE (Still Signed In)",
                "duration_minutes": None,
                "pose_digest": ev.get("arrival_capture_path", ""),
                "_login_dt": dt_obj,
            }
            paired_rows.append(row)
            open_by_key[key] = row
        elif etype == "logout":
            target = open_by_key.pop(key, None)
            if target is not None:
                target["logout_time_utc"] = ts_raw
                if target.get("_login_dt") and dt_obj:
                    target["duration_minutes"] = round((dt_obj - target["_login_dt"]).total_seconds() / 60.0, 2)
            else:
                paired_rows.append(
                    {
                        "employee_name": ev.get("staff_name", "Staff"),
                        "employee_email": ev.get("staff_email", ""),
                        "branch": branch_val,
                        "shift_window": shift_val,
                        "hand_pose_name": pose_val,
                        "login_time_utc": "—",
                        "logout_time_utc": ts_raw,
                        "duration_minutes": None,
                        "pose_digest": ev.get("arrival_capture_path", ""),
                        "_login_dt": None,
                    }
                )

    return [{k: v for k, v in r.items() if k != "_login_dt"} for r in paired_rows]
