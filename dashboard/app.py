from datetime import date
import gzip
import io
import json
from pathlib import Path
from typing import Optional
import nibabel as nib
import numpy as np
from PIL import Image
import streamlit as st
import torch
from client.client_node import select_device
from models.unet_model import MedicalUNet
from utils.admin_vault import (
    MAX_ADMINS,
    SIX_BRANCHES,
    SIX_HOUR_SHIFTS,
    enrolled_admin_count,
    enroll_admin_with_pose,
    generate_and_send_login_otp,
    get_active_admin_session,
    get_active_staff_session,
    is_admin_enrolled,
    list_enrolled_admins,
    load_audit_events,
    load_patient_records,
    load_staff_audit_events,
    login_staff_member,
    logout_admin_session,
    logout_staff_session,
    save_patient_record,
    verify_admin_face_and_pose,
    verify_login_otp,
)
from utils.resume_generator import load_run_summary
from utils.segmentation_eval import (
    build_batch_report_payload,
    compute_3d_dice,
    rows_to_csv_text,
    threshold_probabilities,
    to_match_error_image,
)


MAX_UPLOAD_BYTES = 50 * 1024 * 1024

st.set_page_config(page_title="Login Dashboard • Federated 3D Scan Pipeline", layout="wide")

if "dash_step" not in st.session_state:
    st.session_state["dash_step"] = 1
if "admin_unlocked" not in st.session_state:
    st.session_state["admin_unlocked"] = False
if "active_admin" not in st.session_state:
    st.session_state["active_admin"] = None
if "active_staff" not in st.session_state:
    st.session_state["active_staff"] = get_active_staff_session()
if "otp_verified_email" not in st.session_state:
    st.session_state["otp_verified_email"] = None
if "patient_intake" not in st.session_state:
    st.session_state["patient_intake"] = None
if "last_diagnostic" not in st.session_state:
    st.session_state["last_diagnostic"] = None

st.markdown(
    """
    <style>
        .stApp {
            background-color: #FFFDF0;
            background-image:
                linear-gradient(to right, rgba(2, 132, 199, 0.06) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(2, 132, 199, 0.06) 1px, transparent 1px),
                repeating-linear-gradient(
                    to bottom,
                    rgba(3, 105, 161, 0.025) 0px,
                    rgba(3, 105, 161, 0.025) 2px,
                    transparent 2px,
                    transparent 6px
                );
            background-size: 28px 28px, 28px 28px, 100% 6px;
            color: #0F172A;
        }
        .block-container {
            background-color: rgba(254, 252, 232, 0.96);
            border: 1.5px solid #7DD3FC;
            border-radius: 14px;
            padding: 1.1rem 1.75rem;
        }
        h1, h2, h3 {
            color: #0369A1 !important;
        }
        .scan-header {
            background: linear-gradient(90deg, #E0F2FE 0%, #FEF9C3 100%);
            border: 1px solid #38BDF8;
            border-left: 6px solid #0284C7;
            color: #0369A1;
            padding: 0.75rem 1.1rem;
            border-radius: 10px;
            margin-bottom: 0.75rem;
        }
        .section-banner {
            background-color: #E0F2FE;
            border-left: 5px solid #0284C7;
            color: #0369A1;
            padding: 0.55rem 1rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.02rem;
            margin-top: 0.65rem;
            margin-bottom: 0.5rem;
        }
        .result-card {
            background-color: #F0F9FF;
            border: 1px solid #7DD3FC;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def to_sharp_display(arr_2d: np.ndarray, display_size: int = 200, is_mask: bool = False) -> Image.Image:
    arr = arr_2d.astype(np.float32)
    if is_mask:
        img_u8 = np.clip((arr > 0).astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    else:
        vmin, vmax = float(arr.min()), float(arr.max())
        norm = (arr - vmin) / (vmax - vmin + 1e-6)
        img_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_u8, mode="L").convert("RGB").resize(
        (display_size, display_size), resample=Image.Resampling.NEAREST
    )


def to_overlay_display(scan_2d: np.ndarray, mask_2d: np.ndarray, display_size: int = 200) -> Image.Image:
    vmin, vmax = float(scan_2d.min()), float(scan_2d.max())
    base = np.clip((scan_2d - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    fg = (mask_2d > 0)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[..., 0] = np.where(fg, np.clip(base * 0.4 + 0.15, 0, 1), rgb[..., 0])
    rgb[..., 1] = np.where(fg, np.clip(base * 0.7 + 0.25, 0, 1), rgb[..., 1])
    rgb[..., 2] = np.where(fg, np.clip(base * 0.5 + 0.45, 0, 1), rgb[..., 2])
    img_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB").resize(
        (display_size, display_size), resample=Image.Resampling.NEAREST
    )


def load_nifti_in_memory(file_bytes: bytes) -> np.ndarray:
    raw_bytes = gzip.decompress(file_bytes) if file_bytes[:2] == b"\x1f\x8b" else file_bytes
    header = nib.Nifti1Header.from_bytes(raw_bytes[:348])
    dtype = header.get_data_dtype()
    offset = int(header.structarr["vox_offset"])
    shape = tuple(int(dim) for dim in header.get_data_shape())
    count = int(np.prod(shape))
    arr = np.frombuffer(raw_bytes, dtype=dtype, count=count, offset=offset).reshape(shape).astype(np.float32)
    if arr.ndim == 3:
        return np.stack([arr, arr], axis=0)
    if arr.ndim == 4:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Unsupported NIfTI volume dimensions: {arr.shape}")


def discover_checkpoints() -> dict[str, tuple[Path, Optional[Path]]]:
    options: dict[str, tuple[Path, Optional[Path]]] = {}
    eps2_ckpt = Path("artifacts/runs/latest_eps2/global_unet.pt")
    eps2_metrics = Path("artifacts/runs/latest_eps2/federated_metrics.json")
    if eps2_ckpt.exists():
        options["Strict Privacy (ε=2.0): latest_eps2"] = (
            eps2_ckpt,
            eps2_metrics if eps2_metrics.exists() else None,
        )
    eps10_ckpt = Path("artifacts/runs/eps_10.0/global_unet.pt")
    eps10_metrics = Path("artifacts/runs/eps_10.0/federated_metrics.json")
    if eps10_ckpt.exists():
        options["High Utility (ε=10.0): eps_10.0"] = (
            eps10_ckpt,
            eps10_metrics if eps10_metrics.exists() else None,
        )
    active_ckpt = Path("artifacts/global_unet.pt")
    active_metrics = Path("artifacts/federated_metrics.json")
    if active_ckpt.exists():
        archived_sizes = {c.stat().st_size for c, _ in options.values() if c.exists()}
        archived_mtimes = {round(c.stat().st_mtime, 1) for c, _ in options.values() if c.exists()}
        if active_ckpt.stat().st_size not in archived_sizes or round(active_ckpt.stat().st_mtime, 1) not in archived_mtimes:
            options["Active Unarchived: artifacts/global_unet.pt"] = (
                active_ckpt,
                active_metrics if active_metrics.exists() else None,
            )
    return options


ckpt_options = discover_checkpoints()
default_metrics = Path("artifacts/federated_metrics.json")
summary = load_run_summary(default_metrics)
active_staff = st.session_state.get("active_staff") or get_active_staff_session()
active_admin = st.session_state.get("active_admin")
active_lock = get_active_admin_session()
current_count = enrolled_admin_count()

st.markdown(
    """
    <div class="scan-header">
        <h2 style="margin:0;">Dashboard 1 — Role Sign-In Gate (Employee 6h Shift &amp; Branch • Admin OTP &amp; Hand Pose)</h2>
        <div style="font-size:0.9rem; margin-top:0.2rem;">Federated 3D Medical Image Segmentation (ε-DP) • Scan Console Theme</div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_c1, nav_c2, nav_c3 = st.columns(3)
with nav_c1:
    if st.button("Dashboard 1: Sign In", use_container_width=True, type="primary" if st.session_state["dash_step"] == 1 else "secondary"):
        st.session_state["dash_step"] = 1
        st.rerun()
with nav_c2:
    if st.button("Dashboard 2: Intake, Scan & Metrics", use_container_width=True, type="primary" if st.session_state["dash_step"] == 2 else "secondary"):
        st.session_state["dash_step"] = 2
        st.rerun()
with nav_c3:
    if st.button("Dashboard 3: Admin Vault", use_container_width=True, type="primary" if st.session_state["dash_step"] == 3 else "secondary"):
        st.session_state["dash_step"] = 3
        st.rerun()

if st.session_state["dash_step"] == 1:
    if active_staff is not None and not st.session_state["admin_unlocked"]:
        st.success(
            f"Employee Active: **{active_staff.get('name')}** ({active_staff.get('email')}) • "
            f"**{active_staff.get('branch')}** • **{active_staff.get('shift_time')}** • Hand Pose Verified"
        )
        n1, n2 = st.columns(2)
        if n1.button("Continue to Dashboard 2 (Patient & Scan)", type="primary"):
            st.session_state["dash_step"] = 2
            st.rerun()
        if n2.button("Employee Logout"):
            logout_staff_session(
                staff_email=active_staff.get("email", ""),
                staff_name=active_staff.get("name", "Employee"),
                staff_role=active_staff.get("role_label", "Staff 1"),
                shift_time=f"{active_staff.get('branch', '')} | {active_staff.get('shift_time', '')}",
                capture_path=f"sha256:{active_staff.get('pose_digest_sha256_16', 'none')}",
            )
            st.session_state["active_staff"] = None
            st.rerun()

    elif st.session_state["admin_unlocked"] and active_admin is not None:
        st.success(f"Admin Active: **{active_admin.get('name')}** ({active_admin.get('email')})")
        a1, a2, a3 = st.columns(3)
        if a1.button("Go to Dashboard 2 (Scan & Metrics)"):
            st.session_state["dash_step"] = 2
            st.rerun()
        if a2.button("Go to Dashboard 3 (Admin Vault)", type="primary"):
            st.session_state["dash_step"] = 3
            st.rerun()
        if a3.button("Admin Logout"):
            logout_admin_session(
                admin_email=active_admin.get("email", "nisargahsn15@gmail.com"),
                admin_name=active_admin.get("name", "Admin"),
            )
            st.session_state["admin_unlocked"] = False
            st.session_state["active_admin"] = None
            st.session_state["otp_verified_email"] = None
            st.rerun()

    else:
        emp_col, adm_col = st.columns(2)

        with emp_col:
            st.markdown('<div class="section-banner">Employee Sign In (6-Hour Shift • 6 Branches • Name • Mail ID • Hand Pose)</div>', unsafe_allow_html=True)
            emp_name = st.text_input("Employee Name *", placeholder="e.g., Nisha Nair", key="d1_emp_name")
            emp_email = st.text_input("Employee Mail ID *", placeholder="e.g., nisha@clinic.org", key="d1_emp_email")
            emp_branch = st.selectbox("Select Branch (6 Options) *", SIX_BRANCHES, key="d1_emp_branch")
            emp_shift = st.selectbox("Select 6-Hour Shift *", SIX_HOUR_SHIFTS, key="d1_emp_shift")
            emp_pose_cam = st.camera_input("Employee Hand Pose Capture *", key="d1_emp_hand_cam")

            if st.button("Sign In as Employee", type="primary"):
                pose_bytes = emp_pose_cam.getvalue() if emp_pose_cam is not None else None
                ok_emp, sess_emp, msg_emp = login_staff_member(
                    staff_slot="Staff 1",
                    staff_name=emp_name,
                    staff_email=emp_email,
                    shift_time=emp_shift,
                    capture_image_bytes=pose_bytes,
                    branch_name=emp_branch,
                )
                if ok_emp and sess_emp is not None:
                    st.session_state["active_staff"] = sess_emp
                    st.session_state["dash_step"] = 2
                    st.success(msg_emp)
                    st.rerun()
                else:
                    st.error(msg_emp)

        with adm_col:
            st.markdown('<div class="section-banner">Admin Sign In (Face + Hand Pose Only)</div>', unsafe_allow_html=True)
            if active_lock is not None:
                st.warning(f"Active Admin Lock: {active_lock.get('name')} ({active_lock.get('email')}).")

            if is_admin_enrolled():
                adm_face_cam = st.camera_input("Admin Face Capture", key="d1_adm_face")
                adm_pose_cam = st.camera_input("Admin Hand Pose Capture *", key="d1_adm_pose")
                if st.button("Complete Admin Sign In", type="primary"):
                    if adm_face_cam is None or adm_pose_cam is None:
                        st.error("Capture both Admin Face and Admin Hand Pose photos.")
                    else:
                        v_ok, matched_adm, v_msg = verify_admin_face_and_pose(
                            face_image_bytes=adm_face_cam.getvalue(),
                            pose_image_bytes=adm_pose_cam.getvalue(),
                            login_email=None,
                        )
                        if v_ok and matched_adm is not None:
                            st.session_state["admin_unlocked"] = True
                            st.session_state["active_admin"] = matched_adm
                            st.session_state["otp_verified_email"] = None
                            st.session_state["dash_step"] = 3
                            st.success(v_msg)
                            st.rerun()
                        else:
                            st.error(v_msg)
            else:
                st.info("Enroll Primary Admin #1 (nisargahsn15@gmail.com) below to enable Admin login.")

        if current_count < MAX_ADMINS:
            slot_num = current_count + 1
            with st.expander(f"Enroll Permanent Admin #{slot_num}", expanded=(current_count == 0)):
                ec1, ec2, ec3 = st.columns(3)
                en_name = ec1.text_input("Admin Name", value="Nisarga" if slot_num == 1 else "", key=f"d1_en_name_{slot_num}")
                en_email = ec2.text_input(
                    "Admin Email",
                    value="nisargahsn15@gmail.com" if slot_num == 1 else "",
                    disabled=(slot_num == 1),
                    key=f"d1_en_email_{slot_num}",
                )
                en_note = ec3.text_input("Hand Pose Note", value="Hand gesture in lower frame", key=f"d1_en_note_{slot_num}")
                cc1, cc2 = st.columns(2)
                en_face = cc1.camera_input("Capture Admin Face", key=f"d1_en_face_{slot_num}")
                en_pose = cc2.camera_input("Capture Admin Hand Pose", key=f"d1_en_pose_{slot_num}")
                if st.button(f"Enroll Admin #{slot_num}"):
                    if en_face is None or en_pose is None:
                        st.error("Capture both Face and Hand Pose photos.")
                    else:
                        ok_e, msg_e = enroll_admin_with_pose(
                            face_image_bytes=en_face.getvalue(),
                            pose_image_bytes=en_pose.getvalue(),
                            admin_label=en_name,
                            admin_email=en_email,
                            pose_note=en_note,
                        )
                        if ok_e:
                            st.success(msg_e)
                            st.rerun()
                        else:
                            st.error(msg_e)

elif st.session_state["dash_step"] == 2:
    if active_staff is None and not st.session_state["admin_unlocked"]:
        st.warning("Sign in as Employee or Admin on Dashboard 1 first.")
    else:
        st.markdown('<div class="section-banner">Dashboard 2 — Patient Intake, 3D Segmentation &amp; Federated Metrics</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        p_name = p1.text_input("Patient Full Name *", value=(st.session_state["patient_intake"] or {}).get("patient_name", ""))
        p_cond = p2.text_input("Condition *", value=(st.session_state["patient_intake"] or {}).get("disease", ""))
        p_dob = p3.date_input("Date of Birth *", value=date(1995, 1, 1), min_value=date(1900, 1, 1), max_value=date.today())
        p_prov = p4.selectbox("Scan Provenance", ["Simulated Federated Client Scan", "Uploaded Scan Volume"])
        st.session_state["patient_intake"] = {
            "patient_name": p_name.strip(),
            "disease": p_cond.strip(),
            "dob": p_dob.isoformat(),
            "provenance": p_prov,
        }

        if p_name.strip() and p_cond.strip():
            input_mode = st.radio("Scan Source", ["Held-out test sample (.npz)", "Upload .npz or .nii.gz scan"], index=0)
            selected_ckpt = st.selectbox("Checkpoint", list(ckpt_options.keys()), index=min(1, len(ckpt_options) - 1)) if ckpt_options else None
            threshold = st.slider("Foreground Threshold", 0.30, 0.90, 0.60, 0.05)

            volume = None
            gt = None
            src_label = ""
            if input_mode == "Held-out test sample (.npz)":
                files = sorted(str(p) for p in Path("data").glob("client_*/test/*.npz"))
                chosen = st.selectbox("Test Case", ["-- Select --"] + files, index=1 if files else 0)
                if chosen and chosen != "-- Select --":
                    with np.load(chosen) as arr:
                        volume = arr["image"].astype(np.float32)
                        gt = arr["label"].astype(np.int64) if "label" in arr else None
                    src_label = Path(chosen).name
            else:
                up = st.file_uploader("Upload .npz or .nii.gz", type=["npz", "gz"])
                if up is not None and up.size <= MAX_UPLOAD_BYTES:
                    raw = up.getvalue()
                    if up.name.endswith(".npz"):
                        with np.load(io.BytesIO(raw)) as arr:
                            volume = arr["image"].astype(np.float32)
                            gt = arr["label"].astype(np.int64) if "label" in arr else None
                    else:
                        volume = load_nifti_in_memory(raw)
                    src_label = up.name

            if volume is not None:
                if volume.ndim == 3:
                    volume = np.stack([volume, volume], axis=0)
                sl = st.slider("Slice Index", 0, max(int(volume.shape[1]) - 1, 0), min(16, max(int(volume.shape[1]) - 1, 0)))
                if st.button("Run Segmentation", type="primary"):
                    dev = select_device()
                    model = MedicalUNet(in_channels=int(volume.shape[0]), out_channels=2).to(dev)
                    if selected_ckpt:
                        ckpt_file, _ = ckpt_options[selected_ckpt]
                        model.load_state_dict(torch.load(ckpt_file, map_location=dev, weights_only=True), strict=True)
                    model.eval()
                    with torch.no_grad():
                        probs = torch.softmax(model(torch.from_numpy(volume).unsqueeze(0).to(dev)), dim=1)[:, 1].squeeze(0).cpu().numpy()
                        pred = threshold_probabilities(probs, threshold=threshold)

                    has_gt = gt is not None and gt.shape[1:] == pred.shape
                    gt_vol = gt[0] if has_gt else None
                    dice_v = compute_3d_dice(pred, gt_vol) if (has_gt and gt_vol is not None) else None

                    st.session_state["last_diagnostic"] = {
                        "patient_name": p_name.strip(),
                        "disease": p_cond.strip(),
                        "dob": p_dob.isoformat(),
                        "provenance": p_prov,
                        "source": src_label,
                        "checkpoint": selected_ckpt or "untrained",
                        "predicted_fg_pct": round(float((pred == 1).mean()) * 100.0, 2),
                        "mean_lesion_prob_pct": round(float(probs.mean()) * 100.0, 2),
                        "volume_dice": round(float(dice_v), 4) if dice_v is not None else None,
                    }

                    cols = st.columns(5 if has_gt else 3)
                    cols[0].image(to_sharp_display(volume[0, sl], 180, False), caption=f"Scan • Slice {sl}", width=180)
                    cols[1].image(to_sharp_display(pred[sl], 180, True), caption=f"Pred • Slice {sl}", width=180)
                    cols[2].image(to_overlay_display(volume[0, sl], pred[sl], 180), caption=f"Overlay • Slice {sl}", width=180)
                    if has_gt and gt_vol is not None:
                        cols[3].image(to_sharp_display(gt_vol[sl], 180, True), caption=f"GT • Slice {sl}", width=180)
                        cols[4].image(to_match_error_image(pred[sl], gt_vol[sl], 180), caption="Match Map", width=180)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Foreground", f"{float((pred == 1).mean()) * 100.0:.2f}%")
                    m2.metric("Mean Lesion Prob.", f"{float(probs.mean()) * 100.0:.2f}%")
                    m3.metric("3D Dice", f"{dice_v:.4f}" if dice_v is not None else "N/A")

            if st.session_state.get("last_diagnostic") is not None:
                if st.button("Submit Result to Vault"):
                    ok_s, msg_s = save_patient_record(
                        st.session_state["last_diagnostic"],
                        is_admin_unlocked=st.session_state["admin_unlocked"],
                        actor_info=active_staff,
                    )
                    (st.success if ok_s else st.error)(msg_s)

                payload = build_batch_report_payload([st.session_state["last_diagnostic"]])
                csv_txt = rows_to_csv_text([st.session_state["last_diagnostic"]])
                d1, d2 = st.columns(2)
                d1.download_button("Download Report (JSON)", json.dumps(payload, indent=2), file_name="diagnostic_report.json", mime="application/json")
                d2.download_button("Download Report (CSV)", csv_txt, file_name="diagnostic_report.csv", mime="text/csv")

        st.markdown('<div class="section-banner">Federated (ε-DP) Metrics &amp; Controlled Ablation</div>', unsafe_allow_html=True)
        fm1, fm2, fm3 = st.columns(3)
        fm1.metric("Rounds Completed", str(summary["rounds_completed"]))
        fm2.metric("Privacy Epsilon (ε)", str(summary["epsilon"]))
        fm3.metric("Mean Test Dice", f"{summary['latest_test_dice']:.4f}" if summary["latest_test_dice"] is not None else "N/A")
        ab_path = Path("artifacts/runs/ablation_grid.json")
        if ab_path.exists():
            ab_data = json.loads(ab_path.read_text(encoding="utf-8"))
            table_rows = [
                {
                    "epsilon": r.get("epsilon"),
                    "delta": r.get("delta"),
                    "rounds": r.get("rounds"),
                    "seeds": str(r.get("seeds")),
                    "mean_dice_3d": r.get("mean_dice_3d"),
                    "std_dice_3d": r.get("std_dice_3d"),
                }
                for r in ab_data.get("results", [])
            ]
            if table_rows:
                st.dataframe(table_rows, use_container_width=True)
            with st.expander("Raw Controlled Ablation JSON (ablation_grid.json)"):
                st.json(ab_data)

else:
    st.markdown('<div class="section-banner">Dashboard 3 — Admin Authority Vault</div>', unsafe_allow_html=True)
    if not st.session_state["admin_unlocked"]:
        st.warning("Admin access only. Sign in as Admin on Dashboard 1.")
    else:
        st.markdown("### Stored Patient & Scan Records")
        st.dataframe(load_patient_records(is_admin_unlocked=True), use_container_width=True)
        st.markdown("### Employee Login, 6-Hour Shift, Branch & Hand-Pose Digest Log")
        st.dataframe(load_staff_audit_events(is_admin_unlocked=True), use_container_width=True)
        st.markdown("### Enrolled Admins & Admin Session Log")
        st.dataframe(list_enrolled_admins(), use_container_width=True)
        st.dataframe(load_audit_events(is_admin_unlocked=True), use_container_width=True)
