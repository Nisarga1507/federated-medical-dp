from pathlib import Path
import json
import nibabel as nib
import numpy as np
from PIL import Image
import streamlit as st
import torch
from client.client_node import select_device
from models.unet_model import MedicalUNet
from utils.resume_generator import generate_cv_bullets, load_run_summary


st.set_page_config(page_title="Federated Medical DP Diagnostics", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #FFFDF0;
            color: #0F172A;
        }
        .block-container {
            background-color: #FEFCE8;
            border: 1px solid #FDE047;
            border-radius: 14px;
            padding: 1rem 1.75rem;
        }
        h1, h2, h3 {
            color: #0369A1 !important;
        }
        .section-banner {
            background-color: #E0F2FE;
            border-left: 5px solid #0284C7;
            color: #0369A1;
            padding: 0.65rem 1rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.08rem;
            margin-top: 0.65rem;
            margin-bottom: 0.45rem;
        }
        .result-card {
            background-color: #F0F9FF;
            border: 1px solid #BAE6FD;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def to_sharp_display(arr_2d: np.ndarray, display_size: int = 256, is_mask: bool = False) -> Image.Image:
    arr = arr_2d.astype(np.float32)
    if is_mask:
        img_u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        vmin, vmax = float(arr.min()), float(arr.max())
        norm = (arr - vmin) / (vmax - vmin + 1e-6)
        img_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_u8, mode="L").convert("RGB")
    return pil_img.resize((display_size, display_size), resample=Image.Resampling.NEAREST)


def to_overlay_image(scan_2d: np.ndarray, mask_2d: np.ndarray, display_size: int = 256) -> Image.Image:
    arr = scan_2d.astype(np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    base_rgb = np.stack([norm, norm, norm], axis=-1)

    fg = (mask_2d == 1)[..., None]
    # Highlight predicted lesion region in crisp cyan-blue tint while preserving scan structure
    overlay = base_rgb.copy()
    overlay = np.where(fg, 0.55 * base_rgb + 0.45 * np.array([0.05, 0.65, 0.98]), overlay)
    img_u8 = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB").resize((display_size, display_size), resample=Image.Resampling.NEAREST)


st.title("Federated Learning Multi-Modal Diagnostic Pipeline (ε-DP)")
st.caption("Decentralized MONAI 3D UNet • Flower FedAvg • Differential Privacy • CV Summary Generator")

metrics_path = Path("artifacts/federated_metrics.json")
ckpt_path = Path("artifacts/global_unet.pt")
summary = load_run_summary(metrics_path)

tab_infer, tab_metrics, tab_cv = st.tabs([
    "1. Option → Images → Results",
    "2. Federated & Privacy Metrics",
    "3. Resume / CV Generator",
])

with tab_infer:
    st.markdown('<div class="section-banner">Step 1: Choose Your Input Option</div>', unsafe_allow_html=True)

    input_option = st.radio(
        "Select how you want to provide a scan:",
        [
            "-- Choose an option --",
            "Held-out test sample (.npz)",
            "Upload .npz or .nii.gz scan",
        ],
        index=0,
    )

    volume = None
    ground_truth = None
    source_label = ""

    if input_option == "Held-out test sample (.npz)":
        test_files = sorted(Path("data").glob("client_*/test/*.npz"))
        if test_files:
            file_choices = ["-- Select a test scan --"] + [str(p) for p in test_files]
            chosen_path = st.selectbox("Choose a test case from your split:", file_choices, index=0)
            if chosen_path != "-- Select a test scan --":
                with np.load(chosen_path) as arr:
                    volume = arr["image"].astype(np.float32)
                    ground_truth = arr["label"].astype(np.int64) if "label" in arr else None
                source_label = chosen_path
        else:
            st.warning("No test `.npz` files found yet. Run the server and clients first to create splits.")

    elif input_option == "Upload .npz or .nii.gz scan":
        uploaded = st.file_uploader("Upload a `.npz` (with `image`) or `.nii.gz` scan:", type=["npz", "gz"])
        if uploaded is not None:
            if uploaded.name.endswith(".npz"):
                with np.load(uploaded) as arr:
                    volume = arr["image"].astype(np.float32)
                    ground_truth = arr["label"].astype(np.int64) if "label" in arr else None
            else:
                temp_file = Path("artifacts/_tmp_upload.nii.gz")
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                temp_file.write_bytes(uploaded.getvalue())
                raw_img = nib.load(str(temp_file)).get_fdata().astype(np.float32)
                temp_file.unlink(missing_ok=True)
                if raw_img.ndim == 3:
                    volume = np.stack([raw_img, raw_img], axis=0)
                elif raw_img.ndim == 4:
                    volume = np.moveaxis(raw_img, -1, 0)
            source_label = uploaded.name
    else:
        st.info("Choose an input option above to load scan images and view segmentation results.")

    if volume is not None:
        if volume.ndim == 3:
            volume = np.stack([volume, volume], axis=0)

        in_ch = int(volume.shape[0])
        depth = int(volume.shape[1])
        slice_idx = st.slider(
            "Select Axial / Depth Slice",
            min_value=0,
            max_value=max(depth - 1, 0),
            value=depth // 2,
        )

        if st.button("Show Images & Segmentation Results", type="primary"):
            st.session_state["last_inference_ready"] = True

        if st.session_state.get("last_inference_ready"):
            device = select_device()
            model = MedicalUNet(in_channels=in_ch, out_channels=2).to(device)
            model_loaded = ckpt_path.exists()
            if model_loaded:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
            model.eval()

            with torch.no_grad():
                tensor_in = torch.from_numpy(volume).unsqueeze(0).to(device)
                logits = model(tensor_in)
                probs = torch.softmax(logits, dim=1)[:, 1].squeeze(0).cpu().numpy()
                pred_mask = (probs >= 0.5).astype(np.int64)

            st.markdown('<div class="section-banner">Step 2: Scan & Segmentation Images</div>', unsafe_allow_html=True)
            num_cols = 4 if ground_truth is not None else 3
            cols = st.columns(num_cols)

            cols[0].image(
                to_sharp_display(volume[0, slice_idx], display_size=192, is_mask=False),
                caption=f"Input Scan (Ch 0) • Slice {slice_idx}",
                width=192,
            )
            cols[1].image(
                to_sharp_display(pred_mask[slice_idx], display_size=192, is_mask=True),
                caption=f"Predicted Mask • Slice {slice_idx}",
                width=192,
            )
            cols[2].image(
                to_overlay_image(volume[0, slice_idx], pred_mask[slice_idx], display_size=192),
                caption=f"Prediction Overlay • Slice {slice_idx}",
                width=192,
            )

            gt_dice = None
            if ground_truth is not None:
                cols[3].image(
                    to_sharp_display(ground_truth[0, slice_idx], display_size=192, is_mask=True),
                    caption=f"Ground Truth Mask • Slice {slice_idx}",
                    width=192,
                )
                pred_fg = (pred_mask == 1)
                gt_fg = (ground_truth[0] == 1)
                denom = int(pred_fg.sum() + gt_fg.sum())
                gt_dice = 1.0 if denom == 0 else (2.0 * int((pred_fg & gt_fg).sum())) / denom

            st.markdown('<div class="section-banner">Step 3: Diagnostic & Privacy Results</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            fg_pct = float((pred_mask == 1).mean()) * 100.0
            mean_prob = float(probs.mean()) * 100.0
            m1.metric("Selected Source", Path(source_label).name)
            m2.metric("Predicted Foreground", f"{fg_pct:.2f}%")
            m3.metric("Mean Lesion Prob.", f"{mean_prob:.2f}%")
            if gt_dice is not None:
                m4.metric("3D Volume Dice", f"{gt_dice:.4f}")
            else:
                m4.metric("Privacy Budget (ε)", f"{summary['epsilon']}")

            if not model_loaded:
                st.caption("Note: `artifacts/global_unet.pt` was not found; showing untrained model weights.")
            st.markdown('</div>', unsafe_allow_html=True)

with tab_metrics:
    st.markdown('<div class="section-banner">Federated Learning & Differential Privacy Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed Rounds", str(summary["rounds_completed"]))
    c2.metric("Privacy Epsilon (ε)", f"{summary['epsilon']}")
    c3.metric("Privacy Delta (δ)", f"{summary['delta']:.0e}")
    c4.metric(
        "Mean Test Dice",
        f"{summary['latest_test_dice']:.4f}" if summary["latest_test_dice"] is not None else "Pending",
    )

    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text()))
    else:
        st.info("Run `run_server.py` with two `run_client.py` nodes to populate `artifacts/federated_metrics.json`.")

with tab_cv:
    st.markdown('<div class="section-banner">Resume / CV Bullet Points (Populated From Run Metrics)</div>', unsafe_allow_html=True)
    bullets = generate_cv_bullets(summary)
    for b in bullets:
        st.markdown(f"* {b}")

    st.text_area("Copy-Ready CV Bullets", value="\n".join(f"* {b}" for b in bullets), height=220)
