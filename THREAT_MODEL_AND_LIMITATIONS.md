# Threat Model, Scope & Limitations — Federated 3D Medical Segmentation (`federated-medical-dp`)

## 1. Accurate System Scope
- **Diagnostic Pipeline:** A **federated 3D medical image segmentation pipeline** (`MedicalUNet`) trained across simulated federated clients (`client_1`, `client_2`) using Flower (`FedAvg`) and $(\epsilon, \delta)$-differential privacy ($\delta = 10^{-5}$, gradient clipping norm `1.0`). It is not a multi-modal clinical model unless distinct modalities (such as MRI + CT or imaging + structured clinical features) are fused directly inside the segmentation network.
- **Data & Scale Claims:** Client splits in this workspace represent **simulated federated clients**, not verified independent hospitals. Scaling to **10,000+ scans** is an ongoing/planned engineering target rather than a completed evaluation claim.
- **Access-Control Prototype Boundary:** 5-character email OTP and OpenCV face/hand-pose checks demonstrate a **role-based access-control prototype** for dashboard workflows; they do not constitute clinical-grade or production biometric identity verification.

## 2. Assets & Trust Boundaries
- **Protected Artifacts:** Stored diagnostic records (`artifacts/admin_vault/client_records.json`), admin/staff audit trails (`audit_events.json`, `staff_audit_events.json`), active session locks, and enrolled signature files (`admin_signatures.npz`).
- **Role Boundaries:**
  - **Unauthenticated Users:** Cannot read stored vault records or staff/admin audit logs.
  - **Staff 1 / Staff 2:** Sign in with name, unique email ID, AM/PM shift window, and live hand-pose check for current shift evaluation; cannot read prior stored vault history.
  - **Admin (`nisargahsn15@gmail.com` primary, up to 2 admins, single active session):** Uses email OTP + face & hand-pose verification and holds sole read/audit authority over stored vault data.

## 3. Threat Assumptions & Failure Modes
| Threat / Failure Case | Risk / Impact | Mitigation & Current Limitation |
| :--- | :--- | :--- |
| **Lighting or Pose Drift** | Poor lighting, camera angle, or background clutter can degrade OpenCV feature similarity and cause false rejections or brittle matches. | Hand-pose verification crops the lower gesture region and normalizes vectors; prototype threshold tuning is needed before any real-world deployment. |
| **Unnecessary Biometric Image Retention** | Saving raw camera frames on disk increases privacy exposure if local artifacts are leaked. | Staff arrival checks compute an in-memory SHA-256 verification digest without persisting raw staff capture images to disk. |
| **Concurrent Admin Sessions** | Two admins operating simultaneously can create conflicting session state. | `active_session.json` enforces single-active-admin locking until explicit logout. |
| **Missing Ground Truth on Live Scans** | Uploaded clinical or custom `.npz`/`.nii.gz` scans may lack voxel labels. | Dashboard computes predicted foreground %, mean lesion probability, and slice/match overlays safely without fabricating Dice scores when labels are absent. |
| **SMTP Delivery Outage** | Unreachable or unconfigured SMTP server can block email OTP delivery. | Falls back to local challenge recording (`otp_challenge.json`) in local development when SMTP is unavailable. |

## 4. Consent & Retention Guidance
- Only synthetic, de-identified, or explicitly consented test scans should be loaded during development and evaluation.
- Rotate or purge `artifacts/admin_vault/` test logs before sharing or archiving local workspaces.
