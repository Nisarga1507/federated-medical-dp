# Federated 3D Medical Image Segmentation (ε-DP)

## Privacy-Preserving Biometric RBAC & Shift Analytics

- **Zero Raw Camera Frame Persistence:** Admin face and hand-pose captures are processed in memory via OpenCV into normalized feature vectors (`admin_signatures.npz`), while employee hand-pose logins record named gesture labels (`pose_name`) and SHA-256 verification digests rather than raw webcam images.
- **Paired Branch & Session Audit Trail:** `build_paired_staff_sessions` pairs employee `login` and `logout` events across branches and 6-hour shifts to compute session durations (`duration_minutes`) and filter activity by branch or named hand pose in **Dashboard 3 (Admin Vault)**.

## Reproducibility, Threat Model & Operational Monitoring

- **Containerization & CI/CD:** Includes a production `Dockerfile` and `.github/workflows/ci.yml` running automated `pytest` verification on every push and pull request.
- **Security & Privacy Threat Model:** Documented in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), detailing federated gradient privacy (`Opacus` `ε-DP`), zero-raw-frame biometric storage, and residual-risk mitigations.
- **Operational Telemetry:** `utils/monitoring.py` tracks active clinical sessions, 3D Dice drift alerts, and p95 inference latency across multi-site evaluations.
