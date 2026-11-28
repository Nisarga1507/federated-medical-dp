# Security & Privacy Threat Model (`federated-medical-dp`)

## 1. Assets & Trust Boundaries
- **Sensitive Assets:** Multi-site 3D medical volumes (`.npz` / `.nii.gz`), patient segmentation records, and staff/admin identity metadata.
- **Trust Boundaries:**
  - *Client Hospital Nodes -> Federated Aggregator:* Only model updates clipped and perturbed via Opacus `(ε, δ)`-Differential Privacy leave client nodes; raw patient scans remain local.
  - *Webcam Capture -> Admin Vault:* Live camera frames are processed in memory via OpenCV into normalized feature vectors (`admin_signatures.npz`) and SHA-256 digests (`staff_audit_events.json`); raw frames are never written to disk.

## 2. Threats, Mitigations & Residual Risks
| Threat Vector | Mitigation Implemented | Residual Risk / Next Hardening |
| :--- | :--- | :--- |
| **Gradient / Model Inversion across Sites** | Federated averaging (`Flower`) combined with per-sample gradient clipping and Gaussian noise (`Opacus` `ε = 2.0` / `ε = 10.0`). | High `ε` (`10.0`) trades tighter theoretical privacy for higher 3D Dice (`0.7695`); strict clinical deployment should enforce `ε <= 2.0` with secure aggregation. |
| **Raw Biometric Frame Leakage** | Zero raw image persistence; only low-dimensional embeddings and 16-char SHA-256 digests are stored, and `artifacts/admin_vault/` + `.env` are excluded via `.gitignore`. | Local `admin_signatures.npz` is unencrypted on disk; production deployment should encrypt vault files at rest via KMS/OS keychain. |
| **Presentation / Replay Spoofing** | Dual-factor visual verification requiring both centered face similarity (`>= 0.65`) and lower-frame hand-pose similarity (`>= 0.40`). | Static photo spoofing against lightweight cosine embeddings remains possible without active liveness/challenge-response prompts. |
| **Unauthorized Clinical Record Access** | Role-based gating separating Employee intake/scan workflows from Admin Vault audit & patient record views, backed by paired `login`/`logout` audit trails. | Single-node JSON storage; multi-hospital deployment should migrate audit logs to an append-only database with signed audit hashes. |
