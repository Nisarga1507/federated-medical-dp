# Federated 3D Medical Image Segmentation (ε-DP)

## Privacy-Preserving Biometric RBAC & Shift Analytics

- **Zero Raw Camera Frame Persistence:** Admin face and hand-pose captures are processed in memory via OpenCV into normalized feature vectors (`admin_signatures.npz`), while employee hand-pose logins record named gesture labels (`pose_name`) and SHA-256 verification digests rather than raw webcam images.
- **Paired Branch & Session Audit Trail:** `build_paired_staff_sessions` pairs employee `login` and `logout` events across branches and 6-hour shifts to compute session durations (`duration_minutes`) and filter activity by branch or named hand pose in **Dashboard 3 (Admin Vault)**.

## Reproducibility, Threat Model & Operational Monitoring

- **Containerization & CI/CD:** Includes a production `Dockerfile` and `.github/workflows/ci.yml` running automated `pytest` verification on every push and pull request.
- **Security & Privacy Threat Model:** Documented in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), detailing federated gradient privacy (`Opacus` `ε-DP`), zero-raw-frame biometric storage, and residual-risk mitigations.
- **Operational Telemetry:** `utils/monitoring.py` tracks active clinical sessions, 3D Dice drift alerts, and p95 inference latency across multi-site evaluations.

## Multi-Seed & Multi-Site Privacy–Utility Benchmarks

Evaluated via `scripts/run_privacy_ablation.py` across `client_1` and `client_2` splits over seeds `[42, 123, 777]` (120 total evaluations per checkpoint on `mps`, decision threshold `0.50`):

| Checkpoint / Privacy Regime | Privacy Budget (`ε`) | Overall 3D Dice (Mean ± Std) | `client_1` 3D Dice (n=60) | `client_2` 3D Dice (n=60) | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| High Utility (`eps_10.0`) | `10.0` | `0.7602 ± 0.0470` | `0.7469 ± 0.0515` | `0.7736 ± 0.0375` | `8.90 ms` |
| Active Unarchived (`global_unet.pt`) | `10.0` | `0.7602 ± 0.0470` | `0.7469 ± 0.0515` | `0.7736 ± 0.0375` | `31.14 ms` |
| Strict Privacy (`latest_eps2`) | `2.0` | `0.0570 ± 0.0253` | `0.0542 ± 0.0293` | `0.0597 ± 0.0202` | `8.96 ms` |

- **Single-Scan Diagnostic Reference (`data/client_2/test/case_000.npz`, Slice 16):** Achieves `0.7766` 3D Dice with `2.18%` predicted foreground and `7.83%` mean lesion probability.
- **Privacy–Utility Takeaway:** Tightening the differential privacy budget from `ε = 10.0` to `ε = 2.0` substantially increases gradient noise relative to fine 3D lesion boundaries, directly illustrating why clinical federated deployments must balance formal privacy guarantees against segmentation sensitivity.

## Privacy-Preserving Biometric RBAC & Operational Monitoring

- **Zero Raw Camera Frame Persistence:** Admin face and hand-pose captures are converted in memory into normalized embeddings (`admin_signatures.npz`), while employee hand-pose logins record named gesture labels (`pose_name`) and SHA-256 digests (`sha256:<16-hex>`) in `staff_audit_events.json`.
- **Paired Branch & Shift Duration Analytics:** `build_paired_staff_sessions` in `utils/admin_vault.py` pairs employee login and logout events across 6 clinical branches and 6-hour AM/PM shifts to compute session durations (`duration_minutes`), surface active sessions, and filter audit trails by branch or named hand pose in Dashboard 3 (Admin Vault).
- **Security & Privacy Threat Model:** Full trust-boundary analysis, adversary assumptions, and residual-risk mitigations are documented in `docs/THREAT_MODEL.md` and `THREAT_MODEL_AND_NEXT_STEPS.md`.
- **Operational Telemetry:** `utils/monitoring.py` (`summarize_operational_health`) tracks active clinical sessions, mean 3D Dice, low-Dice drift alerts, and p95 inference latency.

## How to Run (End-to-End)

### 1. Environment & Dependencies
```bash
git clone https://github.com/Nisarga1507/federated-medical-dp.git
cd federated-medical-dp
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Tests (`pytest`)
```bash
PYTHONPATH=. pytest -v
```

### 3. Run Multi-Seed & Multi-Site Privacy–Utility Ablation
```bash
PYTHONPATH=. python scripts/run_privacy_ablation.py
```

### 4. Launch the Streamlit Clinical & Admin Vault Dashboard
```bash
PYTHONPATH=. streamlit run dashboard/app.py
```
Open `http://localhost:8501` to switch between `ε = 10.0` (`eps_10.0`) and `ε = 2.0` (`latest_eps2`), inspect 3D slices and Match Maps, export JSON/CSV reports, and test Employee/Admin Vault workflows.

### 5. Run Federated Server & Clients (`Flower`) or `Docker`
```bash
# Terminal 1 — Start Flower server
PYTHONPATH=. python run_server.py

# Terminal 2 — Connect Client 1 (data/client_1)
PYTHONPATH=. python run_client.py --client-id 1

# Terminal 3 — Connect Client 2 (data/client_2)
PYTHONPATH=. python run_client.py --client-id 2

# Docker build & run
docker build -t federated-medical-dp .
docker run -p 8501:8501 federated-medical-dp
```
