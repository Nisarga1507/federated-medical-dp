from pathlib import Path

BT = "```"

how_to_run = f"""## How to Run (End-to-End)

### 1. Environment & Dependencies
{BT}bash
git clone https://github.com/Nisarga1507/federated-medical-dp.git
cd federated-medical-dp
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
{BT}

### 2. Run Automated Tests (`pytest`)
{BT}bash
PYTHONPATH=. pytest -v
{BT}

### 3. Run Multi-Seed & Multi-Site Privacy–Utility Ablation
{BT}bash
PYTHONPATH=. python scripts/run_privacy_ablation.py
{BT}

### 4. Launch the Streamlit Clinical & Admin Vault Dashboard
{BT}bash
PYTHONPATH=. streamlit run dashboard/app.py
{BT}
Open `http://localhost:8501` to switch between `ε = 10.0` (`eps_10.0`) and `ε = 2.0` (`latest_eps2`), inspect 3D slices and Match Maps, export JSON/CSV reports, and test Employee/Admin Vault workflows.

### 5. Run Federated Server & Clients (`Flower`) or `Docker`
{BT}bash
# Terminal 1 — Start Flower server
PYTHONPATH=. python run_server.py

# Terminal 2 — Connect Client 1 (data/client_1)
PYTHONPATH=. python run_client.py --client-id 1

# Terminal 3 — Connect Client 2 (data/client_2)
PYTHONPATH=. python run_client.py --client-id 2

# Docker build & run
docker build -t federated-medical-dp .
docker run -p 8501:8501 federated-medical-dp
{BT}
"""

p = Path("README.md")
text = p.read_text(encoding="utf-8")
if "## How to Run (End-to-End)" not in text:
    p.write_text(text.rstrip() + "\n\n" + how_to_run.strip() + "\n", encoding="utf-8")
print("Appended How to Run section cleanly.")
