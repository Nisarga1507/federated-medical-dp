# Privacy vs. Utility Benchmark (`federated-medical-dp`)

* **Architecture:** Flower (`FedAvg`) aggregator (`meddp-fedavg-aggregator`) + MONAI 3D U-Net + $(\epsilon, \delta)$-DP gradient clipping and Gaussian noise (`delta = 1e-5`, `max_grad_norm = 1.0`).
* **Case Study (`data/client_2/test/case_000.npz`, Slice 16, threshold `t = 0.60`):**
  * **Strict Privacy ($\epsilon = 2.0$, `latest_eps2`):** 3D Volume Dice `0.1488`, Predicted Foreground `20.46%`, Mean Lesion Probability `47.37%` (heavy background false positives in red under tight noise).
  * **Higher Utility ($\epsilon = 10.0$, `eps_10.0`):** 3D Volume Dice `0.7695`, Predicted Foreground `2.29%`, Mean Lesion Probability `7.94%` (clean lesion overlap in green with minimal red background noise).
* **Diagnostic Tooling:** Interactive threshold slider, Green (TP) / Red (FP) / Blue (FN) Match Maps, archived checkpoint switching, and automated `pytest` regression checks.
