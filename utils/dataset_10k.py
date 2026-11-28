from __future__ import annotations

import json
from pathlib import Path
import numpy as np


DATASET_ROOT = Path("data/dataset_10k")
SHARDS_DIR = DATASET_ROOT / "shards"
MANIFEST_PATH = DATASET_ROOT / "manifest.jsonl"

CONDITIONS = [
    "Glioma Lesion",
    "Meningioma Mass",
    "White Matter Hyperintensity",
    "Cortical Lesion",
    "Subcortical Abnormality",
]


def build_10k_catalog(total_samples: int = 10_000, num_clients: int = 10, seed: int = 1507) -> Path:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= total_samples:
            return MANIFEST_PATH

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    d, h, w = 32, 32, 32
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, d, dtype=np.float32),
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )

    num_templates = 20
    for t_idx in range(num_templates):
        cx = float(rng.uniform(-0.20, 0.20))
        cy = float(rng.uniform(-0.20, 0.20))
        cz = float(rng.uniform(-0.18, 0.18))
        rad = float(rng.uniform(0.25, 0.38))
        mask = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) < rad).astype(np.int64)
        ch0 = rng.normal(0.45, 0.12, size=(d, h, w)).astype(np.float32) + mask * 1.4
        ch1 = rng.normal(0.65, 0.14, size=(d, h, w)).astype(np.float32) + mask * 1.7
        np.savez_compressed(
            SHARDS_DIR / f"shard_{t_idx:02d}.npz",
            image=np.stack([ch0, ch1], axis=0),
            label=mask[np.newaxis, ...],
        )

    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for i in range(total_samples):
            client_num = (i % num_clients) + 1
            tpl_idx = int(rng.integers(0, num_templates))
            year = int(rng.integers(1952, 2004))
            month = int(rng.integers(1, 13))
            day = int(rng.integers(1, 29))
            entry = {
                "record_id": f"CASE-{i + 1:05d}",
                "client_id": f"client_{client_num}",
                "patient_name": f"Client Record #{i + 1:05d}",
                "disease": CONDITIONS[int(rng.integers(0, len(CONDITIONS)))],
                "date_of_birth": f"{year:04d}-{month:02d}-{day:02d}",
                "split": "train" if (i % 10) < 7 else ("val" if (i % 10) < 9 else "test"),
                "shard_path": str(SHARDS_DIR / f"shard_{tpl_idx:02d}.npz"),
            }
            handle.write(json.dumps(entry) + "\n")

    return MANIFEST_PATH


def fetch_catalog_sample(record_id: str) -> tuple[dict | None, np.ndarray | None, np.ndarray | None]:
    manifest = build_10k_catalog()
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["record_id"] == record_id:
                with np.load(row["shard_path"]) as data:
                    return row, data["image"].astype(np.float32), data["label"].astype(np.int64)
    return None, None, None
