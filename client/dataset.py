from pathlib import Path
from typing import Dict, List, Tuple
import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
)


def _create_synthetic_case(
    rng: np.random.Generator,
    in_channels: int,
    spatial_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a synthetic 3D multi-modal volume [C, D, H, W] and binary mask [1, D, H, W]."""
    d, h, w = spatial_size
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, d, dtype=np.float32),
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    radius = np.sqrt(xx**2 + yy**2 + zz**2)
    lesion_mask = (radius < rng.uniform(0.25, 0.42)).astype(np.int64)

    channels = []
    for c in range(in_channels):
        base = rng.normal(loc=0.5 + 0.2 * c, scale=0.15, size=(d, h, w)).astype(np.float32)
        base += lesion_mask.astype(np.float32) * (1.2 + 0.4 * c)
        channels.append(base)

    image = np.stack(channels, axis=0)  # [C, D, H, W]
    label = lesion_mask[np.newaxis, ...]  # [1, D, H, W]
    return image, label


def ensure_split_data(
    data_root: Path,
    client_id: int,
    splits_counts: Dict[str, int],
    in_channels: int,
    spatial_size: Tuple[int, int, int],
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Ensures train/val/test data exists for `client_id`.
    If `.nii.gz` files are placed in `data/client_{id}/{split}/images` and `labels`,
    loads them with MONAI; otherwise creates reproducible `.npz` synthetic 3D cases.
    """
    client_dir = data_root / f"client_{client_id}"
    rng = np.random.default_rng(seed=1000 + client_id * 100)
    result: Dict[str, List[Dict[str, np.ndarray]]] = {}

    for split_name, count in splits_counts.items():
        split_dir = client_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        nifti_images = sorted(split_dir.glob("*_image.nii.gz"))
        if nifti_images:
            records = []
            for img_path in nifti_images:
                lbl_path = split_dir / img_path.name.replace("_image.nii.gz", "_label.nii.gz")
                img_data = nib.load(str(img_path)).get_fdata().astype(np.float32)
                if img_data.ndim == 3:
                    img_data = np.repeat(img_data[np.newaxis, ...], in_channels, axis=0)
                if lbl_path.exists():
                    lbl_data = nib.load(str(lbl_path)).get_fdata().astype(np.int64)
                    if lbl_data.ndim == 3:
                        lbl_data = lbl_data[np.newaxis, ...]
                else:
                    lbl_data = np.zeros((1, *img_data.shape[1:]), dtype=np.int64)
                records.append({"image": img_data, "label": lbl_data})
            result[split_name] = records
            continue

        existing_npz = sorted(split_dir.glob("case_*.npz"))
        if len(existing_npz) < count:
            for idx in range(len(existing_npz), count):
                img, lbl = _create_synthetic_case(rng, in_channels, spatial_size)
                np.savez_compressed(split_dir / f"case_{idx:03d}.npz", image=img, label=lbl)
            existing_npz = sorted(split_dir.glob("case_*.npz"))

        records = []
        for npz_path in existing_npz[:count]:
            with np.load(npz_path) as data:
                records.append({
                    "image": data["image"].astype(np.float32),
                    "label": data["label"].astype(np.int64),
                })
        result[split_name] = records

    return result


def build_dataloaders(
    client_id: int,
    splits_counts: Dict[str, int],
    in_channels: int,
    spatial_size: Tuple[int, int, int],
    batch_size: int = 2,
    data_root: Path = Path("data"),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for a given hospital client."""
    splits = ensure_split_data(
        data_root=data_root,
        client_id=client_id,
        splits_counts=splits_counts,
        in_channels=in_channels,
        spatial_size=spatial_size,
    )

    transforms = Compose([
        EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.long]),
    ])

    loaders = []
    for split_name in ("train", "val", "test"):
        ds = CacheDataset(data=splits[split_name], transform=transforms, cache_rate=1.0, num_workers=0)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
        )
        loaders.append(loader)

    return loaders[0], loaders[1], loaders[2]
