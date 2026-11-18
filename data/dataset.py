"""
================================================================================
Dataset Module: High-Throughput MONAI Caching & DataLoaders for NVIDIA H200
================================================================================
Academic Defense Context:
In deep learning on ultra-fast accelerators like the NVIDIA H200 (up to 4.8 TB/s memory
bandwidth), the computational bottleneck almost always shifts from the GPU compute cores
to host CPU preprocessing and I/O starvation.

High-Performance Data Engineering Strategy:
-------------------------------------------
1. CACHED TRANSFORMS (monai.data.CacheDataset):
   Deterministic operations (NIfTI decompression, orientation to RAS, 1mm spatial resampling)
   are computationally intensive when executed repeatedly on every epoch. CacheDataset
   evaluates all deterministic transforms upfront and caches the resulting 3D arrays in
   host RAM. Subsequent epochs only apply stochastic data augmentations (random cropping,
   rotation, noise), cutting per-batch data preparation time by over 10x and ensuring the
   H200 Tensor Cores stay continuously saturated at near 100% compute duty cycle.

2. PINNED MEMORY & ASYNCHRONOUS DMA TRANSFERS:
   `pin_memory=True` locks host virtual memory pages, enabling direct memory access (DMA)
   between system RAM and the H200's HBM3e high-bandwidth memory via PCIe Gen 5 / NVLink,
   bypassing intermediate CPU page copying.
================================================================================
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import torch
from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate
from config import DataConfig
from data.transforms import get_train_transforms, get_val_transforms
from data.synthetic import create_synthetic_dataset


def discover_or_create_dataset_files(
    data_dir: Path,
    num_synthetic_train: int = 8,
    num_synthetic_val: int = 2
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Finds existing NIfTI files in data_dir, or automatically generates synthetic volumes
    to enable out-of-the-box verification without manual data preparation.
    """
    data_dir = Path(data_dir)
    json_path = data_dir / "dataset.json"

    # 1. Check for standard Medical Segmentation Decathlon dataset.json
    if json_path.exists():
        with open(json_path, "r") as f:
            meta = json.load(f)
        train_files = meta.get("training", [])
        val_files = meta.get("validation", [])
        # Fix relative paths to absolute
        for item in train_files:
            item["image"] = str((data_dir / item["image"]).resolve())
            item["label"] = str((data_dir / item["label"]).resolve())
        for item in val_files:
            item["image"] = str((data_dir / item["image"]).resolve())
            item["label"] = str((data_dir / item["label"]).resolve())
        if train_files:
            return train_files, val_files

    # 2. Check for imagesTr and labelsTr folders
    images_tr = sorted(list(data_dir.glob("imagesTr/*.nii*")))
    labels_tr = sorted(list(data_dir.glob("labelsTr/*.nii*")))
    images_val = sorted(list(data_dir.glob("imagesVal/*.nii*")))
    labels_val = sorted(list(data_dir.glob("labelsVal/*.nii*")))

    if images_tr and labels_tr:
        train_files = [
            {"image": str(img.resolve()), "label": str(lbl.resolve())}
            for img, lbl in zip(images_tr, labels_tr)
        ]
        val_files = [
            {"image": str(img.resolve()), "label": str(lbl.resolve())}
            for img, lbl in zip(images_val, labels_val)
        ]
        if not val_files:
            # If no validation folder, hold out 20% of training cases
            split_idx = max(1, int(len(train_files) * 0.8))
            val_files = train_files[split_idx:]
            train_files = train_files[:split_idx]
        return train_files, val_files

    # 3. If no files exist, automatically generate synthetic brain volumes
    print(f"[Dataset] No NIfTI files found in {data_dir}. Generating synthetic verification dataset...")
    splits = create_synthetic_dataset(
        data_dir,
        num_train=num_synthetic_train,
        num_val=num_synthetic_val
    )
    return splits["train"], splits["val"]


def build_dataloaders(
    train_files: List[Dict[str, str]],
    val_files: List[Dict[str, str]],
    data_cfg: DataConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds the high-throughput MONAI training and validation DataLoaders.

    Args:
        train_files: List of dicts with 'image' and 'label' paths for training.
        val_files: List of dicts with 'image' and 'label' paths for validation.
        data_cfg: Configuration parameters for caching, batch sizes, and worker threads.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transforms = get_train_transforms(data_cfg)
    val_transforms = get_val_transforms(data_cfg)

    print(f"[DataLoader] Initializing CacheDataset (cache_rate={data_cfg.cache_rate})...")

    # Training Dataset
    if data_cfg.cache_rate > 0.0:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=data_cfg.cache_rate,
            num_workers=min(4, data_cfg.num_workers) if data_cfg.num_workers > 0 else 0,
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=data_cfg.cache_rate,
            num_workers=min(4, data_cfg.num_workers) if data_cfg.num_workers > 0 else 0,
        )
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)

    # Note: RandCropByPosNegLabeld returns `num_samples` patches per volume.
    # `list_data_collate` stacks patches into a unified batch dimension (B * num_samples, C, H, W, D).
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        collate_fn=list_data_collate,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=(data_cfg.num_workers > 0),
        prefetch_factor=2 if data_cfg.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.val_batch_size,
        shuffle=False,
        num_workers=max(1, data_cfg.num_workers // 2) if data_cfg.num_workers > 0 else 0,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=(data_cfg.num_workers > 0),
    )

    print(f"[DataLoader] Ready: {len(train_ds)} train volumes, {len(val_ds)} val volumes.")
    return train_loader, val_loader

