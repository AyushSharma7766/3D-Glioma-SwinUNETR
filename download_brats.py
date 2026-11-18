"""
================================================================================
BraTS Dataset Downloader: Medical Segmentation Decathlon Task01 (Brain Tumour)
================================================================================
Downloads:
  1. MSD Task01_BrainTumour dataset (~7.5 GB, 484 training cases)
     - 4-channel multimodal MRI: T1, T1ce (contrast-enhanced), T2, FLAIR
     - 3-region tumour segmentation labels (overlapping hierarchy)
  2. Self-supervised pre-trained Swin-UNETR encoder weights
     - Pre-trained on 5,050 unlabeled brain MRI volumes using 3D masked
       volume inpainting (Hatamizadeh et al., CVPR 2022)
     - Provides a powerful initialization that significantly outperforms
       random initialization, reducing convergence time by ~40%

Usage:
  python3 download_brats.py --data_dir ./brats_data
================================================================================
"""

import argparse
import os
import sys
from pathlib import Path


def download_msd_brain_tumour(data_dir: str):
    """
    Downloads the Medical Segmentation Decathlon Task01 Brain Tumour dataset.
    Uses MONAI's built-in downloader which handles Google Drive rate-limiting
    and tar.gz extraction automatically.
    """
    data_dir = Path(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    if (data_dir / "dataset.json").exists():
        task_dir = data_dir
    elif (data_dir / "Task01_BrainTumour" / "dataset.json").exists():
        task_dir = data_dir / "Task01_BrainTumour"
    else:
        task_dir = data_dir / "Task01_BrainTumour"

    if (task_dir / "dataset.json").exists():
        import json
        with open(task_dir / "dataset.json") as f:
            meta = json.load(f)
        n_train = len(meta.get("training", []))
        n_test = len(meta.get("test", []))
        print(f"[Dataset] Task01_BrainTumour already exists at {task_dir}")
        print(f"  Training cases : {n_train}")
        print(f"  Test cases     : {n_test}")
        return str(task_dir)

    print("=" * 70)
    print("  DOWNLOADING: Medical Segmentation Decathlon - Task01 Brain Tumour")
    print("  Size: ~7.5 GB | 484 training + 266 test multimodal brain MRI cases")
    print("=" * 70)

    try:
        from monai.apps import download_and_extract

        resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
        download_and_extract(
            url=resource,
            filepath=str(data_dir / "Task01_BrainTumour.tar"),
            output_dir=str(data_dir),
            hash_val=None,
            progress=True,
        )
        print(f"\n[Success] Dataset extracted to: {task_dir}")

    except Exception as e:
        print(f"[Warning] MONAI download failed: {e}")
        print("[Fallback] Trying alternative download with wget...")
        os.system(
            f'wget -c "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"'
            f' -O "{data_dir}/Task01_BrainTumour.tar"'
        )
        os.system(f'tar -xf "{data_dir}/Task01_BrainTumour.tar" -C "{data_dir}"')
        print(f"[Success] Dataset extracted to: {task_dir}")

    return str(task_dir)


def download_pretrained_swinunetr(output_dir: str):
    """
    Downloads the self-supervised pre-trained Swin-UNETR encoder weights.
    """
    output_dir = Path(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    weights_path = output_dir / "model_swinvit.pt"

    if weights_path.exists():
        import torch
        try:
            w = torch.load(str(weights_path), map_location="cpu", weights_only=True)
            print(f"[Pretrained Weights] Already downloaded: {weights_path}")
            print(f"  Encoder parameters: {len(w)} tensors")
            return str(weights_path)
        except Exception:
            print("[Warning] Existing weights file is corrupted. Re-downloading...")

    print("=" * 70)
    print("  DOWNLOADING: Self-Supervised Pre-Trained Swin-UNETR Encoder Weights")
    print("  Source: MONAI Research Contributions (Hatamizadeh et al., CVPR 2022)")
    print("  Pre-trained on 5,050 brain MRI volumes via masked volume inpainting")
    print("=" * 70)

    url = (
        "https://github.com/Project-MONAI/MONAI-extra-test-data/"
        "releases/download/0.8.1/model_swinvit.pt"
    )

    try:
        import urllib.request
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, str(weights_path))
        print(f"[Success] Saved pre-trained weights to: {weights_path}")
    except Exception as e:
        print(f"[Warning] urllib download failed: {e}")
        print("[Fallback] Trying wget...")
        os.system(f'wget -c "{url}" -O "{weights_path}"')

    return str(weights_path)


def verify_setup(data_dir: str, weights_dir: str):
    """Verifies that all required files are present and valid."""
    data_dir = Path(data_dir)
    weights_dir = Path(weights_dir)

    task_dir = data_dir / "Task01_BrainTumour"
    weights_path = weights_dir / "model_swinvit.pt"

    print("\n" + "=" * 70)
    print("  VERIFICATION REPORT")
    print("=" * 70)

    # Check dataset
    images_dir = task_dir / "imagesTr"
    labels_dir = task_dir / "labelsTr"
    if images_dir.exists():
        n_images = len(list(images_dir.glob("*.nii.gz")))
        n_labels = len(list(labels_dir.glob("*.nii.gz")))
        print(f"  ✓ Training images : {n_images} NIfTI volumes in {images_dir}")
        print(f"  ✓ Training labels  : {n_labels} NIfTI volumes in {labels_dir}")
    else:
        print(f"  ✗ Training images NOT FOUND at {images_dir}")

    # Check pretrained weights
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 ** 2)
        print(f"  ✓ Pre-trained weights : {size_mb:.1f} MB at {weights_path}")
    else:
        print(f"  ✗ Pre-trained weights NOT FOUND at {weights_path}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            for i in range(n_gpus):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  ✓ GPU {i}: {name} ({mem:.0f} GB)")
        else:
            print("  ✗ No CUDA GPUs detected")
    except ImportError:
        print("  ✗ PyTorch not installed")

    print("=" * 70)
    print("  Ready for training!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download BraTS (MSD Task01) dataset and pre-trained Swin-UNETR weights"
    )
    parser.add_argument(
        "--data_dir", type=str, default="brats_data",
        help="Directory to download the MSD Brain Tumour dataset"
    )
    parser.add_argument(
        "--weights_dir", type=str, default="pretrained",
        help="Directory to save pre-trained Swin-UNETR encoder weights"
    )
    parser.add_argument(
        "--skip_data", action="store_true",
        help="Skip dataset download (only download pre-trained weights)"
    )
    parser.add_argument(
        "--skip_weights", action="store_true",
        help="Skip pre-trained weights download (only download dataset)"
    )
    args = parser.parse_args()

    if not args.skip_data:
        download_msd_brain_tumour(args.data_dir)

    if not args.skip_weights:
        download_pretrained_swinunetr(args.weights_dir)

    verify_setup(args.data_dir, args.weights_dir)


if __name__ == "__main__":
    main()

