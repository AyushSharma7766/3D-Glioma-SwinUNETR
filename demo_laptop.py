"""
================================================================================
BraTS 3D Swin-UNETR - Live Local Laptop Inference Demo
================================================================================
Optimized for live classroom/academic demonstrations:
- Multi-threaded CPU + GPU (NVIDIA RTX) auto-detection
- Configurable overlap (0.25 for ~35s fast CPU demo, 0.5 for fine detail)
- Automatic intracranial brain masking (suppresses background air artifacts)
- Strict anatomical hierarchy: Enhancing Tumor ⊆ Tumor Core ⊆ Whole Tumor
- Auto-detects BraTS label encoding (MSD 1,2,3 vs Raw 1,2,4) to prevent NaN Dice
- Exports clinical NIfTI mask matching the input patient ID
================================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np

# Set multi-threading for fast CPU inference
if hasattr(os, "cpu_count"):
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)

# Ensure dependencies are available
try:
    import nibabel as nib
    import monai
    from monai.networks.nets import SwinUNETR
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        EnsureTyped,
        ConvertToMultiChannelBasedOnBratsClassesd,
        Orientationd,
        Spacingd,
        NormalizeIntensityd,
        Activations,
        AsDiscrete,
    )
    from monai.metrics import compute_dice
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Please run: pip install monai nibabel einops\n")
    sys.exit(1)


def build_swin_unetr(feature_size=48):
    """Builds the 3D Swin-UNETR architecture."""
    model = SwinUNETR(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        feature_size=feature_size,
        use_checkpoint=False,
    )
    return model


def run_laptop_inference(
    checkpoint_path: str = "best_brats_model.pth",
    image_path: str = "BRATS_460.nii.gz",
    label_path: str = None,
    output_path: str = None,
    roi_size: int = 128,
    overlap: float = 0.25,
):
    print("\n" + "=" * 70)
    print("  3D SWIN-UNETR: CLINICAL BRAIN TUMOR SEGMENTATION DEMO")
    print("=" * 70)

    # 1. Hardware Detection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device = torch.device("cuda:0")
        print(f"[*] Compute Device : GPU ({gpu_name}, {vram_gb:.1f} GB VRAM)")
        use_amp = True
    else:
        device = torch.device("cpu")
        print(f"[*] Compute Device : CPU (Multi-threaded with {torch.get_num_threads()} cores)")
        use_amp = False

    # 2. Check File Paths
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        for candidate in [Path("checkpoints") / checkpoint_path, Path("checkpoints/best_brats_model.pth"), Path("best_brats_model.pth")]:
            if candidate.exists():
                ckpt = candidate
                break
    if not ckpt.exists():
        print(f"\n[!] Checkpoint not found at: {ckpt.resolve()}")
        print("    Please place 'best_brats_model.pth' in 'checkpoints/' or the root folder.\n")
        return

    img = Path(image_path)
    if not img.exists():
        for candidate in [Path("sample_scans") / image_path, Path("sample_scans") / Path(image_path).name]:
            if candidate.exists():
                img = candidate
                break
    if not img.exists():
        print(f"\n[!] Patient scan not found at: {img.resolve()}")
        return

    # Auto-resolve output path if not specified
    if output_path is None or output_path == "":
        out_dir = Path("predictions")
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / f"pred_{img.name}")

    print(f"[*] Checkpoint     : {ckpt.name} ({ckpt.stat().st_size / (1024*1024):.1f} MB)")
    print(f"[*] Input MRI      : {img.name}")

    # 3. Model Initialization
    print("[*] Initializing 3D Swin-UNETR architecture (62.2M parameters)...")
    model = build_swin_unetr(feature_size=48)
    
    print("[*] Loading trained weights...")
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state)
    model.to(device)
    model.eval()
    print("    -> Model loaded successfully into memory.")

    # 4. Data Preprocessing & Ground Truth Label Detection
    print("[*] Preprocessing 4-channel MRI (T1, T1Gd, T2, FLAIR)...")
    keys = ["image"]
    has_gt = label_path is not None and Path(label_path).exists()
    
    et_label = 4
    if has_gt:
        keys.append("label")
        print(f"[*] Ground Truth   : {Path(label_path).name} (Dice evaluation enabled)")
        # Check if ground truth uses label 3 (MSD Task01) or label 4 (BraTS Raw)
        gt_raw = nib.load(str(label_path)).get_fdata()
        unique_labels = np.unique(gt_raw)
        if 3 in unique_labels and 4 not in unique_labels:
            et_label = 3

    t_list = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=keys),
    ]
    if has_gt:
        t_list.append(ConvertToMultiChannelBasedOnBratsClassesd(keys="label", et_label=et_label))
    t_list.extend([
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest") if has_gt else ("bilinear",)),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    pipeline = Compose(t_list)

    input_data = {"image": str(img)}
    if has_gt:
        input_data["label"] = str(label_path)

    batch = pipeline(input_data)
    inputs = batch["image"].unsqueeze(0).to(device)
    print(f"    -> Volume Shape: {inputs.shape[2:]} (4 multimodal channels)")

    # 5. Sliding-Window 3D Inference
    print(f"[*] Executing 3D Sliding-Window Swin Transformer Inference (ROI: {roi_size}³, Overlap: {overlap})...")
    start_time = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=(roi_size, roi_size, roi_size),
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
            probs = Activations(sigmoid=True)(logits[0])
            preds = AsDiscrete(threshold=0.5)(probs).cpu().numpy().astype(np.uint8)

    elapsed = time.time() - start_time
    print(f"    -> Inference completed in {elapsed:.2f} seconds!")

    # 6. Anatomical Brain Masking & Hierarchical Reconstruction
    # Suppress background air artifacts outside skull
    brain_mask = (inputs[0].abs() > 1e-3).any(dim=0).cpu().numpy()
    preds[:, ~brain_mask] = 0

    # Anatomical hierarchy: Enhancing Tumor ⊆ Tumor Core ⊆ Whole Tumor
    tc = preds[0] & preds[1]       # TC must reside within WT
    wt = preds[1]
    et = preds[2] & tc             # ET must reside within TC

    # Construct discrete 3D mask matching textbook layered anatomy:
    # 1 = Necrotic Core (Center), 2 = Peritumoral Edema (Outer swelling), 3 = Enhancing Tumor (Middle Rim)
    discrete_mask = np.zeros_like(wt, dtype=np.uint8)
    discrete_mask[(wt == 1) & (tc == 0)] = 1  # Necrotic / Non-enhancing Core (Center)
    discrete_mask[(tc == 1) & (et == 0)] = 2  # Peritumoral Edema (Outer infiltration)
    discrete_mask[et == 1] = 3                # Active Enhancing Tumor (Vascularized Middle Rim)
    discrete_mask[~brain_mask] = 0

    # 7. Quantitative Benchmark Evaluation (Dice Scores)
    if has_gt:
        labels = batch["label"].unsqueeze(0)  # (1, 3, H, W, D)
        # Convert hierarchical binary channels for evaluation tensor
        eval_pred = torch.tensor(np.stack([tc, wt, et], axis=0), dtype=torch.float32).unsqueeze(0)
        eval_gt = labels.float()

        dice_scores = compute_dice(y_pred=eval_pred, y=eval_gt, include_background=True)
        tc_dice = dice_scores[0, 0].item()
        wt_dice = dice_scores[0, 1].item()
        et_dice = dice_scores[0, 2].item()

        # Handle zero-voxel edge cases cleanly (True Negatives = 1.0)
        gt_np = labels[0].cpu().numpy()
        if np.isnan(tc_dice): tc_dice = 1.0 if (tc.sum() == 0 and gt_np[0].sum() == 0) else 0.0
        if np.isnan(wt_dice): wt_dice = 1.0 if (wt.sum() == 0 and gt_np[1].sum() == 0) else 0.0
        if np.isnan(et_dice): et_dice = 1.0 if (et.sum() == 0 and gt_np[2].sum() == 0) else 0.0
        mean_dice = (tc_dice + wt_dice + et_dice) / 3.0

        print("\n" + "-" * 70)
        print("  CLINICAL ACCURACY EVALUATION (MICCAI BraTS Protocol)")
        print("-" * 70)
        print(f"  Whole Tumor (WT) Dice      : {wt_dice:.4f}  ({wt_dice*100:.1f}%)")
        print(f"  Tumor Core (TC) Dice       : {tc_dice:.4f}  ({tc_dice*100:.1f}%)")
        print(f"  Enhancing Tumor (ET) Dice  : {et_dice:.4f}  ({et_dice*100:.1f}%)")
        print(f"  Overall Mean Regional Dice : {mean_dice:.4f}  ({mean_dice*100:.1f}%)")
        print("-" * 70)

    # 8. Export Predicted 3D NIfTI Volume
    orig_meta = batch["image_meta_dict"]
    affine = orig_meta.get("affine", np.eye(4))
    nii_out = nib.Nifti1Image(discrete_mask, affine)
    nib.save(nii_out, output_path)

    v_wt = int((discrete_mask > 0).sum())
    v_tc = int(((discrete_mask == 1) | (discrete_mask == 3)).sum())
    v_et = int((discrete_mask == 3).sum())

    print(f"\n[+] Segmented 3D Mask saved to : {Path(output_path).resolve()}")
    print(f"    Volumetric Counts: Whole Tumor = {v_wt:,} voxels | Core = {v_tc:,} | Enhancing = {v_et:,}")
    print("    Color Legend: 1 = Necrotic Core (Red), 2 = Edema (Green), 3 = Enhancing Tumor (Blue)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Local BraTS Swin-UNETR Inference Demo")
    parser.add_argument("--checkpoint", type=str, default="best_brats_model.pth", help="Path to best_brats_model.pth")
    parser.add_argument("--image", type=str, default="BRATS_460.nii.gz", help="Path to input multimodal MRI (.nii.gz)")
    parser.add_argument("--label", type=str, default="labelBRATS_460.nii.gz", help="Optional path to ground truth label")
    parser.add_argument("--output", type=str, default="", help="Output predicted mask path (default: pred_<image>)")
    parser.add_argument("--roi_size", type=int, default=128, help="ROI patch size (default: 128)")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding window overlap (0.25 for fast ~35s CPU demo)")

    args = parser.parse_args()
    run_laptop_inference(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        label_path=args.label,
        output_path=args.output,
        roi_size=args.roi_size,
        overlap=args.overlap,
    )
