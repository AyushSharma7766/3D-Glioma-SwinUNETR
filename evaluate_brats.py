"""
================================================================================
BraTS Evaluation & Clinical Metric Benchmark
================================================================================
Academic Defense Context:
-------------------------
Evaluates the trained 3D Swin-UNETR on clinical multi-modal brain MRI cases
using the official Medical Image Computing and Computer Assisted Intervention
(MICCAI) BraTS evaluation protocol:

1. TUMOR SUB-REGION DEFINITION:
   - Enhancing Tumor (ET): Actively dividing, vascularized tumor cells
     (label 4 in raw BraTS annotations; channel 2 in multi-channel target).
   - Tumor Core (TC): ET + Necrotic / non-enhancing core
     (labels 1 + 4; channel 0 in multi-channel target).
   - Whole Tumor (WT): TC + peritumoral edematous brain tissue
     (labels 1 + 2 + 4; channel 1 in multi-channel target).

2. QUANTITATIVE METRICS:
   - Dice Similarity Coefficient (DSC): Overlap agreement between prediction and ground truth.
   - 95th Percentile Hausdorff Distance (HD95): Boundary distance metric measuring
     contour localization precision in millimeters, filtering out spuriously distant outliers.
================================================================================
"""

import argparse
from pathlib import Path
import json
import torch
import numpy as np
import nibabel as nib

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
    AsDiscrete
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from train_brats import build_swin_unetr_brats


def evaluate_case(
    checkpoint_path: Path,
    image_path: Path,
    label_path: Path = None,
    output_mask_path: Path = Path("outputs_brats/pred_brats_case.nii.gz"),
    roi_size: int = 128,
    sw_batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 78)
    print("  3D SWIN-UNETR CLINICAL BraTS FULL-VOLUME INFERENCE")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Image      : {image_path}")
    if label_path:
        print(f"  Label      : {label_path}")
    print("=" * 78)

    # 1. Rebuild Architecture and Load Checkpoint
    model = build_swin_unetr_brats(
        pretrained_path=None,
        feature_size=48,
        use_checkpoint=False
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Strip DDP module prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Build Preprocessing Transform
    keys = ["image", "label"] if label_path and Path(label_path).exists() else ["image"]
    transforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=keys),
    ]
    if "label" in keys:
        lbl_data = nib.load(str(label_path)).get_fdata()
        et_label = 3 if 3 in np.unique(lbl_data) else 4
        transforms.append(ConvertToMultiChannelBasedOnBratsClassesd(keys="label", et_label=et_label))
    transforms.extend([
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest") if "label" in keys else ("bilinear",)),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    val_transforms = Compose(transforms)

    data_item = {"image": str(image_path)}
    if "label" in keys:
        data_item["label"] = str(label_path)

    batch = val_transforms(data_item)
    inputs = batch["image"].unsqueeze(0).to(device)  # (1, 4, H, W, D)

    # 3. Sliding Window Inference with Gaussian Importance Blending
    print("[Inference] Executing Gaussian sliding-window volumetric inference (128x128x128)...")
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits = sliding_window_inference(
                inputs,
                roi_size=(roi_size, roi_size, roi_size),
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=0.6,
                mode="gaussian"
            )

        post_sigmoid = Activations(sigmoid=True)
        post_thresh = AsDiscrete(threshold=0.5)

        probs = post_sigmoid(logits[0])
        preds = post_thresh(probs).cpu().numpy().astype(np.uint8)  # (3, H, W, D)

        # Intracranial Brain Masking:
        # Voxels in the empty air outside the patient's skull (zero across all 4 MRI channels)
        # are guaranteed to be background. Suppress background air artifacts.
        brain_mask = (inputs[0].abs() > 1e-3).any(dim=0).cpu().numpy()
        preds[:, ~brain_mask] = 0

    # 4. Map 3-channel BraTS regions back to discrete labels with strict anatomical hierarchy:
    # Hierarchy: Enhancing Tumor (ET) ⊆ Tumor Core (TC) ⊆ Whole Tumor (WT)
    tc = preds[0] & preds[1]       # TC must be inside WT
    wt = preds[1]
    et = preds[2] & tc             # ET must be inside TC

    discrete_label = np.zeros_like(wt, dtype=np.int16)
    discrete_label[(wt == 1) & (tc == 0)] = 2  # Edema (WT \ TC)
    discrete_label[(tc == 1) & (et == 0)] = 1  # Necrotic Core (TC \ ET)
    discrete_label[et == 1] = 3                # Enhancing Core (ET: Label 3)
    discrete_label[~brain_mask] = 0

    # Save NIfTI mask
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    orig_nii = nib.load(str(image_path))
    nib.save(nib.Nifti1Image(discrete_label, orig_nii.affine, orig_nii.header), str(output_mask_path))
    print(f"[Export] Saved predicted segmentation mask to: {output_mask_path}")

    # 5. Compute Quantitative Benchmark Metrics if ground truth is available
    if "label" in keys:
        gt = batch["label"].unsqueeze(0).to(device)  # (1, 3, H, W, D)
        pred_tensor = post_thresh(post_sigmoid(logits)).to(device)

        dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        dice_metric(pred_tensor, gt)
        dices = dice_metric.aggregate().cpu().numpy()

        hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
        hd95_metric(pred_tensor, gt)
        hd95 = hd95_metric.aggregate().cpu().numpy()

        print("\n" + "=" * 60)
        print("  CLINICAL BraTS EVALUATION BENCHMARK RESULTS")
        print("=" * 60)
        regions = ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
        for idx, name in enumerate(regions):
            print(f"  {name:<22} : Dice = {dices[idx]:.4f} ({dices[idx]*100:.2f}%) | HD95 = {hd95[idx]:.2f} mm")
        print("-" * 60)
        print(f"  Mean Regional Dice     : {np.mean(dices):.4f} ({np.mean(dices)*100:.2f}%)")
        print(f"  Mean 95% Hausdorff     : {np.mean(hd95):.2f} mm")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D Swin-UNETR on BraTS multimodal volumes")
    parser.add_argument("--checkpoint", type=str, default="outputs_brats/checkpoints/best_brats_model.pth")
    parser.add_argument("--image", type=str, required=True, help="Path to 4-channel input MRI (.nii.gz)")
    parser.add_argument("--label", type=str, default=None, help="Optional path to ground truth label (.nii.gz)")
    parser.add_argument("--output_mask", type=str, default="outputs_brats/pred_brats_case.nii.gz")
    parser.add_argument("--roi_size", type=int, default=128)
    parser.add_argument("--sw_batch_size", type=int, default=4)
    args = parser.parse_args()

    evaluate_case(
        checkpoint_path=Path(args.checkpoint),
        image_path=Path(args.image),
        label_path=Path(args.label) if args.label else None,
        output_mask_path=Path(args.output_mask),
        roi_size=args.roi_size,
        sw_batch_size=args.sw_batch_size
    )


if __name__ == "__main__":
    main()

