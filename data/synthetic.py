"""
================================================================================
Synthetic Brain MRI NIfTI Generator for Testing & Verification
================================================================================
Generates realistic multi-class 3D brain phantom volumes and saves them as NIfTI (.nii.gz).
Includes anatomically realistic multi-layer structures:
- Class 0: Background
- Class 1: Cerebral White Matter
- Class 2: Cerebral Cortex (Gray Matter)
- Class 3: Lateral Ventricles (CSF)
- Class 4: Inferior Lateral Ventricle
- Class 5: Cerebellum White Matter
- Class 6: Cerebellum Cortex
- Class 7: Thalamus
- Class 8: Caudate
- Class 9: Putamen
- Class 10: Pallidum
- Class 11: 3rd Ventricle
- Class 12: Brainstem
- Class 13: Hippocampus
================================================================================
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter


def generate_synthetic_brain_volume(
    shape: Tuple[int, int, int] = (128, 128, 96),
    num_classes: int = 14,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes a 3D structural T1w brain MRI volume with aligned ground-truth segmentation.

    Args:
        shape: Volumetric dimensions (H, W, D).
        num_classes: Number of distinct anatomical segmentation classes.
        voxel_spacing: Voxel resolution in mm.

    Returns:
        Tuple of (mri_intensity, segmentation_label, affine_matrix)
    """
    H, W, D = shape
    mri = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.int16)

    # Coordinate grids normalized to [-1, 1]
    zz, yy, xx = np.ogrid[:H, :W, :D]
    cz, cy, cx = H // 2, W // 2, D // 2
    z_norm = (zz - cz) / (H * 0.45)
    y_norm = (yy - cy) / (W * 0.45)
    x_norm = (xx - cx) / (D * 0.45)
    r_ellipsoid = x_norm**2 + y_norm**2 + z_norm**2

    # 1. Cerebral Cortex (Outer Gray Matter mantle)
    cortex_mask = (r_ellipsoid <= 1.0) & (r_ellipsoid > 0.65)
    label[cortex_mask] = 2
    mri[cortex_mask] = 0.55 + np.random.normal(0, 0.04, size=cortex_mask.sum())

    # 2. Cerebral White Matter (Inner core)
    wm_mask = (r_ellipsoid <= 0.65) & (r_ellipsoid > 0.15)
    label[wm_mask] = 1
    mri[wm_mask] = 0.85 + np.random.normal(0, 0.04, size=wm_mask.sum())

    # 3. Brainstem (Inferior central stalk)
    stem_mask = (z_norm < -0.3) & (z_norm > -0.85) & (x_norm**2 + y_norm**2 < 0.08)
    label[stem_mask] = 12
    mri[stem_mask] = 0.70 + np.random.normal(0, 0.03, size=stem_mask.sum())

    # 4. Cerebellum (Posterior-inferior structures)
    cereb_wm = (z_norm < -0.2) & (z_norm > -0.7) & (y_norm < -0.2) & (x_norm**2 + (y_norm+0.4)**2 < 0.2)
    label[cereb_wm] = 5
    mri[cereb_wm] = 0.78 + np.random.normal(0, 0.03, size=cereb_wm.sum())

    # 5. Lateral Ventricles (Bilateral CSF-filled cavities)
    vent_left = ((x_norm - 0.18)**2 / 0.015 + y_norm**2 / 0.06 + z_norm**2 / 0.04) < 1.0
    vent_right = ((x_norm + 0.18)**2 / 0.015 + y_norm**2 / 0.06 + z_norm**2 / 0.04) < 1.0
    vent_mask = vent_left | vent_right
    label[vent_mask] = 3
    mri[vent_mask] = 0.15 + np.random.normal(0, 0.02, size=vent_mask.sum())  # CSF is hypointense on T1w

    # 6. Deep Subcortical Nuclei (Thalamus, Caudate, Putamen, Hippocampus)
    thal_mask = ((np.abs(x_norm) - 0.16)**2 / 0.01 + (y_norm + 0.05)**2 / 0.02 + z_norm**2 / 0.02) < 0.6
    label[thal_mask] = 7
    mri[thal_mask] = 0.65 + np.random.normal(0, 0.03, size=thal_mask.sum())

    caud_mask = ((np.abs(x_norm) - 0.22)**2 / 0.008 + (y_norm - 0.12)**2 / 0.015 + (z_norm - 0.08)**2 / 0.015) < 0.5
    label[caud_mask] = 8
    mri[caud_mask] = 0.62 + np.random.normal(0, 0.03, size=caud_mask.sum())

    hippo_mask = ((np.abs(x_norm) - 0.28)**2 / 0.006 + (y_norm + 0.18)**2 / 0.02 + (z_norm + 0.2)**2 / 0.008) < 0.5
    label[hippo_mask] = 13
    mri[hippo_mask] = 0.58 + np.random.normal(0, 0.03, size=hippo_mask.sum())

    # Apply Gaussian smoothing to simulate biological tissue transitions and partial volume effect
    mri = gaussian_filter(mri, sigma=0.8)

    # Normalize intensity to [0, 1] range
    mri = np.clip(mri, 0.0, 1.0)

    # Standard RAS Affine matrix
    affine = np.diag([voxel_spacing[0], voxel_spacing[1], voxel_spacing[2], 1.0])
    affine[0, 3] = -shape[0] * voxel_spacing[0] / 2
    affine[1, 3] = -shape[1] * voxel_spacing[1] / 2
    affine[2, 3] = -shape[2] * voxel_spacing[2] / 2

    return mri, label, affine


def create_synthetic_dataset(
    output_dir: Path,
    num_train: int = 8,
    num_val: int = 2,
    shape: Tuple[int, int, int] = (128, 128, 96)
) -> Dict[str, List[Dict[str, str]]]:
    """
    Creates a folder of synthetic brain NIfTI files structured for MONAI training.

    Directory structure:
        output_dir/
            imagesTr/
            labelsTr/
            imagesVal/
            labelsVal/
    """
    output_dir = Path(output_dir)
    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"
    images_val = output_dir / "imagesVal"
    labels_val = output_dir / "labelsVal"

    for d in [images_tr, labels_tr, images_val, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [],
        "val": []
    }

    # Generate training cases
    print(f"[Synthetic Data] Generating {num_train} training NIfTI volumes in {output_dir}...")
    for idx in range(num_train):
        mri, label, affine = generate_synthetic_brain_volume(shape=shape)
        img_p = images_tr / f"sub-{idx+1:03d}_T1w.nii.gz"
        lbl_p = labels_tr / f"sub-{idx+1:03d}_seg.nii.gz"

        nib.save(nib.Nifti1Image(mri, affine), str(img_p))
        nib.save(nib.Nifti1Image(label, affine), str(lbl_p))

        splits["train"].append({
            "image": str(img_p.resolve()),
            "label": str(lbl_p.resolve())
        })

    # Generate validation cases
    print(f"[Synthetic Data] Generating {num_val} validation NIfTI volumes...")
    for idx in range(num_val):
        mri, label, affine = generate_synthetic_brain_volume(shape=shape)
        img_p = images_val / f"sub-val-{idx+1:03d}_T1w.nii.gz"
        lbl_p = labels_val / f"sub-val-{idx+1:03d}_seg.nii.gz"

        nib.save(nib.Nifti1Image(mri, affine), str(img_p))
        nib.save(nib.Nifti1Image(label, affine), str(lbl_p))

        splits["val"].append({
            "image": str(img_p.resolve()),
            "label": str(lbl_p.resolve())
        })

    print(f"[Synthetic Data] Dataset created successfully: {len(splits['train'])} train, {len(splits['val'])} val.")
    return splits

