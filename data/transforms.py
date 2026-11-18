"""
================================================================================
Data Pipeline Transforms: MONAI Dictionary Transforms for 3D MRI Volumetry
================================================================================
Academic Defense Context:
Medical image preprocessing is fundamentally different from natural image preprocessing:
1. SPATIAL ISOTROPY (Spacingd):
   Clinical MRI scanners acquire slices with variable pixel spacing (e.g. 0.5 x 0.5 x 3.0 mm).
   Anisotropic voxel dimensions skew spatial gradient calculations and distort isotropic
   3D receptive fields. Resampling to an isotropic 1.0 x 1.0 x 1.0 mm space standardizes
   physical anatomical proportions across heterogeneous clinical sites.
   Interpolation mode must be BILINEAR for continuous MRI intensities and NEAREST for discrete
   integer segmentation class masks to prevent non-existent label hallucination.

2. ANATOMICAL COORDINATE ALIGNMENT (Orientationd):
   Different MRI scanners store volumes in neurological (LPS/LAS) or radiological (RAS)
   coordinate conventions. Reorienting to canonical 'RAS' (Right, Anterior, Superior)
   ensures spatial invariants are aligned prior to self-attention computation.

3. POSITIVE/NEGATIVE PATCH BALANCING (RandCropByPosNegLabeld):
   In whole-brain MRI, non-brain background voxels (air, skull, cervical spine) account for
   >70-80% of the field-of-view, while critical structures like the hippocampus represent
   <0.2% of the volume. Naive uniform random cropping starves the network of anatomical
   foreground gradients. RandCropByPosNegLabeld enforces an exact 1:1 ratio between
   patches centered on foreground anatomical structures and background regions.
================================================================================
"""

from typing import Tuple
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
    CastToTyped
)
import torch
from config import DataConfig


def get_train_transforms(data_cfg: DataConfig) -> Compose:
    """
    Constructs the training data transformation pipeline.
    
    Pipeline Steps:
    1. LoadImaged: Reads NIfTI (.nii, .nii.gz) files lazily.
    2. EnsureChannelFirstd: Converts shape (H, W, D) -> (1, H, W, D).
    3. Orientationd: Transforms affine matrix to canonical RAS anatomical space.
    4. Spacingd: Resamples voxels to isotropic 1.0mm spacing (bilinear for img, nearest for mask).
    5. NormalizeIntensityd: Z-score normalization (mean=0, std=1) on non-zero tissue voxels.
    6. RandCropByPosNegLabeld: Extracts 3D balanced patches of spatial size `roi_size`.
    7. RandFlipd / RandRotated: 3D spatial augmentations invariant to anatomical pose.
    8. RandGaussianNoised / RandAdjustContrastd: Intensity augmentations simulating scanner noise/contrast.
    9. EnsureTyped: Converts tensors to PyTorch tensor format on CUDA/CPU.
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=data_cfg.orientation),
        Spacingd(
            keys=["image", "label"],
            pixdim=data_cfg.target_spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=data_cfg.roi_size,
            pos=data_cfg.pos_sample_ratio,
            neg=data_cfg.neg_sample_ratio,
            num_samples=data_cfg.num_samples_per_volume,
            image_key="image",
            image_threshold=0.0,
        ),
        # Spatial rigid and affine augmentations
        RandRotated(
            keys=["image", "label"],
            range_x=0.3,
            range_y=0.3,
            range_z=0.3,
            mode=("bilinear", "nearest"),
            prob=0.3,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
        # Photometric & noise augmentations
        RandGaussianNoised(keys="image", prob=0.2, std=0.08),
        RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.6, 1.5)),
        CastToTyped(keys=["image"], dtype=torch.float32),
        CastToTyped(keys=["label"], dtype=torch.int64),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms(data_cfg: DataConfig) -> Compose:
    """
    Constructs the validation / evaluation transformation pipeline.
    
    Notice: No cropping is applied here. The entire 3D volume is preserved at
    full resolution so that `sliding_window_inference` can evaluate the complete brain.
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=data_cfg.orientation),
        Spacingd(
            keys=["image", "label"],
            pixdim=data_cfg.target_spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CastToTyped(keys=["image"], dtype=torch.float32),
        CastToTyped(keys=["label"], dtype=torch.int64),
        EnsureTyped(keys=["image", "label"]),
    ])

