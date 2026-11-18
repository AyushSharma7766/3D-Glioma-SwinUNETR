"""
================================================================================
Loss Function Module: Combined Dice-Cross-Entropy for Volumetric Segmentation
================================================================================
Academic Defense Context:
Whole-brain anatomical segmentation presents one of the most severe class-imbalance
challenges in medical computer vision:
- The background and large parenchyma (cerebral white matter, cerebral cortex)
  comprise >90% of the cranial volume.
- Vital subcortical structures (e.g., bilateral hippocampus, caudate, amygdala, 3rd ventricle)
  frequently comprise <0.1% of all voxels in the field of view.

Theoretical Justification for DiceCELoss:
------------------------------------------
1. SOFT DICE LOSS COMPONENT:
   Formulation:
       L_Dice = 1 - (2 * sum(p_i * y_i) + epsilon) / (sum(p_i^2) + sum(y_i^2) + epsilon)
   Because Dice loss evaluates the normalized harmonic overlap between the predicted
   probability distribution p and the one-hot ground truth y, it is invariant to
   the absolute volume scale of the structure. A mis-segmented voxel in a 50-voxel
   hippocampus incurs proportional penalty equal to a mis-segmented voxel in a
   50,000-voxel white matter region.
   `squared_pred=True` uses the Minkowski L2 denominator, which sharpens gradients
   and prevents oversmoothing of thin cortical structures.

2. CROSS-ENTROPY LOSS COMPONENT:
   Formulation:
       L_CE = - sum_c y_c * log(p_c)
   Dice loss surfaces suffer from plateau regions with vanishing gradients when
   initial predictions have near-zero overlap with the ground truth mask.
   Cross-Entropy provides smooth, convex, non-saturating gradients far from the
   optimum, accelerating convergence during early training iterations.

3. FUSED OBJECTIVE:
       L_Total = w_dice * L_Dice + w_ce * L_CE
   Fusing both terms yields complementary gradient dynamics: CE drives rapid
   coarse localization, while Dice drives precise topological boundary alignment.
================================================================================
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from config import TrainingConfig


def build_loss_criterion(training_cfg: TrainingConfig) -> nn.Module:
    """
    Constructs the DiceCELoss criterion for multi-class 3D segmentation.

    Args:
        training_cfg: TrainingConfig dataclass specifying weights and smoothing constants.

    Returns:
        nn.Module: Configured DiceCELoss instance.
    """
    criterion = DiceCELoss(
        to_onehot_y=True,          # Automatically converts integer class labels (0..C-1) to one-hot tensors
        softmax=True,              # Applies softmax over output logits across the channel dimension
        squared_pred=True,         # Uses squared denominator in Dice formula for sharper boundary gradients
        smooth_nr=training_cfg.smooth_nr,  # Epsilon in numerator preventing division by zero
        smooth_dr=training_cfg.smooth_dr,  # Epsilon in denominator
        lambda_dice=training_cfg.dice_weight,
        lambda_ce=training_cfg.ce_weight
    )
    return criterion

