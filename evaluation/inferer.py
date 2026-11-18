"""
================================================================================
Inference Module: Full-Resolution Sliding Window Volumetric Evaluation
================================================================================
Academic Defense Context:
1. THE RESOLUTION DILEMMA IN BRAIN SEGMENTATION:
   Volumetric brain MRIs span millions of voxels (e.g., 256 x 256 x 176 = 11.5M voxels).
   Downsampling the entire scan to fit in GPU memory distorts partial volume effects,
   obliterates sub-millimeter anatomical structures (e.g., hippocampal subfields), and
   introduces irreversible spatial interpolation blur.

2. SLIDING WINDOW INFERENCE WITH GAUSSIAN BLENDING:
   `sliding_window_inference` evaluates the native-resolution 3D volume by partitioning it
   into overlapping patches of size `roi_size` (e.g., 96 x 96 x 96):
   - Overlap (typically 50%): Every physical voxel is sampled by multiple adjacent windows.
   - Gaussian Importance Weighting (`mode='gaussian'`): Predictions near the perimeter of
     a patch suffer from reduced receptive field support and padding artifacts.
     A 3D Gaussian kernel assigns maximum weight to the central patch voxels while decaying
     towards the patch boundary:
         w(x, y, z) = exp( - ( (x-cx)^2 + (y-cy)^2 + (z-cz)^2 ) / (2 * sigma^2) )
     Predictions from overlapping windows are accumulated and normalized by the sum of
     Gaussian weights, completely eliminating seamlines, boundary discontinuities, and
     stitching artifacts.
================================================================================
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from config import InferenceConfig


class FullVolumeInferer:
    """
    Evaluator performing sliding-window inference on arbitrary-sized full 3D NIfTI volumes.
    """
    def __init__(
        self,
        roi_size: Tuple[int, int, int],
        num_classes: int,
        infer_cfg: InferenceConfig,
        device: torch.device
    ):
        self.roi_size = roi_size
        self.num_classes = num_classes
        self.sw_batch_size = infer_cfg.sw_batch_size
        self.overlap = infer_cfg.overlap
        self.mode = infer_cfg.blending_mode
        self.device = device

        # Post-processing transforms
        # 1. Output logits -> argmax integer mask -> one-hot (C, H, W, D)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        # 2. Ground-truth integer mask -> one-hot (C, H, W, D)
        self.post_label = AsDiscrete(to_onehot=num_classes)

        # Multi-class Dice metric evaluator (background class 0 is excluded from anatomical score)
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False
        )
        self.dice_metric_per_class = DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False
        )

    @torch.no_grad()
    def predict_volume(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """
        Executes sliding window inference on a single 3D volume.

        Args:
            model: Neural network in eval mode.
            image: Input tensor of shape (1, C_in, H, W, D).

        Returns:
            torch.Tensor: Predicted probability logits of shape (1, C_out, H, W, D).
        """
        model.eval()
        logits = sliding_window_inference(
            inputs=image,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=model,
            overlap=self.overlap,
            mode=self.mode,
            device=self.device
        )
        return logits

    @torch.no_grad()
    def evaluate_dataset(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        amp_autocast_context
    ) -> Dict[str, float]:
        """
        Iterates over the validation DataLoader, running full-resolution sliding window
        inference on each patient volume and computing global & per-class Dice scores.

        Returns:
            Dict containing 'val_mean_dice' and 'val_dice_class_{i}' metrics.
        """
        model.eval()
        self.dice_metric.reset()
        self.dice_metric_per_class.reset()

        for batch_data in val_loader:
            val_inputs = batch_data["image"].to(self.device)
            val_labels = batch_data["label"].to(self.device)

            with amp_autocast_context:
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=model,
                    overlap=self.overlap,
                    mode=self.mode,
                    device=self.device
                )

            # Convert predictions and ground truth to one-hot for Dice metric computation
            val_outputs_list = [self.post_pred(i) for i in val_outputs]
            val_labels_list = [self.post_label(i) for i in val_labels]

            self.dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            self.dice_metric_per_class(y_pred=val_outputs_list, y=val_labels_list)

        mean_dice = float(self.dice_metric.aggregate().item())
        class_dices = self.dice_metric_per_class.aggregate().cpu().numpy().tolist()

        metrics = {"val_mean_dice": mean_dice}
        for class_idx, class_score in enumerate(class_dices):
            # Class index offset by 1 because background (class 0) is excluded
            metrics[f"val_dice_class_{class_idx + 1}"] = float(class_score)

        return metrics

