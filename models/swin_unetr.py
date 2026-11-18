"""
================================================================================
Model Module: 3D Swin-UNETR Architecture for Volumetric Medical Segmentation
================================================================================
Academic Defense Context:
Swin-UNETR (Hatamizadeh et al., CVPR 2022 / Nature Methods) reformulates 3D medical
image segmentation by marrying a hierarchical Swin Transformer encoder with a
fully-convolutional U-Net residual decoder.

Core Architectural Justifications:
----------------------------------
1. THE DUALITY PROBLEM IN VOLUMETRIC MEDICAL IMAGING:
   Conventional 3D CNNs (e.g., 3D U-Net, V-Net) operate with localized convolutional
   kernels (typically 3x3x3). While translationally equivariant and parameter-efficient,
   their effective receptive field grows sub-linearly with network depth. In whole-brain
   MRI segmentation, accurate classification of symmetrical or multi-focal structures
   (e.g., bilateral caudate nuclei, lateral ventricles, cortical gray matter distribution)
   inherently depends on long-range anatomical context across the entire cranial vault.

2. WHY VISION TRANSFORMERS (ViT) FAIL NAIVELY IN 3D:
   Standard Vision Transformers (e.g., UNETr, TransUNet) compute global self-attention
   over flattened patch tokens:
       Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
   For a volumetric ROI of size 96x96x96 with patch size 2x2x2, the number of tokens is:
       N = (96/2) * (96/2) * (96/2) = 48^3 = 110,592 tokens!
   Global self-attention complexity is O(N^2), yielding ~1.22 * 10^10 attention matrix
   entries per head per layer. This leads to catastrophic out-of-memory (OOM) states
   even on enterprise-grade hardware.

3. THE SWIN TRANSFORMER SOLUTION (Shifted Window Self-Attention):
   Swin-UNETR constrains self-attention to non-overlapping 3D local windows of size
   (M x M x M, typically 7x7x7). Computational complexity drops to O(4 * M^3 * N), which
   is strictly linear with respect to volumetric token count N:
   - Window-based Multi-head Self-Attention (W-MSA): Local intra-window token mixing.
   - Shifted Window Multi-head Self-Attention (SW-MSA): In alternate layers, window
     boundaries are translated cyclically by (M//2, M//2, M//2). Cross-window connections
     are formed with zero extra parameter overhead, bridged via cyclic shifting and
     attention masking.

4. RESIDUAL CONVOLUTIONAL DECODER & SKIP CONNECTIONS:
   While the Swin encoder captures high-level global semantic representations, exact
   voxel boundary delineation (e.g., cerebral cortex-to-CSF interface, sub-millimeter
   hippocampal subfield borders) requires high-frequency spatial gradients.
   Swin-UNETR routes feature representations from the encoder stages (at 1x, 1/2x, 1/4x,
   1/8x, 1/16x resolutions) directly to the decoder stages via residual blocks and
   transposed convolutions, synthesizing sharp, topological segmentation masks.
================================================================================
"""

import torch
import torch.nn as nn
import inspect
from typing import Tuple, Optional
from monai.networks.nets import SwinUNETR
from config import ModelConfig, HardwareOptimizationConfig


def build_swin_unetr(
    model_cfg: ModelConfig,
    roi_size: Tuple[int, int, int],
    hw_cfg: Optional[HardwareOptimizationConfig] = None
) -> nn.Module:
    """
    Instantiates and configures a 3D Swin-UNETR model for whole-brain MRI segmentation.

    Args:
        model_cfg: ModelConfig dataclass specifying depths, heads, feature size, and channels.
        roi_size: Volumetric patch dimension (H, W, D) fed into the patch partition layer.
        hw_cfg: Hardware configuration for compilation and acceleration.

    Returns:
        nn.Module: Configured 3D Swin-UNETR instance.
    """
    # Defensive check: ensure spatial dimensions are divisible by 32 (5 downsampling stages)
    for dim in roi_size:
        if dim % 32 != 0:
            raise ValueError(
                f"Swin-UNETR spatial ROI dimension {dim} must be divisible by 32 "
                f"(current: {roi_size}). Recommended: (96, 96, 96) or (128, 128, 128)."
            )

    # Dynamic parameter inspection: MONAI >= 1.4 removed `img_size` in favor of dynamic resolution
    init_params = inspect.signature(SwinUNETR.__init__).parameters
    swin_kwargs = {
        "in_channels": model_cfg.in_channels,
        "out_channels": model_cfg.out_channels,
        "feature_size": model_cfg.feature_size,
        "depths": model_cfg.depths,
        "num_heads": model_cfg.num_heads,
        "norm_name": model_cfg.norm_name,
        "drop_rate": model_cfg.drop_rate,
        "attn_drop_rate": model_cfg.attn_drop_rate,
        "dropout_path_rate": model_cfg.dropout_path_rate,
        "use_checkpoint": model_cfg.use_checkpoint,
        "spatial_dims": model_cfg.spatial_dims,
    }
    if "img_size" in init_params:
        swin_kwargs["img_size"] = roi_size

    model = SwinUNETR(**swin_kwargs)

    # Architectural Model Summary & Parameter Audit
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 78)
    print("  3D SWIN-UNETR ARCHITECTURAL SPECIFICATION")
    print("=" * 78)
    print(f"  Spatial Dimensions        : {model_cfg.spatial_dims}D Volumetric")
    print(f"  Input Channels            : {model_cfg.in_channels} (Structural T1w MRI)")
    print(f"  Output Anatomical Classes : {model_cfg.out_channels}")
    print(f"  Receptive ROI Size        : {roi_size}")
    print(f"  Base Feature Embedding (C): {model_cfg.feature_size}")
    print(f"  Hierarchical Depths       : {model_cfg.depths}")
    print(f"  Multi-Head Self-Attention : {model_cfg.num_heads}")
    print(f"  Normalization             : {model_cfg.norm_name.upper()}")
    print(f"  Gradient Checkpointing    : {'ENABLED' if model_cfg.use_checkpoint else 'DISABLED (H200 Native VRAM)'}")
    print(f"  Total Parameters          : {total_params:,} ({total_params * 4 / (1024**2):.2f} MB in FP32)")
    print(f"  Trainable Parameters      : {trainable_params:,}")
    print("=" * 78)

    # Optional Hopper-specific PyTorch 2.0 Graph Compilation
    if hw_cfg and hw_cfg.use_torch_compile and hasattr(torch, "compile"):
        print("[Optimization] Compiling Swin-UNETR graph with torch.compile (Inductor backend)...")
        model = torch.compile(model, mode="reduce-overhead")

    return model


if __name__ == "__main__":
    # Self-test: instantiate model on dummy input to verify forward tensor shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    test_roi = (96, 96, 96)
    net = build_swin_unetr(cfg, test_roi).to(device)

    print("\nRunning dummy forward pass validation...")
    dummy_input = torch.randn(1, 1, *test_roi, device=device)
    with torch.no_grad():
        out = net(dummy_input)
    print(f"Forward pass successful! Input: {dummy_input.shape} -> Output Logits: {out.shape}")
    assert out.shape == (1, cfg.out_channels, *test_roi), "Output shape mismatch!"

