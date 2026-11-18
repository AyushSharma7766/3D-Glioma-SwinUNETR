"""
================================================================================
Configuration Module for 3D Swin-UNETR Whole-Brain MRI Segmentation
Optimized for NVIDIA H200 GPU (Hopper Architecture, 141 GB HBM3e)
================================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import torch


@dataclass
class ModelConfig:
    """
    Swin-UNETR Architectural Hyperparameters.
    
    Academic Justification:
    - spatial_dims=3: Fully 3D volumetric convolutions and 3D shifted window self-attention.
    - in_channels=1: Monomodal structural MRI (e.g., T1-weighted MPRAGE/MP2RAGE).
    - out_channels=14: Multi-class segmentation of 14 key anatomical structures
      (Background, Cerebral WM, Cerebral Cortex, Lateral Ventricles, Inferior Lateral Ventricles,
       Cerebellum WM, Cerebellum Cortex, Thalamus, Caudate, Putamen, Pallidum, 3rd Ventricle,
       Brainstem, Hippocampus, Amygdala).
    - feature_size=48: Initial embedding dimension (C=48). Higher embedding dimensions
      preserve rich contextual representations for fine subcortical boundaries.
    - depths=(2, 2, 2, 2): Number of Swin Transformer blocks at each hierarchical stage.
    - num_heads=(3, 6, 12, 24): Multi-head self-attention heads scaled across pyramid depths.
    - drop_rate=0.0, attn_drop_rate=0.0: Regularization via stochastic depth.
    - norm_name='instance': Instance normalization is empirically superior to batch
      normalization for small batch volumetric medical image segmentation.
    """
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 14
    feature_size: int = 48
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    norm_name: str = "instance"
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    use_checkpoint: bool = True  # Gradient checkpointing: recomputes activations to save VRAM (needed when GPU is shared)


@dataclass
class DataConfig:
    """
    Data Pipeline and MONAI Transform Hyperparameters.
    
    Academic Justification:
    - roi_size=(96, 96, 96): Volumetric patch size. Large 3D receptive field captures
      macro-structural cranial context while keeping self-attention memory tractable.
    - pixdim=(1.0, 1.0, 1.0): Isotropic 1mm voxel spacing standardization, eliminating
      anisotropic slice thickness variance across diverse MRI scanner acquisitions.
    - orientation='RAS': Standardized Right-Anterior-Superior anatomical coordinate system.
    - num_samples_per_volume=4: Number of random positive/negative balanced crops per volume.
    - pos_ratio=1.0, neg_ratio=1.0: 50% foreground / 50% background sampling balance,
      preventing background-dominated gradient starvation.
    """
    roi_size: Tuple[int, int, int] = (96, 96, 96)
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    orientation: str = "RAS"
    num_samples_per_volume: int = 4
    pos_sample_ratio: float = 1.0
    neg_sample_ratio: float = 1.0
    cache_rate: float = 1.0  # Cache 100% of preprocessed volumes in system RAM
    num_workers: int = 0  # 0 disables /dev/shm IPC to prevent Docker shared memory exhaustion
    pin_memory: bool = True
    batch_size: int = 4  # Aggressive batch size enabled by 141 GB HBM3e
    val_batch_size: int = 1


@dataclass
class HardwareOptimizationConfig:
    """
    NVIDIA H200 (Hopper sm_90) Specific Acceleration Settings.
    
    Academic Justification:
    - AMP Precision (bfloat16): Hopper 4th-gen Tensor Cores provide up to 3.9 PFLOPS
      with native BF16 arithmetic. BF16 possesses an 8-bit dynamic range identical
      to FP32 (exponent range [-126, 127]), completely eliminating numerical underflow
      and removing the computational overhead of dynamic loss scaling (GradScaler).
    - torch.compile: Compiles the SwinUNETR graph into fused Triton/CUDA kernels,
      optimizing memory bandwidth utilization on the H200's 4.8 TB/s HBM3e bus.
    - cudnn_benchmark: Enables cuDNN heuristic auto-tuner to select optimal convolution algorithms.
    """
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"  # Options: 'bfloat16' (recommended on H200) or 'float16'
    use_torch_compile: bool = False  # Optional JIT Inductor compilation
    cudnn_benchmark: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0  # Gradient clipping threshold for Transformer stability


@dataclass
class TrainingConfig:
    """
    Optimization and Learning Schedule Configuration.
    
    Academic Justification:
    - Optimizer (AdamW): Decoupled weight decay regularization prevents over-fitting
      on over-parameterized Vision Transformer self-attention projection weights.
    - Scheduler (CosineAnnealingWarmRestarts / CosineAnnealingLR): Warmup stabilizes
      stochastic attention maps in early epochs; cosine decay ensures smooth convergence.
    - Loss (DiceCELoss): Fuses distribution-based soft Dice loss (combating volumetric
      class imbalance) with voxel-wise Cross-Entropy loss (providing strong gradient signals
      for initial convergence).
    """
    max_epochs: int = 200
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    val_interval: int = 2
    save_interval: int = 10
    
    # Loss weighting
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    smooth_nr: float = 1e-5
    smooth_dr: float = 1e-5


@dataclass
class InferenceConfig:
    """
    Full-Resolution Sliding Window Validation Configuration.
    
    Academic Justification:
    - sliding_window_inference: Evaluates arbitrary-sized full-brain volumes (e.g., 256x256x176)
      at native resolution without distortion from downsampling/interpolation.
    - overlap=0.5: 50% window overlap balances spatial continuity and compute cost.
    - mode='gaussian': Gaussian weighting assigns higher importance to central patch voxels,
      attenuating boundary stitching artifacts at patch seams.
    """
    roi_size: Tuple[int, int, int] = (96, 96, 96)
    sw_batch_size: int = 4
    overlap: float = 0.5
    blending_mode: str = "gaussian"


@dataclass
class MLflowConfig:
    """
    Experiment Lifecycle and Provenance Tracking.
    """
    experiment_name: str = "3D-Brain-SwinUNETR-Segmentation"
    run_name: Optional[str] = "H200-SwinUNETR-Baseline"
    tracking_uri: Optional[str] = "file:./mlruns"
    log_artifacts: bool = True


@dataclass
class PipelineConfig:
    """Master configuration aggregating all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareOptimizationConfig = field(default_factory=HardwareOptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    
    # File system paths
    data_dir: Path = Path("data/brain_mri")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

