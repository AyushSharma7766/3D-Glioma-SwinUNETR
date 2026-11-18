"""
================================================================================
SOTA 3D Swin-UNETR Training on BraTS (MSD Task01 Brain Tumour)
Multi-GPU DDP · Pre-Trained Encoder · BF16 AMP · NVIDIA H200
================================================================================

Academic Defense Context:
-------------------------
This pipeline implements the complete state-of-the-art methodology from
Hatamizadeh et al. (CVPR 2022, Nature Methods 2025) for 3D brain tumour
segmentation using Swin-UNETR with the following SOTA techniques:

1. SELF-SUPERVISED PRE-TRAINING (Transfer Learning):
   The Swin Transformer encoder is initialized from weights pre-trained on
   5,050 unlabeled brain MRI volumes via a 3D masked volume inpainting task.
   This pretext task forces the encoder to learn rich, hierarchical volumetric
   representations of brain anatomy (gyral folding, tissue boundaries, ventricular
   geometry) BEFORE any segmentation-specific fine-tuning. Published results show
   a +2-3% Dice improvement over random initialization.

2. MULTI-MODAL INPUT (4-Channel MRI):
   BraTS provides four complementary MRI sequences per patient:
   - T1-weighted: structural anatomy, gray/white matter contrast
   - T1 contrast-enhanced (T1ce): gadolinium-enhancing tumour boundaries
   - T2-weighted: edema and fluid hyperintensity
   - FLAIR: perilesional edema with CSF suppression
   Each modality contributes orthogonal diagnostic information; jointly they
   enable precise delineation of overlapping tumour sub-regions.

3. HIERARCHICAL TUMOUR REGION EVALUATION (BraTS Protocol):
   BraTS defines three nested, overlapping evaluation regions:
   - ET (Enhancing Tumour): Active, contrast-enhancing tumour core (label 4)
   - TC (Tumour Core): ET + necrotic/non-enhancing core (labels 1 + 4)
   - WT (Whole Tumour): TC + peritumoral edema (labels 1 + 2 + 4)
   Loss and metrics are computed per-region using sigmoid activation (NOT softmax)
   because these regions overlap hierarchically.

4. MULTI-GPU DDP (Distributed Data Parallel):
   Scales training across multiple NVIDIA H200 GPUs with synchronized gradient
   averaging, linear learning rate scaling, and per-GPU batch sharding.

Usage:
  # Single GPU:
  python3 train_brats.py --data_dir brats_data --gpus 1

  # Multi-GPU (4x H200):
  CUDA_VISIBLE_DEVICES=0,5,6,7 torchrun --nproc_per_node=4 train_brats.py \
      --data_dir brats_data --gpus 4

================================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass

try:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, max(rlimit[1], 65536)))
except Exception:
    pass

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# MONAI imports
from monai.data import (
    CacheDataset,
    Dataset,
    decollate_batch,
    list_data_collate,
    load_decathlon_datalist,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)
from monai.utils import set_determinism

import mlflow


# =============================================================================
#   BraTS-SPECIFIC TRANSFORMS
# =============================================================================

def get_brats_train_transforms(roi_size: Tuple[int, int, int] = (128, 128, 128), et_label: int = 3):
    """
    Constructs the training-time transform pipeline for BraTS multimodal MRI.

    Key design choices:
    - ConvertToMultiChannelBasedOnBratsClassesd: Converts single-channel integer
      labels into 3-channel binary masks for [TC, WT, ET] regions (et_label=3 for MSD Task01).
    - CropForegroundd: Removes empty background slices to focus compute on brain.
    - RandSpatialCropd: Extracts random 128³ patches for Swin-UNETR input.
    - Intensity augmentation: Scale ±10%, shift ±10% to simulate scanner variability.
    - Spatial augmentation: Random flips along all 3 axes (50% probability each).
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label", et_label=et_label),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=[roi_size[0], roi_size[1], roi_size[2]],
        ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=roi_size,
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])


def get_brats_val_transforms(et_label: int = 3):
    """
    Constructs the validation transform pipeline (no augmentation, full volume).
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label", et_label=et_label),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


# =============================================================================
#   DATASET PREPARATION
# =============================================================================

def prepare_brats_datalists(
    data_dir: str, val_frac: float = 0.2, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Loads and splits the MSD Task01 BrainTumour dataset into train/val sets.

    The dataset.json from MSD provides file paths in a standard format.
    We perform a reproducible random split using the specified seed.
    """
    data_dir = Path(data_dir)
    if (data_dir / "dataset.json").exists():
        task_dir = data_dir
    elif (data_dir / "Task01_BrainTumour" / "dataset.json").exists():
        task_dir = data_dir / "Task01_BrainTumour"
    else:
        task_dir = data_dir / "Task01_BrainTumour"
    json_path = task_dir / "dataset.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"dataset.json not found at {json_path}. "
            f"Run 'python3 download_brats.py --data_dir {data_dir}' first."
        )

    with open(json_path) as f:
        meta = json.load(f)

    # Build file list with absolute paths
    all_files = []
    for item in meta["training"]:
        img_path = str((task_dir / item["image"]).resolve())
        lbl_path = str((task_dir / item["label"]).resolve())
        all_files.append({"image": img_path, "label": lbl_path})

    # Reproducible random split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_files))
    n_val = int(len(all_files) * val_frac)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_files = [all_files[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]

    return train_files, val_files


# =============================================================================
#   MODEL CONSTRUCTION
# =============================================================================

def build_swin_unetr_brats(
    pretrained_path: Optional[str] = None,
    feature_size: int = 48,
    use_checkpoint: bool = True,
) -> SwinUNETR:
    """
    Constructs a 3D Swin-UNETR model configured for BraTS brain tumour segmentation.

    Architecture:
    - in_channels=4: Accepts 4-channel multimodal MRI (T1, T1ce, T2, FLAIR)
    - out_channels=3: Predicts 3 overlapping tumour regions (TC, WT, ET)
    - feature_size=48: Base embedding dimension (C) for the Swin encoder
    - use_checkpoint: Gradient checkpointing for O(sqrt(N)) memory

    Pre-trained weight loading:
    - model.load_from() maps pre-trained Swin encoder weights to the encoder
      submodule, leaving the randomly-initialized CNN decoder untouched.
    - This asymmetric initialization is intentional: the encoder benefits from
      rich anatomical priors, while the decoder learns task-specific boundary
      delineation from the labelled BraTS data.
    """
    import inspect
    init_params = inspect.signature(SwinUNETR.__init__).parameters

    kwargs = {
        "in_channels": 4,
        "out_channels": 3,
        "feature_size": feature_size,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.0,
        "use_checkpoint": use_checkpoint,
        "spatial_dims": 3,
    }
    # MONAI >= 1.4 removed img_size
    if "img_size" in init_params:
        kwargs["img_size"] = (128, 128, 128)

    model = SwinUNETR(**kwargs)

    # Load self-supervised pre-trained encoder weights
    if pretrained_path and Path(pretrained_path).exists():
        if feature_size != 48:
            print(f"[Pre-Training Warning] MONAI pre-trained weights are built for feature_size=48 (current is {feature_size}).")
            print("  -> Initializing feature_size={feature_size} model from scratch with random weights.")
        else:
            print(f"[Pre-Training] Loading self-supervised encoder weights from: {pretrained_path}")
            pretrained_weights = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            model.load_from(weights=pretrained_weights)
            print("[Pre-Training] Swin Transformer encoder initialized with 5,050-volume pre-trained weights")
    else:
        print("[Pre-Training] No pre-trained weights found. Training from random initialization.")

    # Parameter audit
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Swin-UNETR: {total:,} params ({total * 4 / 1024**2:.1f} MB FP32), {trainable:,} trainable")

    return model


# =============================================================================
#   TRAINING ENGINE
# =============================================================================

class BraTSTrainer:
    """
    State-of-the-art BraTS training engine with:
    - BF16 Automatic Mixed Precision (H200 Tensor Cores)
    - Sliding window validation with Gaussian blending
    - Per-region Dice metrics (ET, TC, WT)
    - Best-model checkpointing
    - MLflow experiment tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Wrap in DDP for multi-GPU
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)

        # Loss: DiceLoss with sigmoid (regions overlap in BraTS)
        self.criterion = DiceLoss(
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
        )

        # Optimizer: AdamW with decoupled weight decay
        base_lr = args.lr
        if world_size > 1:
            # Linear learning rate scaling rule (Goyal et al., 2017)
            base_lr *= world_size
            if self.is_main:
                print(f"[DDP] Scaled LR: {args.lr} × {world_size} GPUs = {base_lr}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=args.weight_decay,
        )

        # Scheduler: Cosine annealing with linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-7,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )

        # Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_thresh = AsDiscrete(threshold=0.5)

        # AMP
        self.amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

        # Tracking
        self.best_dice = 0.0
        self.best_epoch = 0

        # Directories
        self.checkpoint_dir = Path(args.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        """Runs a single training epoch."""
        self.model.train()
        epoch_loss = 0.0
        n_steps = 0

        for batch_data in self.train_loader:
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient clipping for Transformer stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        return epoch_loss / max(n_steps, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Runs validation with sliding window inference and Gaussian blending.
        Reports per-region Dice scores (TC, WT, ET) following BraTS protocol.
        """
        self.model.eval()
        roi_size = (self.args.roi_size, self.args.roi_size, self.args.roi_size)

        for batch_data in self.val_loader:
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                outputs = sliding_window_inference(
                    inputs,
                    roi_size=roi_size,
                    sw_batch_size=self.args.sw_batch_size,
                    predictor=self.model if not isinstance(self.model, DDP) else self.model.module,
                    overlap=0.5,
                    mode="gaussian",
                )

            # Post-process: sigmoid → threshold
            outputs_list = decollate_batch(outputs)
            labels_list = decollate_batch(labels)
            outputs_post = [self.post_thresh(self.post_sigmoid(x)) for x in outputs_list]
            labels_post = [x for x in labels_list]

            self.dice_metric(y_pred=outputs_post, y=labels_post)

        # Aggregate per-region Dice: shape (3,) for [TC, WT, ET]
        per_region_dice = self.dice_metric.aggregate().cpu().numpy()
        self.dice_metric.reset()

        metrics = {
            "dice_TC": float(per_region_dice[0]),
            "dice_WT": float(per_region_dice[1]),
            "dice_ET": float(per_region_dice[2]),
            "dice_mean": float(np.mean(per_region_dice)),
        }
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Saves model checkpoint if this is the best validation result."""
        if not self.is_main:
            return

        mean_dice = metrics["dice_mean"]
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        # Always save latest
        latest_path = self.checkpoint_dir / "latest_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dice": self.best_dice,
            "metrics": metrics,
        }, str(latest_path))

        # Save best
        if mean_dice > self.best_dice:
            self.best_dice = mean_dice
            self.best_epoch = epoch

            best_path = self.checkpoint_dir / "best_brats_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state,
                "metrics": metrics,
            }, str(best_path))

            print(
                f"  ★ NEW BEST MODEL: Mean Dice = {mean_dice:.4f} "
                f"(TC={metrics['dice_TC']:.4f}, WT={metrics['dice_WT']:.4f}, ET={metrics['dice_ET']:.4f}) "
                f"→ saved to {best_path}"
            )

    def train(self):
        """Full training lifecycle."""
        args = self.args

        if self.is_main:
            print("\n" + "=" * 78)
            print(f"  STARTING BraTS TRAINING: {args.epochs} Epochs on {self.world_size} GPU(s)")
            print(f"  AMP: {args.amp_dtype.upper()} | ROI: {args.roi_size}³ | Batch: {args.batch_size}")
            print(f"  Pre-trained: {'YES (5,050 volumes)' if args.pretrained else 'NO (random init)'}")
            print("=" * 78 + "\n")

            # MLflow
            mlflow.set_tracking_uri("file:./mlruns_brats")
            mlflow.set_experiment(args.experiment_name)
            mlflow.start_run(run_name=args.run_name)
            mlflow.log_params({
                "model": "SwinUNETR",
                "in_channels": 4,
                "out_channels": 3,
                "feature_size": args.feature_size,
                "roi_size": args.roi_size,
                "batch_size": args.batch_size * self.world_size,
                "learning_rate": args.lr * self.world_size,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "amp_dtype": args.amp_dtype,
                "num_gpus": self.world_size,
                "pretrained": args.pretrained,
            })

        for epoch in range(1, args.epochs + 1):
            start = time.time()

            # Set epoch for DistributedSampler
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            elapsed = time.time() - start

            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                print(
                    f"Epoch [{epoch:03d}/{args.epochs}] "
                    f"Loss: {train_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.1f}s | "
                    f"VRAM: {gpu_mem:.1f} GB"
                )
                mlflow.log_metrics({"train_loss": train_loss, "learning_rate": lr}, step=epoch)

            # Validation
            if epoch % args.val_interval == 0 or epoch == args.epochs:
                if self.is_main:
                    print("  → Running sliding window validation...")

                metrics = self.validate(epoch)

                if self.is_main:
                    print(
                        f"  → Val Dice: Mean={metrics['dice_mean']:.4f} | "
                        f"TC={metrics['dice_TC']:.4f} | "
                        f"WT={metrics['dice_WT']:.4f} | "
                        f"ET={metrics['dice_ET']:.4f}"
                    )
                    mlflow.log_metrics(metrics, step=epoch)
                    self.save_checkpoint(epoch, metrics)

        # Final summary
        if self.is_main:
            print("\n" + "=" * 78)
            print("  TRAINING COMPLETE")
            print(f"  Best Mean Dice : {self.best_dice:.4f} at Epoch {self.best_epoch}")
            print(f"  Best Checkpoint: {self.checkpoint_dir / 'best_brats_model.pth'}")
            print("=" * 78)
            mlflow.log_metrics({"best_dice": self.best_dice, "best_epoch": self.best_epoch})
            mlflow.end_run()


# =============================================================================
#   DISTRIBUTED TRAINING SETUP
# =============================================================================

def setup_ddp(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
#   MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SOTA 3D Swin-UNETR BraTS Training (Multi-GPU H200)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="brats_data")
    parser.add_argument("--output_dir", type=str, default="outputs_brats")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Fraction for validation split")

    # Architecture
    parser.add_argument("--feature_size", type=int, default=48, help="Swin-UNETR base feature dim")
    parser.add_argument("--roi_size", type=int, default=128, help="Cubic ROI patch size")
    parser.add_argument("--pretrained", type=str, default="pretrained/model_swinvit.pt",
                        help="Path to pre-trained Swin encoder weights")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to full SwinUNETR checkpoint (.pth) to resume or fine-tune from")
    parser.add_argument("--et_label", type=int, default=3,
                        help="Label index for Enhancing Tumor (3 for MSD Task01, 4 for raw BraTS)")

    # Training
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Linear warmup epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (scaled by GPU count)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--val_interval", type=int, default=5, help="Epochs between validations")

    # Hardware
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for DDP")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="MONAI CacheDataset cache rate")
    parser.add_argument("--sw_batch_size", type=int, default=4, help="Sliding window batch size")

    # MLflow
    parser.add_argument("--experiment_name", type=str, default="BraTS-SwinUNETR-H200")
    parser.add_argument("--run_name", type=str, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = f"H200-{args.gpus}GPU-fs{args.feature_size}-roi{args.roi_size}-ep{args.epochs}"
    return args


def main_worker(rank: int = 0, world_size: int = 1, args=None):
    """Main training worker (called per GPU in DDP)."""
    is_main = (rank == 0)

    if world_size > 1:
        setup_ddp(rank, world_size)

    set_determinism(args.seed)

    if is_main:
        print("=" * 78)
        print("  3D SWIN-UNETR BraTS BRAIN TUMOUR SEGMENTATION")
        print("  NVIDIA H200 · MONAI · Multi-GPU DDP · Pre-Trained Encoder")
        print("=" * 78)

    # 1. Prepare data lists
    train_files, val_files = prepare_brats_datalists(args.data_dir, args.val_frac, args.seed)
    if is_main:
        print(f"[Data] {len(train_files)} training + {len(val_files)} validation cases")

    # 2. Build transforms
    roi = (args.roi_size, args.roi_size, args.roi_size)
    train_transforms = get_brats_train_transforms(roi, et_label=args.et_label)
    val_transforms = get_brats_val_transforms(et_label=args.et_label)

    # 3. Build datasets
    if is_main:
        print(f"[Data] Building CacheDataset (cache_rate={args.cache_rate}, et_label={args.et_label})...")

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=args.cache_rate,
        num_workers=min(4, args.num_workers),
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=args.cache_rate,
        num_workers=min(4, args.num_workers),
    )

    # 4. Build DataLoaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=(args.num_workers > 0),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.num_workers > 0),
        persistent_workers=(args.num_workers > 0),
    )

    # 5. Build model
    model = build_swin_unetr_brats(
        pretrained_path=args.pretrained if not args.resume else None,
        feature_size=args.feature_size,
        use_checkpoint=True,
    )

    if args.resume and Path(args.resume).exists():
        if is_main:
            print(f"[Resume/Fine-Tune] Loading full model weights from: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        if is_main:
            print("[Resume/Fine-Tune] Full model weights loaded successfully for fine-tuning!")

    # 6. Train
    trainer = BraTSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        rank=rank,
        world_size=world_size,
    )
    trainer.train()

    if world_size > 1:
        cleanup_ddp()


def main():
    args = parse_args()

    if args.gpus > 1:
        # Multi-GPU: use torchrun
        # This path is reached when launched via: torchrun --nproc_per_node=N train_brats.py
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", args.gpus))
        main_worker(rank, world_size, args)
    else:
        # Single GPU
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()

