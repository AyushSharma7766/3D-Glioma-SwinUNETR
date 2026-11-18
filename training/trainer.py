"""
================================================================================
Trainer Module: High-Performance H200 Training Engine with AMP & MLflow
================================================================================
Academic Defense Context:
NVIDIA H200 GPU Architecture Optimizations:
1. AUTOMATIC MIXED PRECISION (AMP) VIA BFLOAT16:
   Hopper (sm_90) introduces 4th-generation Tensor Cores with native Brain Floating
   Point (BF16) execution pipelines:
   - FP16 vs BF16 Trade-off: FP16 allocates 5 bits to the exponent (range [-14, 15]),
     frequently triggering gradient underflow that requires dynamic loss scaling (GradScaler).
     BF16 allocates 8 bits to the exponent—identical to standard FP32 (range [-126, 127]).
     Gradient underflow/overflow is virtually eliminated.
   - Performance Gain: Eliminates GradScaler reduction barriers, reduces memory traffic
     by 50% compared to FP32, and utilizes Hopper's 3.9 PFLOPS BF16 Tensor Cores.

2. ASYNCHRONOUS GPU PIPELINING & MAXIMUM BATCHING:
   With 141 GB HBM3e (the largest single-GPU memory capacity in existence), we maximize
   batch size, significantly stabilizing batch statistics and variance in Vision
   Transformer gradient updates.
================================================================================
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import mlflow
from config import PipelineConfig
from evaluation.inferer import FullVolumeInferer


class H200SwinUNETRTrainer:
    """
    Production-grade training orchestrator for 3D Swin-UNETR on NVIDIA H200.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        cfg: PipelineConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Move model to target device
        self.model.to(self.device)

        # Hardware setup: cuDNN auto-tuner
        if self.device.type == "cuda" and cfg.hardware.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # Setup Automatic Mixed Precision (AMP)
        self.amp_enabled = cfg.hardware.amp_enabled and (self.device.type == "cuda")
        if cfg.hardware.amp_dtype.lower() == "bfloat16":
            self.amp_dtype = torch.bfloat16
            # BF16 possesses FP32 dynamic range; GradScaler is unnecessary on Hopper
            self.scaler = None
            print("[Hardware AMP] Using bfloat16 precision (Native Hopper H200 Tensor Cores). GradScaler: OFF.")
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
            print("[Hardware AMP] Using float16 precision with dynamic GradScaler.")

        # Optimizer: Decoupled weight decay (AdamW)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            eps=1e-8
        )

        # Learning Rate Schedule: Linear Warmup followed by Cosine Annealing Decay
        warmup_epochs = cfg.training.warmup_epochs
        total_epochs = cfg.training.max_epochs
        warmup_sched = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs)
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=cfg.training.min_learning_rate
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )

        # Full-Resolution Sliding Window Validation Inferer
        self.inferer = FullVolumeInferer(
            roi_size=cfg.data.roi_size,
            num_classes=cfg.model.out_channels,
            infer_cfg=cfg.inference,
            device=self.device
        )

        # Checkpoint tracking state
        self.best_metric = -1.0
        self.best_metric_epoch = -1

    def _get_autocast_context(self):
        """Returns the appropriate AMP autocast context manager."""
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled
        )

    def train_epoch(self, epoch: int) -> float:
        """
        Executes one full training epoch over all balanced 3D volumetric patches.
        """
        self.model.train()
        epoch_loss = 0.0
        step_count = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch_data in enumerate(self.train_loader):
            step_count += 1
            inputs = batch_data["image"].to(self.device, non_blocking=True)
            labels = batch_data["label"].to(self.device, non_blocking=True)

            with self._get_autocast_context():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # Backward pass under AMP
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.cfg.hardware.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.hardware.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.hardware.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.hardware.max_grad_norm
                    )
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()

        return epoch_loss / max(1, step_count)

    def train(self):
        """
        Main execution loop orchestrating training, sliding window validation,
        checkpoint persistence, and MLflow logging.
        """
        cfg = self.cfg
        print("\n" + "=" * 78)
        print(f"  STARTING TRAINING: {cfg.training.max_epochs} Epochs on {self.device}")
        print(f"  AMP Mode: {cfg.hardware.amp_dtype.upper()} | H200 Tensor Core Optimization: ACTIVE")
        print("=" * 78 + "\n")

        # Initialize MLflow tracking run
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        if cfg.mlflow.tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        with mlflow.start_run(run_name=cfg.mlflow.run_name):
            # Log all hyperparameter configurations
            mlflow.log_params({
                "model_name": "SwinUNETR",
                "spatial_dims": cfg.model.spatial_dims,
                "in_channels": cfg.model.in_channels,
                "out_channels": cfg.model.out_channels,
                "feature_size": cfg.model.feature_size,
                "roi_size": str(cfg.data.roi_size),
                "batch_size": cfg.data.batch_size,
                "learning_rate": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
                "amp_dtype": cfg.hardware.amp_dtype,
                "max_epochs": cfg.training.max_epochs,
                "val_interval": cfg.training.val_interval,
                "pos_sample_ratio": cfg.data.pos_sample_ratio,
                "neg_sample_ratio": cfg.data.neg_sample_ratio,
            })

            for epoch in range(1, cfg.training.max_epochs + 1):
                start_time = time.time()
                train_loss = self.train_epoch(epoch)
                self.scheduler.step()
                epoch_duration = time.time() - start_time
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Telemetry: GPU Memory Utilization
                gpu_mem_gb = 0.0
                if self.device.type == "cuda":
                    gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

                print(
                    f"Epoch [{epoch:03d}/{cfg.training.max_epochs:03d}] "
                    f"Train Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Time: {epoch_duration:.1f}s | H200 VRAM: {gpu_mem_gb:.2f} GB"
                )

                # Log training epoch metrics to MLflow
                metrics_to_log = {
                    "train_loss": train_loss,
                    "learning_rate": current_lr,
                    "epoch_time_sec": epoch_duration,
                    "gpu_vram_gb": gpu_mem_gb
                }

                # Validation step
                if epoch % cfg.training.val_interval == 0:
                    val_start = time.time()
                    print(f"  -> Running Full-Volume Sliding Window Inference on Validation Set...")
                    val_metrics = self.inferer.evaluate_dataset(
                        self.model,
                        self.val_loader,
                        self._get_autocast_context()
                    )
                    val_duration = time.time() - val_start
                    mean_dice = val_metrics["val_mean_dice"]
                    metrics_to_log.update(val_metrics)
                    metrics_to_log["val_time_sec"] = val_duration

                    print(f"  -> Validation Mean Dice: {mean_dice:.4f} (took {val_duration:.1f}s)")

                    # Checkpoint persistence (Best Model)
                    if mean_dice > self.best_metric:
                        self.best_metric = mean_dice
                        self.best_metric_epoch = epoch
                        best_ckpt_path = cfg.checkpoint_dir / "best_metric_model.pth"
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_metric": self.best_metric,
                            "config": cfg
                        }, best_ckpt_path)
                        print(f"  [Checkpoint] New best model saved to {best_ckpt_path} (Dice: {self.best_metric:.4f})")
                        if cfg.mlflow.log_artifacts:
                            mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")

                # Log aggregated epoch metrics to MLflow
                mlflow.log_metrics(metrics_to_log, step=epoch)

                # Regular periodic checkpoint
                if epoch % cfg.training.save_interval == 0:
                    latest_ckpt_path = cfg.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_metric": self.best_metric,
                    }, latest_ckpt_path)

        print("\n" + "=" * 78)
        print("  TRAINING COMPLETE")
        print(f"  Best Validation Mean Dice : {self.best_metric:.4f} at Epoch {self.best_metric_epoch}")
        print(f"  Best Checkpoint Path     : {cfg.checkpoint_dir / 'best_metric_model.pth'}")
        print("=" * 78)

