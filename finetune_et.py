"""
finetune_et.py — High-Speed ET Fine-Tuner for 3D Swin-UNETR on NVIDIA H200 Cluster
================================================================================
Fine-tunes the existing checkpoint (best_brats_model.pth) for 15 epochs on 4x H200 GPUs
with et_label=3 (MSD Task01 specification) to train Enhancing Tumor (ET) detection.
"""

import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_brats import (
    parse_args,
    prepare_brats_datalists,
    get_brats_train_transforms,
    get_brats_val_transforms,
    build_swin_unetr_brats,
    BraTSTrainer,
    setup_ddp,
    cleanup_ddp,
    CacheDataset,
    DataLoader,
    DistributedSampler,
    list_data_collate,
    set_determinism,
)


def main():
    args = parse_args()

    # Fine-tuning overrides
    args.et_label = 3
    args.warmup_epochs = 1
    if args.epochs == 300:
        args.epochs = 15
    if args.val_interval == 5:
        args.val_interval = 2

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", args.gpus))
    is_main = (rank == 0)

    if world_size > 1:
        setup_ddp(rank, world_size)

    set_determinism(args.seed)

    if is_main:
        print("=" * 78)
        print("  3D SWIN-UNETR BraTS ENHANCING TUMOR (ET) FINE-TUNING")
        print(f"  Target: 4x NVIDIA H200 | Epochs: {args.epochs} | ET Label: {args.et_label}")
        print("=" * 78)

    # 1. Prepare data lists
    train_files, val_files = prepare_brats_datalists(args.data_dir, args.val_frac, args.seed)
    if is_main:
        print(f"[Data] {len(train_files)} training + {len(val_files)} validation cases")

    # 2. Build transforms with et_label=3
    roi = (args.roi_size, args.roi_size, args.roi_size)
    train_transforms = get_brats_train_transforms(roi, et_label=args.et_label)
    val_transforms = get_brats_val_transforms(et_label=args.et_label)

    # 3. Datasets
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

    # 4. DataLoaders
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

    # 5. Build model & load existing checkpoint
    model = build_swin_unetr_brats(
        pretrained_path=None,
        feature_size=args.feature_size,
        use_checkpoint=True,
    )

    resume_path = args.resume
    if not resume_path or not Path(resume_path).exists():
        default_ckpt = Path(args.output_dir) / "checkpoints" / "best_brats_model.pth"
        if default_ckpt.exists():
            resume_path = str(default_ckpt)

    if resume_path and Path(resume_path).exists():
        if is_main:
            print(f"[Fine-Tune] Loading starting weights from: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        if is_main:
            print("[Fine-Tune] Checkpoint weights loaded! Swin-UNETR ready for ET fine-tuning.")
    else:
        if is_main:
            print("[Warning] No checkpoint found to resume from! Training from scratch.")

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


if __name__ == "__main__":
    main()

