"""Training Loop and Fold Management Engine for SegFormer-B2.

Controls:
- Fixed 10-epoch training regime per fold.
- Mixed precision training with gradient accumulation and clipping.
- Validation after every epoch and tracking best val_flood_iou.
- Checkpoint saving and interruption resumption.
- Held-out test event evaluation.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.data.dataset import FloodPlanetDataset
from src.evaluation.evaluate import evaluate_dataset
from src.models.segformer import SegFormerFloodModel
from src.utils.logging import get_logger
from src.utils.seed import seed_everything
from src.visualization.visualize import plot_training_curves, save_prediction_panel

logger = get_logger("Trainer")


class FoldTrainer:
    """Trains SegFormer-B2 on one LOEO fold for exactly 10 epochs."""

    def __init__(
        self,
        config: Dict[str, Any],
        fold_spec: Dict[str, Any],
        manifest_df: pd.DataFrame,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.fold_spec = fold_spec
        self.manifest_df = manifest_df
        self.fold_id = fold_spec["fold_id"]
        self.test_event = fold_spec["test_event"]
        self.train_events = fold_spec["train_events"]
        self.val_events = fold_spec["val_events"]

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logger.info(f"Initializing FoldTrainer for {self.fold_id} (Test Event: {self.test_event}) on {self.device}")

        # Directories
        out_root = config["paths"].get("output_dir", "outputs")
        self.checkpoints_dir = config["paths"].get("checkpoints_dir", os.path.join(out_root, "checkpoints"))
        self.results_dir = config["paths"].get("results_dir", os.path.join(out_root, "results"))
        viz_root = config["paths"].get("visualizations_dir", os.path.join(out_root, "visualizations"))
        self.visualizations_dir = os.path.join(viz_root, self.test_event)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

        self.best_checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.fold_id}_best.pt")
        self.last_checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.fold_id}_last.pt")
        self.metrics_json_path = os.path.join(self.results_dir, f"{self.fold_id}_metrics.json")

        # Datasets
        self._setup_datasets()

    def _setup_datasets(self) -> None:
        """Create fold-specific train, val, and test datasets ensuring zero leakage."""
        data_cfg = self.config["data"]
        dataset_root = data_cfg["dataset_root"]
        input_size = data_cfg.get("input_size", 512)
        rgb_bands = tuple(data_cfg.get("rgb_bands", (2, 1, 0)))
        image_mean = tuple(data_cfg.get("image_mean", (0.485, 0.456, 0.406)))
        image_std = tuple(data_cfg.get("image_std", (0.229, 0.224, 0.225)))
        ignore_index = self.config["model"].get("ignore_index", 255)

        train_df = self.manifest_df[self.manifest_df["event"].isin(self.train_events)].copy()
        val_df = self.manifest_df[self.manifest_df["event"].isin(self.val_events)].copy()
        test_df = self.manifest_df[self.manifest_df["event"] == self.test_event].copy()

        assert len(train_df) > 0, f"No training samples for fold {self.fold_id}"
        assert len(val_df) > 0, f"No validation samples for fold {self.fold_id}"
        assert len(test_df) > 0, f"No test samples for fold {self.fold_id}"

        self.train_dataset = FloodPlanetDataset(
            manifest_df=train_df,
            dataset_root=dataset_root,
            split="train",
            input_size=input_size,
            rgb_bands=rgb_bands,
            image_mean=image_mean,
            image_std=image_std,
            ignore_index=ignore_index,
        )

        self.val_dataset = FloodPlanetDataset(
            manifest_df=val_df,
            dataset_root=dataset_root,
            split="val",
            input_size=input_size,
            rgb_bands=rgb_bands,
            image_mean=image_mean,
            image_std=image_std,
            ignore_index=ignore_index,
        )

        self.test_dataset = FloodPlanetDataset(
            manifest_df=test_df,
            dataset_root=dataset_root,
            split="test",
            input_size=input_size,
            rgb_bands=rgb_bands,
            image_mean=image_mean,
            image_std=image_std,
            ignore_index=ignore_index,
        )

        logger.info(
            f"Fold {self.fold_id} datasets ready: "
            f"Train={len(self.train_dataset)} samples, "
            f"Val={len(self.val_dataset)} tiles ({len(val_df)} chips), "
            f"Test={len(self.test_dataset)} tiles ({len(test_df)} chips)"
        )

    def is_completed(self) -> bool:
        """Check if this fold has already completed training and evaluation."""
        return os.path.exists(self.metrics_json_path) and os.path.exists(self.best_checkpoint_path)

    def train_and_evaluate(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Execute the full 10-epoch training and held-out test evaluation."""
        if self.is_completed() and not force_retrain:
            logger.info(f"Fold {self.fold_id} is already completed. Loading saved results from {self.metrics_json_path}")
            with open(self.metrics_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        train_cfg = self.config["training"]
        seed = train_cfg.get("seed", 42) + self.fold_spec["fold_index"]
        seed_everything(seed)

        # Fresh model initialization per fold
        model_cfg = self.config["model"]
        model = SegFormerFloodModel(
            encoder_checkpoint=model_cfg.get("encoder_checkpoint", "nvidia/mit-b2"),
            num_classes=model_cfg.get("num_classes", 2),
            ignore_index=model_cfg.get("ignore_index", 255),
            id2label=model_cfg.get("id2label"),
            label2id=model_cfg.get("label2id"),
            pretrained=True,
        ).to(self.device)

        batch_size = train_cfg.get("batch_size", 4)
        num_workers = self.config["data"].get("num_workers", 0)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True if len(self.train_dataset) > batch_size else False,
        )

        # Optimization setup
        lr = float(train_cfg.get("learning_rate", 6e-5))
        weight_decay = float(train_cfg.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_epochs = train_cfg.get("epochs", 10)  # Exactly 10 epochs
        accum_steps = train_cfg.get("gradient_accumulation_steps", 2)
        total_steps = (len(train_loader) // accum_steps) * total_epochs
        warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
        )

        use_amp = train_cfg.get("amp", True) and (self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start_epoch = 1
        best_val_metric = -1.0
        best_epoch = -1
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_flood_iou": [],
            "learning_rates": [],
        }

        # Check for resumption
        if os.path.exists(self.last_checkpoint_path) and not force_retrain:
            logger.info(f"Resuming fold {self.fold_id} from checkpoint: {self.last_checkpoint_path}")
            checkpoint = torch.load(self.last_checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if use_amp and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_metric = checkpoint.get("best_val_metric", -1.0)
            best_epoch = checkpoint.get("best_epoch", -1)
            history = checkpoint.get("history", history)
            logger.info(f"Resumed at epoch {start_epoch} (Best val metric so far: {best_val_metric:.4f} at epoch {best_epoch})")

        # Training loop
        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start_time = time.time()
            model.train()
            train_loss_total = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    outputs = model(images, labels=masks)
                    loss = outputs["loss"] / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    grad_clip = train_cfg.get("gradient_clipping", 1.0)
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                train_loss_total += loss.item() * accum_steps

                if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                    step_loss = loss.item() * accum_steps
                    pct = int((step + 1) / len(train_loader) * 100)
                    elapsed = time.time() - epoch_start_time
                    logger.info(
                        f"[{self.fold_id}] Epoch {epoch}/{total_epochs} | "
                        f"Batch {step + 1}/{len(train_loader)} ({pct:2d}%) | "
                        f"Loss: {step_loss:.4f} | Elapsed: {elapsed:.1f}s"
                    )

            avg_train_loss = train_loss_total / max(1, len(train_loader))
            current_lr = scheduler.get_last_lr()[0]

            # Validation at end of epoch
            logger.info(f"[{self.fold_id}] Validating epoch {epoch}/{total_epochs} on {len(self.val_dataset.samples)} chips...")
            val_metrics, _ = evaluate_dataset(
                model=model,
                dataset=self.val_dataset,
                batch_size=batch_size,
                device=self.device,
                num_workers=num_workers,
            )

            val_flood_iou = val_metrics["flood_iou"]
            val_loss = val_metrics["eval_loss"]

            history["train_loss"].append(round(avg_train_loss, 5))
            history["val_loss"].append(round(val_loss, 5))
            history["val_flood_iou"].append(round(val_flood_iou, 5))
            history["learning_rates"].append(current_lr)

            epoch_duration = time.time() - epoch_start_time
            logger.info(
                f"[{self.fold_id}] Epoch {epoch}/{total_epochs} ({epoch_duration:.1f}s) | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Flood IoU: {val_flood_iou:.4f} | LR: {current_lr:.2e}"
            )

            # Check for best validation metric
            is_best = val_flood_iou > best_val_metric
            if is_best:
                best_val_metric = val_flood_iou
                best_epoch = epoch
                logger.info(f"==> New best validation Flood IoU for {self.fold_id}: {best_val_metric:.4f} at epoch {epoch}")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "val_flood_iou": val_flood_iou,
                    "config": self.config,
                    "fold_spec": self.fold_spec,
                }, self.best_checkpoint_path)

            # Save last checkpoint for resume
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "best_val_metric": best_val_metric,
                "best_epoch": best_epoch,
                "history": history,
            }, self.last_checkpoint_path)

        # Plot training curves
        curve_path = os.path.join(self.visualizations_dir, f"{self.fold_id}_training_curves.png")
        plot_training_curves(history, curve_path, fold_name=f"{self.fold_id} ({self.test_event})")

        # Load best checkpoint for held-out test evaluation
        logger.info(f"Loading best checkpoint from epoch {best_epoch} for test evaluation on {self.test_event}...")
        best_ckpt = torch.load(self.best_checkpoint_path, map_location=self.device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        # Evaluate once on held-out test event
        test_metrics, test_cm = evaluate_dataset(
            model=model,
            dataset=self.test_dataset,
            batch_size=batch_size,
            device=self.device,
            num_workers=num_workers,
        )

        logger.info(
            f"=== Test Results for Held-Out Event: {self.test_event} ===\n"
            f"  Flood IoU:    {test_metrics['flood_iou']:.4f}\n"
            f"  Non-flood IoU:{test_metrics['non_flood_iou']:.4f}\n"
            f"  mIoU:         {test_metrics['miou']:.4f}\n"
            f"  F1 / Dice:    {test_metrics['f1']:.4f}\n"
            f"  Precision:    {test_metrics['precision']:.4f}\n"
            f"  Recall:       {test_metrics['recall']:.4f}\n"
            f"  Accuracy:     {test_metrics['accuracy']:.4f}\n"
            f"============================================================"
        )

        # Save representative visual predictions for test event
        self._save_test_visualizations(model)

        # Compile final fold results
        final_results = {
            "fold_id": self.fold_id,
            "test_event": self.test_event,
            "training_event_count": len(self.train_events),
            "validation_event_count": len(self.val_events),
            "test_image_count": len(self.test_dataset.samples),
            "best_validation_epoch": best_epoch,
            "best_val_flood_iou": round(best_val_metric, 6),
            "test_metrics": test_metrics,
            "history": history,
            "training_completed": True,
        }

        with open(self.metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)

        # Free GPU memory
        del model, optimizer, scheduler
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return final_results

    def _save_test_visualizations(self, model: nn.Module, max_samples: int = 3) -> None:
        """Generate and save 4-panel prediction visualizations for test chips."""
        model.eval()
        num_samples = min(max_samples, len(self.test_dataset.samples))
        
        for s_idx in range(num_samples):
            full_sample = self.test_dataset.get_full_sample(s_idx)
            rgb_img = full_sample["rgb_image"]
            target_mask = full_sample["target_mask"]
            chip_id = full_sample["filename"]
            tiles = full_sample["tiles"]
            full_shape = full_sample["shape"]

            # Run tile inference
            tile_logits = []
            tile_coords = []
            for ymin, xmin, ymax, xmax in tiles:
                crop_img = rgb_img[ymin:ymax, xmin:xmax]
                pad_h = self.config["data"]["input_size"] - crop_img.shape[0]
                pad_w = self.config["data"]["input_size"] - crop_img.shape[1]
                if pad_h > 0 or pad_w > 0:
                    crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
                
                from src.data.preprocessing import normalize_image
                norm_img = normalize_image(
                    crop_img,
                    mean=tuple(self.config["data"].get("image_mean", (0.485, 0.456, 0.406))),
                    std=tuple(self.config["data"].get("image_std", (0.229, 0.224, 0.225))),
                )
                tensor_crop = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                
                with torch.no_grad():
                    out = model(tensor_crop)
                    logit_tile = out["logits"].squeeze(0).cpu().numpy()
                tile_logits.append(logit_tile)
                tile_coords.append((ymin, xmin, ymax, xmax))

            from src.data.preprocessing import reconstruct_full_image_logits
            full_logits = reconstruct_full_image_logits(tile_logits, tile_coords, full_shape, num_classes=2)
            pred_mask = np.argmax(full_logits, axis=0)

            vis_path = os.path.join(self.visualizations_dir, f"{chip_id}_panel.png")
            save_prediction_panel(
                rgb_image=rgb_img,
                ground_truth=target_mask,
                prediction=pred_mask,
                save_path=vis_path,
                event_name=self.test_event,
                chip_id=chip_id,
                ignore_index=self.config["model"].get("ignore_index", 255),
            )
