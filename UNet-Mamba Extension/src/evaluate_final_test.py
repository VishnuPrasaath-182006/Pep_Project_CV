"""
Stage 8: Final Held-Out Test Evaluation (Single-Pass Reporting)
Evaluates locked checkpoints on the held-out test partition:
- Events: Bangladesh (15 scenes), Cambodia (20 scenes), Nepal (9 scenes) - Total 44 scenes
- Models:
  - E0: Frozen Baseline UNet-Mamba (checkpoints/best_model.pth)
  - E1: Locked CASR with Neutral Guidance (checkpoints/best_e1_model.pth)
  - E2: Locked CASR with RemoteCLIP Guidance (checkpoints/best_e2_model.pth)
- Metric outputs saved to results/final/
"""

import os
import sys
import time
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import load_config, BASE_DIR, TEST_EVENTS
from model import build_model
from casr import CASRAdapter
from candidate_selector import (
    select_candidate_regions,
    RemoteCLIPGuidance
)
from evaluate_source_chips import compute_binary_metrics


def run_final_test_evaluation():
    print("=" * 75)
    print("STAGE 8: FINAL HELD-OUT TEST EVALUATION (SINGLE PASS)")
    print("=" * 75)
    
    device = torch.device("cpu")
    config = load_config()
    
    # 1. Output Directories
    out_dirs = [
        BASE_DIR / "results" / "final",
        BASE_DIR.parent / "results" / "final"
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    log_file_path = out_dirs[0] / "final_test_evaluation.log"
    log_lines = []
    
    def log(msg):
        print(msg)
        log_lines.append(msg)
        
    log(f"Execution Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Test Events ({len(TEST_EVENTS)}): {TEST_EVENTS}")
    log(f"Execution Device: CPU (deterministic, seed 2026)")
    
    # 2. Checkpoint Loading
    ckpt_base_path = BASE_DIR / "checkpoints" / "best_model.pth"
    ckpt_e1_path = BASE_DIR / "checkpoints" / "best_e1_model.pth"
    ckpt_e2_path = BASE_DIR / "checkpoints" / "best_e2_model.pth"
    
    for p in [ckpt_base_path, ckpt_e1_path, ckpt_e2_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required checkpoint: {p}")
            
    # Load Base Model
    log("\nLoading Frozen Base UNet-Mamba...")
    base_model = build_model(config).to(device)
    base_ckpt = torch.load(ckpt_base_path, map_location=device, weights_only=False)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    log(f"  Base UNet-Mamba loaded from: {ckpt_base_path.name} (Epoch {base_ckpt.get('epoch', 'N/A')})")
    
    # Load RemoteCLIP Guidance
    log("\nLoading Frozen RemoteCLIP Guidance Module...")
    scaler_path = BASE_DIR.parent / "artifacts" / "CAVE_Flood_Antigravity" / "remoteclip_score_scaler.json"
    rc_guidance = RemoteCLIPGuidance(device="cpu", scaler_path=scaler_path)
    log(f"  RemoteCLIP ViT-B/32 initialized with scaler from: {scaler_path.name}")
    
    # Load CASR Adapters
    log("\nLoading Locked CASR Adapters...")
    adapter_e1 = CASRAdapter(in_channels=9, hidden_channels=32, num_classes=2, lambda_res=1.0)
    data_e1 = torch.load(ckpt_e1_path, map_location=device, weights_only=True)
    adapter_e1.load_state_dict(data_e1["casr_state_dict"])
    adapter_e1.eval()
    for param in adapter_e1.parameters():
        param.requires_grad = False
    log(f"  CASR E1 loaded from: {ckpt_e1_path.name} (Epoch {data_e1.get('epoch')})")
    
    adapter_e2 = CASRAdapter(in_channels=9, hidden_channels=32, num_classes=2, lambda_res=1.0)
    data_e2 = torch.load(ckpt_e2_path, map_location=device, weights_only=True)
    adapter_e2.load_state_dict(data_e2["casr_state_dict"])
    adapter_e2.eval()
    for param in adapter_e2.parameters():
        param.requires_grad = False
    log(f"  CASR E2 loaded from: {ckpt_e2_path.name} (Epoch {data_e2.get('epoch')})")
    
    # 3. Load Test Manifest
    test_manifest = BASE_DIR / "manifests" / "test.csv"
    df_test = pd.read_csv(test_manifest)
    total_scenes = len(df_test)
    log(f"\nTest Manifest Loaded: {test_manifest.name} ({total_scenes} scenes across {df_test['event'].nunique()} events)")
    
    patches_dir = BASE_DIR / "manifests" / "processed_data" / "processed_data" / "test"
    norm_mean = np.array(config["data"]["norm_mean"], dtype=np.float32).reshape(4, 1, 1)
    norm_std = np.array(config["data"]["norm_std"], dtype=np.float32).reshape(4, 1, 1)
    
    quadrant_specs = [
        (0, 0, "p0_0_0"),
        (0, 512, "p1_0_512"),
        (512, 0, "p2_512_0"),
        (512, 512, "p3_512_512")
    ]
    
    # 4. Confusion Matrix Trackers
    experiments = ["E0", "E1", "E2"]
    event_counts = {
        exp: {evt: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "chips": 0} for evt in TEST_EVENTS}
        for exp in experiments
    }
    pooled_counts = {
        exp: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "chips": 0}
        for exp in experiments
    }
    
    total_valid_pixels_counted = 0
    start_eval_time = time.time()
    
    log("\n" + "=" * 75)
    log(f"BEGINNING INFERENCE PASS ACROSS ALL {total_scenes} TEST SCENES...")
    log("=" * 75)
    
    for idx in tqdm(range(total_scenes), desc="Evaluating Test Scenes"):
        row = df_test.iloc[idx]
        event_name = row["event"]
        stem = row["stem"]
        
        # 1024x1024 full chip arrays
        recon_img = np.zeros((4, 1024, 1024), dtype=np.uint16)
        recon_mask = np.full((1024, 1024), -1, dtype=np.int64)
        
        logits_e0 = np.zeros((2, 1024, 1024), dtype=np.float32)
        logits_e1 = np.zeros((2, 1024, 1024), dtype=np.float32)
        logits_e2 = np.zeros((2, 1024, 1024), dtype=np.float32)
        
        # Load quadrants
        for top, left, q_tag in quadrant_specs:
            patch_file = patches_dir / f"{event_name}_{stem}_{q_tag}.npz"
            p_data = np.load(patch_file)
            
            raw_img = p_data["image"]      # (4, 512, 512)
            raw_mask = p_data["mask"]      # (512, 512)
            
            recon_img[:, top:top+512, left:left+512] = raw_img
            recon_mask[top:top+512, left:left+512] = raw_mask
            
            # 1. Base UNetMamba Forward Pass
            img_norm = (raw_img.astype(np.float32) - norm_mean) / (norm_std + 1e-6)
            x_t = torch.from_numpy(img_norm).unsqueeze(0).float()
            
            with torch.no_grad():
                z_base = base_model(x_t)
                if isinstance(z_base, (tuple, list)):
                    z_base = z_base[0]
                    
            z_base_np = z_base.squeeze(0).numpy()
            logits_e0[:, top:top+512, left:left+512] = z_base_np
            
            # 2. Flood prob & Candidate Region Selection
            flood_prob = np.exp(z_base_np[1] - np.logaddexp(z_base_np[0], z_base_np[1]))
            p_valid = (raw_img != 0).any(axis=0) & (raw_mask != -1)
            cand_res = select_candidate_regions(
                flood_prob=flood_prob,
                valid_mask=p_valid,
                patch_size=512,
                region_size=128,
                top_k=4,
                min_coverage=0.5
            )
            cand_mask = cand_res["candidate_mask"]
            selected_regions = cand_res["selected_regions"]
            entropy_map = cand_res["entropy_map"]
            grad_map = cand_res["gradient_map"]
            
            # 3. Guidance Channels
            # E1 Guidance: All zeros
            guidance_e1 = np.zeros((512, 512), dtype=np.float32)
            
            # E2 Guidance: Scaled RemoteCLIP guidance
            guidance_e2 = np.zeros((512, 512), dtype=np.float32)
            if len(selected_regions) > 0:
                guidance_e2, _ = rc_guidance.build_guidance_map(raw_img, selected_regions)
                
            # 4. Assemble 9-channel Feature Tensors
            def make_features(g_map):
                return np.concatenate([
                    img_norm,
                    flood_prob[np.newaxis, ...],
                    entropy_map[np.newaxis, ...],
                    grad_map[np.newaxis, ...],
                    cand_mask[np.newaxis, ...],
                    g_map[np.newaxis, ...]
                ], axis=0).astype(np.float32)
                
            feat_e1 = torch.from_numpy(make_features(guidance_e1)).unsqueeze(0).float()
            feat_e2 = torch.from_numpy(make_features(guidance_e2)).unsqueeze(0).float()
            cand_t = torch.from_numpy(cand_mask).unsqueeze(0).unsqueeze(0).float()
            
            with torch.no_grad():
                # E1 forward pass
                _, _, res1 = adapter_e1(feat_e1, cand_t)
                z1 = z_base + res1
                logits_e1[:, top:top+512, left:left+512] = z1.squeeze(0).numpy()
                
                # E2 forward pass
                _, _, res2 = adapter_e2(feat_e2, cand_t)
                z2 = z_base + res2
                logits_e2[:, top:top+512, left:left+512] = z2.squeeze(0).numpy()
                
        # 5. Full Chip Validity Mask & Confusion Calculation
        chip_valid = (recon_img != 0).any(axis=0) & (recon_mask != -1)
        valid_pixel_count = int(chip_valid.sum())
        total_valid_pixels_counted += valid_pixel_count
        
        y_true = recon_mask[chip_valid]
        
        preds = {
            "E0": np.argmax(logits_e0, axis=0)[chip_valid],
            "E1": np.argmax(logits_e1, axis=0)[chip_valid],
            "E2": np.argmax(logits_e2, axis=0)[chip_valid]
        }
        
        for exp in experiments:
            y_p = preds[exp]
            tp = int(np.sum((y_p == 1) & (y_true == 1)))
            fp = int(np.sum((y_p == 1) & (y_true == 0)))
            fn = int(np.sum((y_p == 0) & (y_true == 1)))
            tn = int(np.sum((y_p == 0) & (y_true == 0)))
            
            event_counts[exp][event_name]["tp"] += tp
            event_counts[exp][event_name]["fp"] += fp
            event_counts[exp][event_name]["fn"] += fn
            event_counts[exp][event_name]["tn"] += tn
            event_counts[exp][event_name]["chips"] += 1
            
            pooled_counts[exp]["tp"] += tp
            pooled_counts[exp]["fp"] += fp
            pooled_counts[exp]["fn"] += fn
            pooled_counts[exp]["tn"] += tn
            pooled_counts[exp]["chips"] += 1
            
    total_eval_time = time.time() - start_eval_time
    log(f"\nInference completed in {total_eval_time:.2f} seconds ({total_eval_time / total_scenes:.2f} s/scene)")
    log(f"Total Unique Valid Test Pixels Evaluated: {total_valid_pixels_counted:,}")
    
    # 6. Compute Final Metrics
    final_json = {
        "evaluation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_test_scenes": total_scenes,
        "total_valid_pixels": total_valid_pixels_counted,
        "evaluation_time_seconds": total_eval_time,
        "checkpoints": {
            "E0": str(ckpt_base_path.name),
            "E1": str(ckpt_e1_path.name),
            "E2": str(ckpt_e2_path.name)
        },
        "experiments": {}
    }
    
    event_rows = []
    summary_rows = []
    
    for exp in experiments:
        exp_res = {}
        # Pooled metrics
        p_c = pooled_counts[exp]
        p_met = compute_binary_metrics(p_c["tp"], p_c["fp"], p_c["fn"], p_c["tn"], p_c["chips"])
        p_met.update(p_c)
        exp_res["pooled"] = p_met
        
        # Per-event metrics
        exp_res["per_event"] = {}
        evt_ious = []
        evt_mious = []
        evt_dices = []
        
        for evt in TEST_EVENTS:
            c = event_counts[exp][evt]
            met = compute_binary_metrics(c["tp"], c["fp"], c["fn"], c["tn"], c["chips"])
            met.update(c)
            exp_res["per_event"][evt] = met
            
            evt_ious.append(met["flood_iou"])
            evt_mious.append(met["mean_iou"])
            evt_dices.append(met["flood_dice"])
            
            event_rows.append({
                "experiment": exp,
                "event": evt,
                "chips": c["chips"],
                "valid_pixels": c["tp"] + c["fp"] + c["fn"] + c["tn"],
                "flood_iou": met["flood_iou"],
                "bg_iou": met["bg_iou"],
                "mean_iou": met["mean_iou"],
                "flood_dice": met["flood_dice"],
                "flood_precision": met["flood_precision"],
                "flood_recall": met["flood_recall"],
                "specificity": met["specificity"],
                "pixel_accuracy": met["pixel_accuracy"],
                "tp": c["tp"],
                "fp": c["fp"],
                "fn": c["fn"],
                "tn": c["tn"]
            })
            
        mean_evt_iou = float(np.mean(evt_ious))
        std_evt_iou = float(np.std(evt_ious))
        mean_evt_miou = float(np.mean(evt_mious))
        mean_evt_dice = float(np.mean(evt_dices))
        
        exp_res["event_macro"] = {
            "mean_event_flood_iou": mean_evt_iou,
            "std_event_flood_iou": std_evt_iou,
            "mean_event_mean_iou": mean_evt_miou,
            "mean_event_dice": mean_evt_dice
        }
        
        final_json["experiments"][exp] = exp_res
        
        summary_rows.append({
            "experiment": exp,
            "mean_event_flood_iou": mean_evt_iou,
            "std_event_flood_iou": std_evt_iou,
            "pooled_flood_iou": p_met["flood_iou"],
            "pooled_flood_dice": p_met["flood_dice"],
            "pooled_macro_miou": p_met["mean_iou"],
            "pooled_precision": p_met["flood_precision"],
            "pooled_recall": p_met["flood_recall"],
            "pooled_accuracy": p_met["pixel_accuracy"],
            "tp": p_c["tp"],
            "fp": p_c["fp"],
            "fn": p_c["fn"],
            "tn": p_c["tn"]
        })
        
    df_event_metrics = pd.DataFrame(event_rows)
    df_summary = pd.DataFrame(summary_rows)
    
    # Save CSVs & JSONs
    for d in out_dirs:
        df_event_metrics.to_csv(d / "final_test_event_metrics.csv", index=False)
        df_summary.to_csv(d / "final_test_summary.csv", index=False)
        with open(d / "final_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)
            
    log("\n" + "=" * 75)
    log("FINAL TEST SUMMARY (OVERALL ACROSS HELD-OUT EVENTS)")
    log("=" * 75)
    log(df_summary[["experiment", "mean_event_flood_iou", "std_event_flood_iou", "pooled_flood_iou", "pooled_flood_dice", "pooled_macro_miou", "pooled_precision", "pooled_recall"]].to_string(index=False))
    
    log("\n" + "=" * 75)
    log("FINAL TEST EVENT-BY-EVENT BREAKDOWN")
    log("=" * 75)
    log(df_event_metrics[["experiment", "event", "chips", "flood_iou", "flood_dice", "flood_precision", "flood_recall"]].to_string(index=False))
    
    # Display differences in percentage points
    e0_sum = df_summary[df_summary["experiment"] == "E0"].iloc[0]
    e1_sum = df_summary[df_summary["experiment"] == "E1"].iloc[0]
    e2_sum = df_summary[df_summary["experiment"] == "E2"].iloc[0]
    
    log("\n" + "=" * 75)
    log("ABSOLUTE DIFFERENCES (IN PERCENTAGE POINTS pp OVER BASELINE E0):")
    log("=" * 75)
    log(f"E1 Mean Event Flood IoU: {e1_sum['mean_event_flood_iou']:.4f} vs E0 {e0_sum['mean_event_flood_iou']:.4f} -> {(e1_sum['mean_event_flood_iou'] - e0_sum['mean_event_flood_iou'])*100:+.2f} pp")
    log(f"E2 Mean Event Flood IoU: {e2_sum['mean_event_flood_iou']:.4f} vs E0 {e0_sum['mean_event_flood_iou']:.4f} -> {(e2_sum['mean_event_flood_iou'] - e0_sum['mean_event_flood_iou'])*100:+.2f} pp")
    log(f"E1 Pooled Flood IoU:     {e1_sum['pooled_flood_iou']:.4f} vs E0 {e0_sum['pooled_flood_iou']:.4f} -> {(e1_sum['pooled_flood_iou'] - e0_sum['pooled_flood_iou'])*100:+.2f} pp")
    log(f"E2 Pooled Flood IoU:     {e2_sum['pooled_flood_iou']:.4f} vs E0 {e0_sum['pooled_flood_iou']:.4f} -> {(e2_sum['pooled_flood_iou'] - e0_sum['pooled_flood_iou'])*100:+.2f} pp")
    log(f"E1 Pooled Dice:          {e1_sum['pooled_flood_dice']:.4f} vs E0 {e0_sum['pooled_flood_dice']:.4f} -> {(e1_sum['pooled_flood_dice'] - e0_sum['pooled_flood_dice'])*100:+.2f} pp")
    log(f"E2 Pooled Dice:          {e2_sum['pooled_flood_dice']:.4f} vs E0 {e0_sum['pooled_flood_dice']:.4f} -> {(e2_sum['pooled_flood_dice'] - e0_sum['pooled_flood_dice'])*100:+.2f} pp")
    
    # Save log file
    for d in out_dirs:
        with open(d / "final_test_evaluation.log", "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
            
    log("\n[STAGE 8 COMPLETE] Saved final test metrics, CSVs, JSON, and evaluation log.")

if __name__ == "__main__":
    run_final_test_evaluation()
