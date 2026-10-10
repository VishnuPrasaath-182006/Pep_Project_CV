#!/usr/bin/env python3
"""Comprehensive Results Evaluation and Inspection Tool for SegFormer-B2 LOEO-CV.

Displays accuracy, precision, recall, F1/Dice, specificity, and IoU metrics across
individual events, folds, and aggregated macro/pooled benchmarks.

Usage:
    python scripts/evaluate_results.py
    python scripts/evaluate_results.py --fold 0
    python scripts/evaluate_results.py --event Bangladesh
    python scripts/evaluate_results.py --sort precision
    python scripts/evaluate_results.py --sort accuracy
"""

import argparse
import json
import os
import sys
from typing import Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def format_percent(val: float) -> str:
    return f"{val * 100:6.2f}%"


def print_single_fold_evaluation(row: dict):
    print("\n" + "=" * 70)
    print(f"   DETAILED EVALUATION: {row['fold_id'].upper()} - {row['test_event'].upper()} (Held-out)")
    print("=" * 70)
    print(f"  Test Chips Count   : {row['test_chips_count']}")
    print(f"  Best Val Epoch     : {row.get('best_val_epoch', 'N/A')}")
    print(f"  Best Val Flood IoU : {row.get('best_val_flood_iou', 0.0):.4f}")
    print("-" * 70)
    print("  CONFUSION MATRIX (Pixel-level counts)")
    print("-" * 70)
    tp = int(row['tp'])
    tn = int(row['tn'])
    fp = int(row['fp'])
    fn = int(row['fn'])
    total = tp + tn + fp + fn
    print(f"  True Positives  (TP - Flood correctly detected) : {tp:12,d} ({tp/total*100:5.2f}%)")
    print(f"  True Negatives  (TN - Land correctly detected)  : {tn:12,d} ({tn/total*100:5.2f}%)")
    print(f"  False Positives (FP - False Alarm)              : {fp:12,d} ({fp/total*100:5.2f}%)")
    print(f"  False Negatives (FN - Missed Flood)             : {fn:12,d} ({fn/total*100:5.2f}%)")
    print(f"  Total Valid Pixels Evaluated                    : {total:12,d}")
    print("-" * 70)
    print("  PERFORMANCE METRICS")
    print("-" * 70)
    print(f"  Accuracy    : {row['accuracy']:.4f}  ({format_percent(row['accuracy'])})  [ (TP + TN) / Total ]")
    print(f"  Precision   : {row['precision']:.4f}  ({format_percent(row['precision'])})  [ TP / (TP + FP) - Reliability of flood alarms ]")
    print(f"  Recall      : {row['recall']:.4f}  ({format_percent(row['recall'])})  [ TP / (TP + FN) - Detection rate of flood water ]")
    print(f"  F1 / Dice   : {row['f1_score']:.4f}  ({format_percent(row['f1_score'])})  [ Harmonic mean of Precision & Recall ]")
    print(f"  Specificity : {row['specificity']:.4f}  ({format_percent(row['specificity'])})  [ TN / (TN + FP) - Correct land identification ]")
    print(f"  Flood IoU   : {row['flood_iou']:.4f}  ({format_percent(row['flood_iou'])})  [ TP / (TP + FP + FN) - Jaccard index ]")
    print(f"  mIoU        : {row['miou']:.4f}  ({format_percent(row['miou'])})  [ Mean of Flood IoU and Non-Flood IoU ]")
    print("-" * 70)
    print("  SCIENTIFIC EVALUATION & INTERPRETATION")
    print("-" * 70)
    if row['recall'] > row['precision']:
        diff = (row['recall'] - row['precision']) * 100
        print(f"  * Recall is {diff:.1f}% higher than Precision: The model has HIGH flood sensitivity")
        print(f"    (catches {format_percent(row['recall'])} of all flooded areas), but produces false alarms on moist soils/shadows.")
    else:
        diff = (row['precision'] - row['recall']) * 100
        print(f"  * Precision is {diff:.1f}% higher than Recall: The model is CONSERVATIVE")
        print(f"    (predictions are {format_percent(row['precision'])} pure flood), but under-segments shallow or turbid flood fringes.")
    print("=" * 70 + "\n")


def print_full_summary(event_df: pd.DataFrame, summary_data: dict, sort_by: str = "fold_id"):
    if sort_by in event_df.columns:
        event_df = event_df.sort_values(by=sort_by, ascending=(sort_by in ["fold_id", "test_event"]))

    print("\n" + "=" * 105)
    print("                    SEGFORMER-B2 LOEO-CV BENCHMARK RESULTS TABLE")
    print("=" * 105)
    header = f"{'Fold':<8} {'Held-Out Event':<14} {'Chips':>6} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'F1/Dice':>10} {'Flood IoU':>11} {'mIoU':>9}"
    print(header)
    print("-" * 105)

    for _, row in event_df.iterrows():
        f_id = str(row['fold_id'])
        evt = str(row['test_event'])
        chips = int(row['test_chips_count'])
        acc = f"{row['accuracy']:.4f}"
        prec = f"{row['precision']:.4f}"
        rec = f"{row['recall']:.4f}"
        f1 = f"{row['f1_score']:.4f}"
        iou = f"{row['flood_iou']:.4f}"
        miou = f"{row['miou']:.4f}"
        print(f"{f_id:<8} {evt:<14} {chips:>6d} {acc:>10} {prec:>11} {rec:>10} {f1:>10} {iou:>11} {miou:>9}")

    print("-" * 105)

    # Macro averages
    macro = summary_data.get("event_macro_mean", {})
    macro_std = summary_data.get("event_macro_std", {})
    if macro:
        m_acc = f"{macro.get('accuracy', 0.0):.4f}"
        m_prec = f"{macro.get('precision', 0.0):.4f}"
        m_rec = f"{macro.get('recall', 0.0):.4f}"
        m_f1 = f"{macro.get('f1', 0.0):.4f}"
        m_iou = f"{macro.get('flood_iou', 0.0):.4f}"
        m_miou = f"{macro.get('miou', 0.0):.4f}"
        total_chips = int(event_df['test_chips_count'].sum())
        print(f"{'MACRO':<8} {'Mean (19 evts)':<14} {total_chips:>6d} {m_acc:>10} {m_prec:>11} {m_rec:>10} {m_f1:>10} {m_iou:>11} {m_miou:>9}")
        if macro_std:
            std_iou = f"+/-{macro_std.get('flood_iou', 0.0):.4f}"
            std_f1 = f"+/-{macro_std.get('f1', 0.0):.4f}"
            std_miou = f"+/-{macro_std.get('miou', 0.0):.4f}"
            print(f"{'STD':<8} {'Variation':<14} {'':>6} {'':>10} {'':>11} {'':>10} {std_f1:>10} {std_iou:>11} {std_miou:>9}")

    print("=" * 105)

    # Pixel Pooled Global Metrics
    pooled = summary_data.get("pixel_pooled_metrics", {})
    if pooled:
        print("\n" + "=" * 70)
        print("          GLOBAL PIXEL-POOLED BENCHMARK METRICS (All 19 Events)")
        print("=" * 70)
        tp = pooled.get("tp", 0)
        tn = pooled.get("tn", 0)
        fp = pooled.get("fp", 0)
        fn = pooled.get("fn", 0)
        total_pix = pooled.get("total_valid_pixels", tp + tn + fp + fn)

        print(f"  Total Evaluated Pixels : {total_pix:,d} across 366 full-resolution chips")
        print(f"  Total True Positives   : {tp:,d}")
        print(f"  Total True Negatives   : {tn:,d}")
        print(f"  Total False Positives  : {fp:,d}")
        print(f"  Total False Negatives  : {fn:,d}")
        print("-" * 70)
        print(f"  Overall Accuracy       : {pooled.get('accuracy', 0.0):.4f}  ({format_percent(pooled.get('accuracy', 0.0))})")
        print(f"  Pooled Precision       : {pooled.get('precision', 0.0):.4f}  ({format_percent(pooled.get('precision', 0.0))})")
        print(f"  Pooled Recall          : {pooled.get('recall', 0.0):.4f}  ({format_percent(pooled.get('recall', 0.0))})")
        print(f"  Pooled F1 / Dice       : {pooled.get('f1', 0.0):.4f}  ({format_percent(pooled.get('f1', 0.0))})")
        print(f"  Pooled Specificity     : {pooled.get('specificity', 0.0):.4f}  ({format_percent(pooled.get('specificity', 0.0))})")
        print(f"  Pooled Flood IoU       : {pooled.get('flood_iou', 0.0):.4f}  ({format_percent(pooled.get('flood_iou', 0.0))})")
        print(f"  Pooled Mean IoU (mIoU) : {pooled.get('miou', 0.0):.4f}  ({format_percent(pooled.get('miou', 0.0))})")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="View and evaluate SegFormer-B2 accuracy, precision, and benchmark results.")
    parser.add_argument("--fold", type=int, default=None, help="Inspect specific fold index (0 to 18)")
    parser.add_argument("--event", type=str, default=None, help="Inspect specific event name (e.g., Bangladesh, Bolivia, etc.)")
    parser.add_argument("--sort", type=str, default="fold_id", choices=["fold_id", "accuracy", "precision", "recall", "f1_score", "flood_iou", "miou"], help="Sort table by metric")
    parser.add_argument("--results_dir", type=str, default="outputs/results", help="Directory containing results")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir)
    event_csv_path = os.path.join(results_dir, "event_metrics.csv")
    summary_json_path = os.path.join(results_dir, "summary_metrics.json")

    if not os.path.exists(event_csv_path):
        print(f"[ERROR] Event metrics CSV not found at: {event_csv_path}")
        print("Please run `python -m scripts.aggregate_results` first.")
        sys.exit(1)

    event_df = pd.read_csv(event_csv_path)

    summary_data = {}
    if os.path.exists(summary_json_path):
        with open(summary_json_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    # Check if user requested a specific fold or event
    if args.fold is not None:
        fold_str = f"fold_{args.fold:02d}"
        match = event_df[event_df["fold_id"] == fold_str]
        if match.empty:
            print(f"[ERROR] Fold {args.fold} ({fold_str}) not found in results.")
            sys.exit(1)
        print_single_fold_evaluation(match.iloc[0].to_dict())
        return

    if args.event is not None:
        match = event_df[event_df["test_event"].str.lower() == args.event.lower()]
        if match.empty:
            print(f"[ERROR] Event '{args.event}' not found in results.")
            print(f"Available events: {', '.join(event_df['test_event'].tolist())}")
            sys.exit(1)
        print_single_fold_evaluation(match.iloc[0].to_dict())
        return

    # Otherwise print full benchmark table and pooled metrics
    print_full_summary(event_df, summary_data, sort_by=args.sort)


if __name__ == "__main__":
    main()
