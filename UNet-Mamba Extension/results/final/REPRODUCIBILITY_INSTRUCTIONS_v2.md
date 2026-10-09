# CAVE-Flood v1: Scientific Reproducibility & Audit Guide (v2)

**Document**: `REPRODUCIBILITY_INSTRUCTIONS_v2.md`  
**Date**: October 9, 2026  
**Auditor**: Senior Computer Vision Research Engineer & Scientific Reproducibility Auditor  
**Project**: CAVE-Flood v1 — UNet-Mamba + CASR + RemoteCLIP  

This guide provides fully executable, non-destructive instructions to independently audit, verify, and reproduce the CAVE-Flood v1 results without initiating expensive GPU training.

---

## 1. Quick Verification Commands

From the workspace root (`c:\Users\AI\Pictures\BaseModel Extension`):

### 1.1 Verify Checkpoint Integrity & SHA-256 Hashes
Verifies that original and locked model weights have not been modified:
```powershell
Get-FileHash -Algorithm SHA256 `
  "UNet-Mamba\checkpoints\best_model.pth", `
  "UNet-Mamba\checkpoints\best_e1_model.pth", `
  "UNet-Mamba\checkpoints\best_e2_model.pth" | Format-Table -AutoSize
```
**Expected Hashes**:
- `best_model.pth`: `58674C84E97E16C7AB86F2E218BA8738C7AAF69A8C6FD8C384DE00BEEC9EB76D`
- `best_e1_model.pth`: `84A02D1F1FE8371BF799EDB0DDCE28F69D20DD90B3DA503C06EAC87F521226E9`
- `best_e2_model.pth`: `19347F5F64385C46F16C27B345A7F143E56B51F23C1FE7E887906B6E7C19D2D9`

### 1.2 Run Mathematical Reconciliation & Consistency Audit
Verifies that all event and pooled confusion counts sum perfectly and all metric formulas match:
```powershell
python "UNet-Mamba/src/audit_reconciliation.py"
```
**Expected Output**:
- `[E0] Confusion Matrix Sum Check: PASS (TP: 22,201,266 | FP: 2,046,482 | FN: 4,365,354 | TN: 17,524,242)`
- `[E1] Confusion Matrix Sum Check: PASS (TP: 22,425,765 | FP: 2,267,106 | FN: 4,140,855 | TN: 17,303,618)`
- `[E2] Confusion Matrix Sum Check: PASS (TP: 22,404,736 | FP: 2,241,494 | FN: 4,161,884 | TN: 17,329,230)`
- All per-event formula derivations match to $< 10^{-7}$.

### 1.3 Run Demonstration Inference Pipeline
Tests end-to-end forward pass on raw 4-band PlanetScope imagery:
```powershell
python UNet-Mamba/src/demo_inference.py --patch_path artifacts/CAVE_Flood_Antigravity/cache/candidate_regions/val/Bolivia_BOL_1015_p0.npz --mode E2
```
**Expected Output**:
- Valid pixels: 262,144
- Baseline E0 Flood Area: 55.66%
- Refined E2 Flood Area: 56.26%
- Visualization output saved to `results/demo_output/demo_inference_Bolivia_BOL_1015_p0.png`.

### 1.4 Inspect Key Audit and Results Documents
- Root Cause Analysis & Reconciliation: `results/final/METRIC_RECONCILIATION_AUDIT.md`
- Corrected Final Results Report (Markdown): `CAVE_Flood_Final_Results_Report_v2.md`
- Corrected Final Results Report (HTML): `CAVE_Flood_Final_Results_Report_v2.html`
- Corrected Test Summary CSV: `results/final/final_test_summary_corrected_v2.csv`
- Corrected Test Event Metrics CSV: `results/final/final_test_event_metrics_corrected_v2.csv`

---

## 2. Directory Layout & Artifact Map

```
BaseModel Extension/
├── UNet-Mamba/
│   ├── checkpoints/
│   │   ├── best_model.pth          (Baseline UNet-Mamba, 171.8 MB, SHA256 verified)
│   │   ├── best_e1_model.pth       (Locked CASR Neutral, 54.4 KB, Epoch 5)
│   │   └── best_e2_model.pth       (Locked CASR RemoteCLIP, 54.4 KB, Epoch 5)
│   ├── src/
│   │   ├── casr.py                 (CASRAdapter module)
│   │   ├── candidate_selector.py   (Uncertainty ranking & RemoteCLIP guidance)
│   │   ├── demo_inference.py       (End-to-end demo visualizer)
│   │   ├── evaluate_final_test.py  (Single-pass test evaluator)
│   │   └── audit_reconciliation.py (Mathematical verification script)
│   └── manifests/                  (Data split manifests: train.csv, val.csv, test.csv)
├── results/
│   ├── demo_output/                (Visualized demo masks)
│   └── final/
│       ├── METRIC_RECONCILIATION_AUDIT.md
│       ├── CAVE_Flood_Final_Results_Report_v2.md
│       ├── CAVE_Flood_Final_Results_Report_v2.html
│       ├── final_test_summary_corrected_v2.csv
│       ├── final_test_event_metrics_corrected_v2.csv
│       ├── final_test_metrics_corrected_v2.json
│       ├── STAGE7_RELEASE_AUDIT.md
│       └── final_test_evaluation.log
├── CAVE_Flood_Final_Results_Report_v2.md
├── CAVE_Flood_Final_Results_Report_v2.html
├── SHA256SUMS_v2.txt
├── CAVE_Flood_Final_Results.zip    (Original release archive preserved)
└── CAVE_Flood_Final_Results_v2.zip (Versioned corrected release archive)
```
