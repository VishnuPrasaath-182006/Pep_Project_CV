# CAVE-Flood v1: Final Report Metric Reconciliation & Integrity Audit

**Document**: `results/final/REPORT_METRIC_RECONCILIATION.md`  
**Date**: October 9, 2026  
**Auditor**: Independent Senior Computer Vision Research Engineer, PyTorch Evaluation Specialist & Scientific Reproducibility Auditor  
**Project**: CAVE-Flood v1 — UNet-Mamba + CASR + RemoteCLIP  
**Status**: COMPLETE & VERIFIED — ALL DISCREPANCIES RECONCILED AGAINST RAW TEST LABELS & EVALUATOR ARTIFACTS

---

## 1. Executive Summary & Purpose

This document provides a strict, item-by-item reconciliation of all numerical discrepancies identified between the original experimental report (`CAVE_Flood_Final_Results_Report.md`), the published confusion matrices, the single-pass evaluator outputs (`results/final/final_test_summary.csv`, `final_test_metrics.json`), and the raw physical segmentation labels.

Every discrepancy is analyzed across:
- **Original Reported Value**
- **Recomputed Value (from Published Confusion Matrix & Evaluator)**
- **Mathematical Metric Definition**
- **Authoritative Source Artifact**
- **Resolution Status**
- **Scientific Limitations / Remaining Caveats**

---

## 2. Master Item-by-Item Metric Reconciliation Table

| Metric / Parameter | Original Reported Value | Recomputed Value | Metric Definition | Source Artifact | Discrepancy Resolved? | Remaining Limitation / Caveats |
| :--- | :---: | :---: | :--- | :--- | :---: | :--- |
| **Held-Out Test Accuracy (E0 Baseline)** | `0.965421` | **`0.861027`** (Evaluator) <br> *`0.929817` (Synthetic CM)* | $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{FN} + \text{TN}}$ | `results/final/final_test_summary.csv` (col: `pooled_accuracy`) | **RESOLVED** | `0.965421` was an unverified hallucinated value. True test accuracy on 46.14M valid pixels is 86.10%. |
| **Held-Out Test Accuracy (E1 CASR)** | `0.965128` | **`0.861111`** (Evaluator) <br> *`0.929861` (Synthetic CM)* | $\frac{\text{TP} + \text{TN}}{\text{Total Valid}}$ | `results/final/final_test_summary.csv` (col: `pooled_accuracy`) | **RESOLVED** | True test accuracy is 86.11% (+0.01 pp over E0). |
| **Held-Out Test Accuracy (E2 CASR)** | `0.965215` | **`0.861211`** (Evaluator) <br> *`0.929912` (Synthetic CM)* | $\frac{\text{TP} + \text{TN}}{\text{Total Valid}}$ | `results/final/final_test_summary.csv` (col: `pooled_accuracy`) | **RESOLVED** | True test accuracy is 86.12% (+0.02 pp over E0). |
| **Pooled Macro mIoU (E0 Baseline)** | `0.754020` | **`0.754020`** (Evaluator) <br> *`0.841599` (Synthetic CM)* | $\frac{\text{IoU}_{\text{flood}} + \text{IoU}_{\text{bg}}}{2}$ | `results/final/final_test_summary.csv` (col: `pooled_macro_miou`) | **RESOLVED** | The reported text value `0.754020` was correct from the evaluator. The discrepancy arose because calculating Macro mIoU on the published synthetic CM yields `0.841599`. |
| **Pooled Macro mIoU (E1 CASR)** | `0.753758` | **`0.753758`** (Evaluator) <br> *`0.842405` (Synthetic CM)* | $\frac{\text{IoU}_{\text{flood}} + \text{IoU}_{\text{bg}}}{2}$ | `results/final/final_test_summary.csv` (col: `pooled_macro_miou`) | **RESOLVED** | Authoritative value is `0.753758`. From synthetic CM, it was `0.842405`. |
| **Pooled Macro mIoU (E2 CASR)** | `0.753955` | **`0.753955`** (Evaluator) <br> *`0.842433` (Synthetic CM)* | $\frac{\text{IoU}_{\text{flood}} + \text{IoU}_{\text{bg}}}{2}$ | `results/final/final_test_summary.csv` (col: `pooled_macro_miou`) | **RESOLVED** | Authoritative value is `0.753955`. From synthetic CM, it was `0.842433`. |
| **E0 Confusion Matrix (TP, FP, FN, TN)** | TP: 11,211,885<br>FP: 1,033,485<br>FN: 2,204,591<br>TN: 31,687,383 | **TP: 22,201,266**<br>**FP: 2,046,482**<br>**FN: 4,365,354**<br>**TN: 17,524,242** | Integer pixel classifications on valid pixels | `results/final/final_test_summary.csv` | **RESOLVED** | Original matrix was reverse-engineered by LLM to match ratios; true matrix matches raw test masks to exact integer (Flood GT = 26,566,620). |
| **E1 Confusion Matrix (TP, FP, FN, TN)** | TP: 11,325,258<br>FP: 1,144,792<br>FN: 2,091,218<br>TN: 31,576,076 | **TP: 22,425,765**<br>**FP: 2,267,106**<br>**FN: 4,140,855**<br>**TN: 17,303,618** | Integer pixel classifications on valid pixels | `results/final/final_test_summary.csv` | **RESOLVED** | True counts locked. Sum of TP+FP+FN+TN = 46,137,344 invariant. |
| **E2 Confusion Matrix (TP, FP, FN, TN)** | TP: 11,314,639<br>FP: 1,131,850<br>FN: 2,101,837<br>TN: 31,589,018 | **TP: 22,404,736**<br>**FP: 2,241,494**<br>**FN: 4,161,884**<br>**TN: 17,329,230** | Integer pixel classifications on valid pixels | `results/final/final_test_summary.csv` | **RESOLVED** | True counts locked. Sum of TP+FP+FN+TN = 46,137,344 invariant. |
| **Mean Event Flood IoU Difference (E2 vs E1)** | `+0.15 pp` (misstated as E2 vs E1) | **`+0.02 pp`** | $\text{IoU}_{\text{event}}^{(E2)} - \text{IoU}_{\text{event}}^{(E1)}$ | `results/final/final_test_summary.csv` | **RESOLVED** | E2 achieves `0.754468` vs E1's `0.754259`, which is **+0.02 pp**, not +0.15 pp. (+0.15 pp is E2 vs E0 baseline: $0.754468 - 0.752949$). |
| **Validation E1/E2 Confusion Counts & Accuracy** | Val report table listed mismatched accuracy | **E0 Val Acc: 0.806841**<br>**E1 Val Acc: 0.809112**<br>**E2 Val Acc: 0.808845** | $\frac{\text{TP} + \text{TN}}{\text{Total Valid}}$ | `UNet-Mamba/results/metrics/e0_val_metrics_unique_source_pixels.json` | **RESOLVED** | Validated across 114 chips (117,355,812 valid pixels). E1 Val TP: 33,806,312; E2 Val TP: 33,736,541. |
| **Bangladesh E2 Flood IoU Typo** | `0.7554` (in Final Report v1) | **`0.5754`** | $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$ for Bangladesh | `results/final/final_test_event_metrics.csv` | **RESOLVED** | Corrected transposition typo. |
| **Test Domain Geography Description** | "Unseen test continents" | **"Held-out Asian flood events"** | Geographic classification of test split | `Dataset/test/` (Bangladesh, Cambodia, Nepal) | **RESOLVED** | All three held-out test events are in Asia. |
| **Full-System Parameter Accounting** | "14.30M total parameters (0.084% trainable)" | **UNet-Mamba + CASR: 14,306,053**<br>*(or ~102.16M including frozen RemoteCLIP ViT-B/32)* | Total parameter count of all loaded modules | `UNet-Mamba/src/casr.py` & RemoteCLIP weights | **RESOLVED** | Explicitly labeled as UNet-Mamba + CASR only, noting RemoteCLIP adds ~87.85M frozen parameters. |

---

## 3. Discrepancy Breakdown & Forensic Explanation

### 3.1 Why Test Accuracy Was Reported as ~0.965
1. **The Phenomenon**: The text of the initial report stated pooled pixel accuracy was `0.965421` (E0), `0.965128` (E1), and `0.965215` (E2).
2. **The Forensic Trace**:
   - On the report's published synthetic confusion matrix ($TP = 11.21\text{M}, TN = 31.69\text{M}$), accuracy is $\frac{11,211,885 + 31,687,383}{46,137,344} = \mathbf{0.929817}$.
   - On the true physical labels ($TP = 22.20\text{M}, TN = 17.52\text{M}$), accuracy is $\frac{22,201,266 + 17,524,242}{46,137,344} = \mathbf{0.861027}$.
   - The figure `0.965421` did not match either matrix. It was an unverified, hallucinated figure generated during LLM text synthesis.
3. **Resolution**: All deliverables and reports now present the authoritative ground-truth accuracy:
   - **E0**: **`0.861027`** (86.10%)
   - **E1**: **`0.861111`** (86.11%, +0.01 pp)
   - **E2**: **`0.861211`** (86.12%, +0.02 pp)

### 3.2 Why Macro mIoU Appeared Inconsistent
1. **The Phenomenon**: The report stated Macro mIoU was `0.754020` (E0), `0.753758` (E1), and `0.753955` (E2), but evaluating the published synthetic confusion matrix gave `0.841599`, `0.842405`, and `0.842433`.
2. **The Forensic Trace**:
   - The reported text numbers (`0.754020`, `0.753758`, `0.753955`) were **100% genuine and correct**—they were directly copied from `results/final/final_test_summary.csv` produced by the evaluator.
   - However, the authoring agent published an artificial confusion matrix that did not generate those numbers:
     $$\text{Synthetic BG IoU} = \frac{31,687,383}{31,687,383 + 1,033,485 + 2,204,591} = 0.907286 \implies \frac{0.775911 + 0.907286}{2} = \mathbf{0.841599}$$
     $$\text{True Evaluator BG IoU} = \frac{17,524,242}{17,524,242 + 2,046,482 + 4,365,354} = 0.732127 \implies \frac{0.775913 + 0.732127}{2} = \mathbf{0.754020}$$
3. **Resolution**: Once the true authoritative confusion matrix is restored, the Macro mIoU formula $(\text{Flood IoU} + \text{BG IoU}) / 2$ evaluates to **`0.754020`** identically.

---

## 4. Recomputation of All Primary Metrics from Authoritative Confusion Counts

All metrics below are computed from the authoritative counts in `results/final/final_test_summary.csv` ($N = 46,137,344$ valid pixels):

### 4.1 E0 Baseline (UNet-Mamba)
- $\text{TP} = 22,201,266$, $\text{FP} = 2,046,482$, $\text{FN} = 4,365,354$, $\text{TN} = 17,524,242$
- **Flood IoU**: $\frac{22,201,266}{22,201,266 + 2,046,482 + 4,365,354} = \frac{22,201,266}{28,613,102} = \mathbf{0.775913}$ [Pixel-Pooled]
- **Background IoU**: $\frac{17,524,242}{17,524,242 + 2,046,482 + 4,365,354} = \frac{17,524,242}{23,936,078} = \mathbf{0.732127}$ [Pixel-Pooled]
- **Macro mIoU**: $\frac{0.775913 + 0.732127}{2} = \mathbf{0.754020}$ [Pixel-Pooled]
- **Flood Dice (F1)**: $\frac{2 \times 22,201,266}{2 \times 22,201,266 + 2,046,482 + 4,365,354} = \frac{44,402,532}{50,814,368} = \mathbf{0.873818}$ [Pixel-Pooled]
- **Flood Precision**: $\frac{22,201,266}{22,201,266 + 2,046,482} = \frac{22,201,266}{24,247,748} = \mathbf{0.915601}$ [Pixel-Pooled]
- **Flood Recall**: $\frac{22,201,266}{22,201,266 + 4,365,354} = \frac{22,201,266}{26,566,620} = \mathbf{0.835683}$ [Pixel-Pooled]
- **Specificity (TNR)**: $\frac{17,524,242}{17,524,242 + 2,046,482} = \frac{17,524,242}{19,570,724} = \mathbf{0.895431}$ [Pixel-Pooled]
- **Pixel Accuracy**: $\frac{22,201,266 + 17,524,242}{46,137,344} = \frac{39,725,508}{46,137,344} = \mathbf{0.861027}$ [Pixel-Pooled]
- **Mean Event Flood IoU**: $\frac{0.571064 (\text{BGD}) + 0.810344 (\text{KHM}) + 0.877438 (\text{NPL})}{3} = \mathbf{0.752949}$ [Mean Across Events]

### 4.2 E1 CASR (Neutral Guidance $g=0$)
- $\text{TP} = 22,425,765$, $\text{FP} = 2,267,106$, $\text{FN} = 4,140,855$, $\text{TN} = 17,303,618$
- **Flood IoU**: $\frac{22,425,765}{28,833,726} = \mathbf{0.777762}$ (+0.18 pp over E0) [Pixel-Pooled]
- **Background IoU**: $\frac{17,303,618}{23,711,579} = \mathbf{0.729754}$ (-0.24 pp) [Pixel-Pooled]
- **Macro mIoU**: $\frac{0.777762 + 0.729754}{2} = \mathbf{0.753758}$ (-0.03 pp) [Pixel-Pooled]
- **Flood Dice (F1)**: $\frac{44,851,530}{51,259,495} = \mathbf{0.874990}$ (+0.12 pp) [Pixel-Pooled]
- **Flood Precision**: $\frac{22,425,765}{24,692,871} = \mathbf{0.908188}$ (-0.74 pp) [Pixel-Pooled]
- **Flood Recall**: $\frac{22,425,765}{26,566,620} = \mathbf{0.844133}$ (+0.85 pp) [Pixel-Pooled]
- **Specificity (TNR)**: $\frac{17,303,618}{19,570,724} = \mathbf{0.884158}$ (-1.13 pp) [Pixel-Pooled]
- **Pixel Accuracy**: $\frac{39,729,383}{46,137,344} = \mathbf{0.861111}$ (+0.01 pp) [Pixel-Pooled]
- **Mean Event Flood IoU**: $\frac{0.574577 + 0.813946 + 0.874253}{3} = \mathbf{0.754259}$ (+0.13 pp over E0) [Mean Across Events]

### 4.3 E2 CASR (RemoteCLIP Guidance $g$)
- $\text{TP} = 22,404,736$, $\text{FP} = 2,241,494$, $\text{FN} = 4,161,884$, $\text{TN} = 17,329,230$
- **Flood IoU**: $\frac{22,404,736}{28,808,114} = \mathbf{0.777723}$ (+0.18 pp over E0) [Pixel-Pooled]
- **Background IoU**: $\frac{17,329,230}{23,732,608} = \mathbf{0.730187}$ (-0.19 pp) [Pixel-Pooled]
- **Macro mIoU**: $\frac{0.777723 + 0.730187}{2} = \mathbf{0.753955}$ (-0.01 pp) [Pixel-Pooled]
- **Flood Dice (F1)**: $\frac{44,809,472}{51,212,850} = \mathbf{0.874965}$ (+0.11 pp) [Pixel-Pooled]
- **Flood Precision**: $\frac{22,404,736}{24,646,230} = \mathbf{0.909053}$ (-0.65 pp) [Pixel-Pooled]
- **Flood Recall**: $\frac{22,404,736}{26,566,620} = \mathbf{0.843342}$ (+0.77 pp) [Pixel-Pooled]
- **Specificity (TNR)**: $\frac{17,329,230}{19,570,724} = \mathbf{0.885467}$ (-1.00 pp) [Pixel-Pooled]
- **Pixel Accuracy**: $\frac{39,733,966}{46,137,344} = \mathbf{0.861211}$ (+0.02 pp) [Pixel-Pooled]
- **Mean Event Flood IoU**: $\frac{0.575396 + 0.813292 + 0.874718}{3} = \mathbf{0.754468}$ (+0.15 pp over E0, **+0.02 pp over E1**) [Mean Across Events]

---

## 5. Distinction Between Aggregation Levels

To eliminate nomenclature confusion:

1. **Pixel-Pooled Metrics**:
   - Confusion counts are accumulated over all pixels across all scenes in the partition before applying the formula.
   - Example: Pooled Flood IoU $\text{TP}_{\text{total}} / (\text{TP}_{\text{total}} + \text{FP}_{\text{total}} + \text{FN}_{\text{total}})$.
   - Strongly weighted by larger scenes with more valid pixels.
2. **Mean Event Flood IoU (Primary Evaluated Metric)**:
   - Arithmetic average of each event's pooled IoU: $\frac{1}{3} (\text{IoU}_{\text{BGD}} + \text{IoU}_{\text{KHM}} + \text{IoU}_{\text{NPL}})$.
   - Prevents larger events (Cambodia, 20 chips) from overshadowing smaller events (Nepal, 9 chips).
   - Authoritative values: E0 = `0.752949`, E1 = `0.754259`, E2 = `0.754468`.

---

## 6. Audit Conclusion & Verification Sign-Off

All metrics, confusion counts, and aggregation methods are now 100% reconciled across:
- `results/final/CAVE_Flood_Final_Report_CORRECTED.md`
- `results/final/REPORT_METRIC_RECONCILIATION.md`
- `results/final/final_test_summary_corrected_v2.csv`
- `results/final/final_test_metrics_corrected_v2.json`
- `CONFUSION_MATRIX_PROVENANCE.md`
- `CORRECTED_METRICS.md`

Every discrepancy has been mathematically explained and verified by physical execution of `UNet-Mamba/src/run_all_audits.py`.
