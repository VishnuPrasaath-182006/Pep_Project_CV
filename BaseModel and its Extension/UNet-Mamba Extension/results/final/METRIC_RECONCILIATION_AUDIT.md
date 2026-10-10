# CAVE-Flood v1: Metric Reconciliation & Root Cause Audit Report

**Document**: `results/final/METRIC_RECONCILIATION_AUDIT.md`  
**Date**: October 9, 2026  
**Auditor**: Independent Senior Computer-Vision Research Engineer, PyTorch Evaluation Specialist & Scientific Reproducibility Auditor  
**Project**: CAVE-Flood v1 — UNet-Mamba + CASR + RemoteCLIP  
**Audit Scope**: Investigation of numerical discrepancies between the final experimental report, evaluation code, saved metrics artifacts, and published confusion matrices.

---

## 1. Executive Summary & Audit Verdict

### Verdict: **`VALID WITH LIMITATIONS (REPORT CORRECTIONS IMPLEMENTED)`**

An exhaustive, non-destructive investigation was conducted across the CAVE-Flood v1 codebase, evaluation pipeline, raw prediction logs, and serialized evaluation artifacts (`results/final/final_test_summary.csv`, `final_test_event_metrics.csv`, and `final_test_metrics.json`).

### Key Audit Findings:
1. **The Model Checkpoints and Evaluator Outputs are 100% Intact and Authoritative**:
   The single-pass test evaluation executed by `UNet-Mamba/src/evaluate_final_test.py` across all **44 held-out test scenes** (**46,137,344 unique valid test pixels**) generated exact, mathematically flawless confusion matrices and metrics in `results/final/final_test_summary.csv` and `final_test_metrics.json`.
2. **The Discrepancy Stemmed From a Report Transcription / Hallucination Bug**:
   When the markdown report (`CAVE_Flood_Final_Results_Report.md`) was composed, the authoring agent copied the correct IoU, Dice, Precision, and Recall figures from the log summary, but did **not** transcribe the true confusion counts from `final_test_summary.csv`. Instead, the authoring agent synthesized an erroneous confusion matrix (`TP = 11,211,885`, `FP = 1,033,485`, `FN = 2,204,591`, `TN = 31,687,383`) that roughly halved the true flood positive count and inflated the background negative count to match the 46,137,344 pixel total.
3. **Mathematical Inconsistencies Traced to the Root Cause**:
   - The reported accuracy (`0.965421`) in the text was hallucinated. Calculating accuracy from the report's published synthetic confusion matrix yields $\approx 0.929817$. In reality, the true evaluator accuracy on the held-out test set is **`0.861027`** (86.10%), which matches the authoritative confusion matrix $(TP = 22,201,266, TN = 17,524,242)$.
   - The reported Macro mIoU (`0.754020`) in the text was the true evaluator metric computed from the real confusion matrix ($0.775913$ flood IoU + $0.732127$ background IoU). However, recalculating Macro mIoU from the published synthetic confusion matrix gives $\approx 0.841599$, exposing the internal contradiction between the text metrics and the published matrix.
   - Per-event specificity and accuracy columns in Table 7.2 of the report were also transcribed with synthetic values rather than the evaluator's output.
   - An isolated typographic error in `CAVE_Flood_Final_Report.md` line 121 displayed Bangladesh E2 IoU as `0.7554` instead of `0.5754`.

All original checkpoints, logs, and artifacts have been preserved without modification. A versioned, corrected report (`CAVE_Flood_Final_Results_Report_v2.md`) and versioned bundle (`CAVE_Flood_Final_Results_v2.zip`) have been created.

---

## 2. Root Cause Analysis (Task 3)

### 2.1 File Paths and Line Numbers of Anomalies

| File Path | Line Range | Nature of Error | Underlying Cause |
| :--- | :---: | :--- | :--- |
| `CAVE_Flood_Final_Results_Report.md` | Lines 144–147 | Erroneous Confusion Matrix (`TP=11,211,885`, `TN=31,687,383`) | LLM transcription bug: authoring agent generated synthetic counts rather than reading `final_test_summary.csv`. |
| `CAVE_Flood_Final_Results_Report.md` | Line 143 | Erroneous Pooled Pixel Accuracy (`0.965421`) | Hallucinated accuracy value, conflicting with both the published matrix ($0.929817$) and the true evaluator output ($0.861027$). |
| `CAVE_Flood_Final_Results_Report.md` | Lines 153–164 | Per-event Specificity and Accuracy mismatch | Erroneous values transcribed in table 7.2 (e.g. Bangladesh specificity listed as 0.9497 instead of true 0.9069). |
| `CAVE_Flood_Final_Report.md` | Line 121 | Typo in Bangladesh E2 Flood IoU (`0.7554`) | Transposition typo in markdown: `0.7554` typed instead of `0.5754`. |
| `build_html_report.py` | Line 38–58 | HTML report inherits markdown errors | Generated HTML faithfully mirrored the erroneous markdown text. |

### 2.2 Proof of Hallucinated Confusion Matrix vs True Evaluator Counts

The evaluator script `UNet-Mamba/src/evaluate_final_test.py` processes each scene quadrant, reconstructs the full $1024 \times 1024$ chip, identifies valid pixels via `chip_valid = (recon_img != 0).any(axis=0) & (recon_mask != -1)`, and accumulates counts in `event_counts` and `pooled_counts` (lines 243–260).

When `evaluate_final_test.py` completed at `2026-10-09 17:51:36`, it recorded the true counts into:
- `results/final/final_test_summary.csv`
- `results/final/final_test_event_metrics.csv`
- `results/final/final_test_metrics.json`

However, in `final_test_evaluation.log` (lines 27–30), the terminal summary table printed only:
```
experiment  mean_event_flood_iou  std_event_flood_iou  pooled_flood_iou  pooled_flood_dice  pooled_macro_miou  pooled_precision  pooled_recall
        E0              0.752949             0.131469          0.775913           0.873818           0.754020          0.915601       0.835683
        E1              0.754259             0.129421          0.777762           0.874990           0.753758          0.908188       0.844133
        E2              0.754468             0.129101          0.777723           0.874965           0.753955          0.909053       0.843342
```
Because the terminal stdout did not print the raw TP, FP, FN, TN columns, an authoring agent attempting to produce an expanded markdown table without reading the CSV files reverse-engineered a synthetic confusion matrix:
- It set $\text{TP} \approx 11.2\text{M}$ (instead of true $22.2\text{M}$)
- It set $\text{FP} \approx 1.03\text{M}$ (instead of true $2.05\text{M}$)
- It set $\text{FN} \approx 2.20\text{M}$ (instead of true $4.37\text{M}$)
- It set $\text{TN} = 46,137,344 - (\text{TP} + \text{FP} + \text{FN}) = 31,687,383$ (instead of true $17,524,242$).

This synthetic matrix preserved Flood IoU $\approx 0.7759$, Flood Precision $\approx 0.9156$, Flood Recall $\approx 0.8357$, and Flood Dice $\approx 0.8738$, but it completely distorted Background IoU, Accuracy, and the physical ratio of flood pixels to non-flood pixels.

---

## 3. Side-by-Side Mathematical Reconciliation (Task 1)

All metrics are calculated using standard zero-denominator protected confusion matrix formulas:
$$\text{Flood IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
$$\text{Background IoU} = \frac{\text{TN}}{\text{TN} + \text{FP} + \text{FN}}$$
$$\text{Binary Macro mIoU} = \frac{\text{Flood IoU} + \text{Background IoU}}{2}$$
$$\text{Flood Dice} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
$$\text{Pixel Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{FN} + \text{TN}}$$

### 3.1 E0 Baseline (UNet-Mamba) Reconciliation Table

| Metric / Count | Published Report Text | Published CM Derived | Authoritative Ground Truth (`final_test_summary.csv`) | Status & Reconciliation Notes |
| :--- | :---: | :---: | :---: | :--- |
| **True Positives (TP)** | 11,211,885 | 11,211,885 | **22,201,266** | Corrected to authoritative evaluator count (+10,989,381 px). |
| **False Positives (FP)** | 1,033,485 | 1,033,485 | **2,046,482** | Corrected to authoritative evaluator count (+1,012,997 px). |
| **False Negatives (FN)** | 2,204,591 | 2,204,591 | **4,365,354** | Corrected to authoritative evaluator count (+2,160,763 px). |
| **True Negatives (TN)** | 31,687,383 | 31,687,383 | **17,524,242** | Corrected to authoritative evaluator count (-14,163,141 px). |
| **Total Valid Pixels** | 46,137,344 | 46,137,344 | **46,137,344** | Verified: Sum of TP+FP+FN+TN is invariant (46,137,344). |
| **Flood IoU** | 0.775913 | 0.775911 | **0.775913** | Evaluator matches to 6 decimal places ($22,201,266 / 28,613,102$). |
| **Background IoU** | Not reported | 0.907286 | **0.732127** | Evaluator matches ($17,524,242 / 23,936,078$). |
| **Binary Macro mIoU** | 0.754020 | **0.841599** | **0.754020** | **Discrepancy Resolved**: Report value (0.754020) is correct from evaluator; published CM derived (0.841599) was artifact of synthetic CM. |
| **Flood Dice (F1)** | 0.873818 | 0.873817 | **0.873818** | Evaluator matches ($44,402,532 / 50,814,368$). |
| **Precision** | 0.915601 | 0.915602 | **0.915601** | Evaluator matches ($22,201,266 / 24,247,748$). |
| **Recall** | 0.835683 | 0.835680 | **0.835683** | Evaluator matches ($22,201,266 / 26,566,620$). |
| **Specificity** | Not reported | 0.968415 | **0.895431** | Evaluator matches ($17,524,242 / 19,570,724$). |
| **Pixel Accuracy** | 0.965421 | **0.929817** | **0.861027** | **Discrepancy Resolved**: True accuracy is 0.861027 ($39,725,508 / 46,137,344$). 0.965421 was hallucinated; 0.929817 was from synthetic CM. |

---

### 3.2 E1 CASR (Neutral Guidance $g=0$) Reconciliation Table

| Metric / Count | Published Report Text | Published CM Derived | Authoritative Ground Truth (`final_test_summary.csv`) | Status & Reconciliation Notes |
| :--- | :---: | :---: | :---: | :--- |
| **True Positives (TP)** | 11,325,258 | 11,325,258 | **22,425,765** | Corrected to authoritative evaluator count. |
| **False Positives (FP)** | 1,144,792 | 1,144,792 | **2,267,106** | Corrected to authoritative evaluator count. |
| **False Negatives (FN)** | 2,091,218 | 2,091,218 | **4,140,855** | Corrected to authoritative evaluator count. |
| **True Negatives (TN)** | 31,576,076 | 31,576,076 | **17,303,618** | Corrected to authoritative evaluator count. |
| **Total Valid Pixels** | 46,137,344 | 46,137,344 | **46,137,344** | Verified: Sum of TP+FP+FN+TN = 46,137,344. |
| **Flood IoU** | 0.777762 | 0.777766 | **0.777762** | Evaluator matches ($22,425,765 / 28,833,726$). |
| **Background IoU** | Not reported | 0.907043 | **0.729754** | Evaluator matches ($17,303,618 / 23,711,579$). |
| **Binary Macro mIoU** | 0.753758 | **0.842405** | **0.753758** | **Discrepancy Resolved**: True Macro mIoU is 0.753758. |
| **Flood Dice (F1)** | 0.874990 | 0.874992 | **0.874990** | Evaluator matches ($44,851,530 / 51,259,491$). |
| **Precision** | 0.908188 | 0.908197 | **0.908188** | Evaluator matches ($22,425,765 / 24,692,871$). |
| **Recall** | 0.844133 | 0.844131 | **0.844133** | Evaluator matches ($22,425,765 / 26,566,620$). |
| **Specificity** | Not reported | 0.965013 | **0.884158** | Evaluator matches ($17,303,618 / 19,570,724$). |
| **Pixel Accuracy** | 0.965128 | **0.929861** | **0.861111** | **Discrepancy Resolved**: True accuracy is 0.861111 ($39,729,383 / 46,137,344$). |

---

### 3.3 E2 CASR (RemoteCLIP Guidance $g$) Reconciliation Table

| Metric / Count | Published Report Text | Published CM Derived | Authoritative Ground Truth (`final_test_summary.csv`) | Status & Reconciliation Notes |
| :--- | :---: | :---: | :---: | :--- |
| **True Positives (TP)** | 11,314,639 | 11,314,639 | **22,404,736** | Corrected to authoritative evaluator count. |
| **False Positives (FP)** | 1,131,850 | 1,131,850 | **2,241,494** | Corrected to authoritative evaluator count. |
| **False Negatives (FN)** | 2,101,837 | 2,101,837 | **4,161,884** | Corrected to authoritative evaluator count. |
| **True Negatives (TN)** | 31,589,018 | 31,589,018 | **17,329,230** | Corrected to authoritative evaluator count. |
| **Total Valid Pixels** | 46,137,344 | 46,137,344 | **46,137,344** | Verified: Sum of TP+FP+FN+TN = 46,137,344. |
| **Flood IoU** | 0.777723 | 0.777728 | **0.777723** | Evaluator matches ($22,404,736 / 28,808,114$). |
| **Background IoU** | Not reported | 0.907139 | **0.730187** | Evaluator matches ($17,329,230 / 23,732,608$). |
| **Binary Macro mIoU** | 0.753955 | **0.842433** | **0.753955** | **Discrepancy Resolved**: True Macro mIoU is 0.753955. |
| **Flood Dice (F1)** | 0.874965 | 0.874968 | **0.874965** | Evaluator matches ($44,809,472 / 51,212,850$). |
| **Precision** | 0.909053 | 0.909063 | **0.909053** | Evaluator matches ($22,404,736 / 24,646,230$). |
| **Recall** | 0.843342 | 0.843339 | **0.843342** | Evaluator matches ($22,404,736 / 26,566,620$). |
| **Specificity** | Not reported | 0.965409 | **0.885467** | Evaluator matches ($17,329,230 / 19,570,724$). |
| **Pixel Accuracy** | 0.965215 | **0.929912** | **0.861211** | **Discrepancy Resolved**: True accuracy is 0.861211 ($39,733,966 / 46,137,344$). |

---

## 4. Per-Event Verification & Aggregation (Task 2)

### 4.1 Complete Event Confusion Counts (Authoritative Ground Truth)

Every valid source chip pixel is counted exactly once. There is zero overlap, zero sensor NoData inclusion, and zero padding contamination.

| Experiment | Event | Scenes | Valid Pixels | TP | FP | FN | TN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0** | Bangladesh | 15 | 15,728,640 | 3,348,498 | 1,012,498 | 1,502,129 | 9,865,515 |
| **E0** | Cambodia | 20 | 20,971,520 | 13,381,935 | 649,086 | 2,484,005 | 4,456,494 |
| **E0** | Nepal | 9 | 9,437,184 | 5,470,833 | 384,898 | 379,220 | 3,202,233 |
| **E0 Sum** | **Held-Out Pooled** | **44** | **46,137,344** | **22,201,266** | **2,046,482** | **4,365,354** | **17,524,242** |
| | | | | | | | |
| **E1** | Bangladesh | 15 | 15,728,640 | 3,414,231 | 1,091,570 | 1,436,396 | 9,786,443 |
| **E1** | Cambodia | 20 | 20,971,520 | 13,515,107 | 738,678 | 2,350,833 | 4,366,902 |
| **E1** | Nepal | 9 | 9,437,184 | 5,496,427 | 436,858 | 353,626 | 3,150,273 |
| **E1 Sum** | **Held-Out Pooled** | **44** | **46,137,344** | **22,425,765** | **2,267,106** | **4,140,855** | **17,303,618** |
| | | | | | | | |
| **E2** | Bangladesh | 15 | 15,728,640 | 3,412,666 | 1,080,710 | 1,437,961 | 9,797,303 |
| **E2** | Cambodia | 20 | 20,971,520 | 13,497,394 | 729,076 | 2,368,546 | 4,376,504 |
| **E2** | Nepal | 9 | 9,437,184 | 5,494,676 | 431,708 | 355,377 | 3,155,423 |
| **E2 Sum** | **Held-Out Pooled** | **44** | **46,137,344** | **22,404,736** | **2,241,494** | **4,161,884** | **17,329,230** |

### 4.2 Mathematical Invariants Verified
1. **Pooled CM Equals Sum of Event CM**:
   - For E0: $\sum \text{TP} = 3,348,498 + 13,381,935 + 5,470,833 = 22,201,266$ (Exact).
   - For E1: $\sum \text{TP} = 3,414,231 + 13,515,107 + 5,496,427 = 22,425,765$ (Exact).
   - For E2: $\sum \text{TP} = 3,412,666 + 13,497,394 + 5,494,676 = 22,404,736$ (Exact).
   - All FP, FN, TN sums match their pooled counterparts to the exact integer.
2. **Identical Ground-Truth Mask**:
   - Total flood ground-truth pixels ($\text{TP} + \text{FN}$) across all 44 scenes:
     - E0: $22,201,266 + 4,365,354 = 26,566,620$
     - E1: $22,425,765 + 4,140,855 = 26,566,620$
     - E2: $22,404,736 + 4,161,884 = 26,566,620$
     - **Verified**: Identical to the exact pixel across all 3 models.
   - Total non-flood ground-truth pixels ($\text{TN} + \text{FP}$):
     - E0: $17,524,242 + 2,046,482 = 19,570,724$
     - E1: $17,303,618 + 2,267,106 = 19,570,724$
     - E2: $17,329,230 + 2,241,494 = 19,570,724$
     - **Verified**: Identical to the exact pixel across all 3 models.
3. **Macro Event IoU vs Pooled IoU**:
   - Mean Event Flood IoU:
     - E0: $\frac{0.571111 + 0.810288 + 0.877446}{3} = 0.752949$
     - E1: $\frac{0.574574 + 0.813937 + 0.874265}{3} = 0.754259$
     - E2: $\frac{0.575362 + 0.813340 + 0.874703}{3} = 0.754468$
   - Pooled Flood IoU is weighted by valid pixels and flood extent (0.775913 for E0, 0.777762 for E1, 0.777723 for E2). The report clearly distinguishes Mean Event IoU (macro across events) from Pixel-Pooled IoU.
4. **Validation Dataset Aggregation**:
   - Bolivia (100 chips): 102,675,748 valid pixels. E0 TP: 25,485,263, FP: 7,771,044, FN: 12,230,549, TN: 57,188,892.
   - US-Dakota (14 chips): 14,680,064 valid pixels. E0 TP: 7,498,662, FP: 1,780,471, FN: 886,297, TN: 4,514,634.
   - Total valid validation pixels: $102,675,748 + 14,680,064 = \mathbf{117,355,812}$.
   - Sum of E0 validation counts: $32,983,925 + 9,551,515 + 13,116,846 + 61,703,526 = 117,355,812$ (Exact).
5. **Zero Scene Leakage Across Partitions**:
   - Train partition: 14 distinct events (Colombia, Nigeria, US-Carolina, etc.).
   - Validation partition: Bolivia (100 chips), US-Dakota (14 chips).
   - Test partition: Bangladesh (15 chips), Cambodia (20 chips), Nepal (9 chips).
   - Pairwise intersection between partitions is strictly $\emptyset$.

---

## 5. Correct Scientific Interpretation (Task 5)

1. **E1 vs E0 Baseline Comparison**:
   - **Validation**: Mean Event Flood IoU improves by **+0.43 percentage points (pp)** (0.6490 $\rightarrow$ 0.6533). Pixel-pooled recall improves by **+1.78 pp** (0.7155 $\rightarrow$ 0.7333), while precision drops by **-0.68 pp** (0.7754 $\rightarrow$ 0.7686).
   - **Held-Out Test**: Mean Event Flood IoU improves by **+0.13 pp** (0.7529 $\rightarrow$ 0.7543). Pooled Flood IoU improves by **+0.18 pp** (0.7759 $\rightarrow$ 0.7778). Recall improves by **+0.85 pp** (0.8357 $\rightarrow$ 0.8441), while precision drops by **-0.74 pp** (0.9156 $\rightarrow$ 0.9082).
   - **Physical Mechanism**: CASR primarily operates as an edge false-negative recoverer, converting ambiguous border pixels into flood. On the test set, E1 recovers **+224,499 flood pixels** while introducing **+220,624 false positive pixels**; because the true flood set is larger than the background margin, net IoU increases.
2. **E2 vs E0 Comparison**:
   - **Validation**: Mean Event Flood IoU improves by **+0.41 pp** (0.6490 $\rightarrow$ 0.6531).
   - **Held-Out Test**: Mean Event Flood IoU improves by **+0.15 pp** (0.7529 $\rightarrow$ 0.7545). Pooled Flood IoU improves by **+0.18 pp** (0.7759 $\rightarrow$ 0.7777). Recall improves by **+0.77 pp** (0.8357 $\rightarrow$ 0.8433), while precision drops by **-0.65 pp** (0.9156 $\rightarrow$ 0.9091).
3. **E2 vs E1 Additional Guidance Effect**:
   - On Validation, E1 leads E2 by **+0.02 pp** (0.6533 vs 0.6531).
   - On Held-Out Test, E2 leads E1 by **+0.02 pp** in Mean Event Flood IoU (0.7545 vs 0.7543) and leads by **+0.08 pp** in Bangladesh (0.5754 vs 0.5746). However, in Pixel-Pooled Flood IoU, E1 leads E2 by **+0.004 pp** (0.777762 vs 0.777723).
   - **Scientific Qualification**: The marginal gain of E2 over E1 is small ($\approx +0.02$ pp). **RemoteCLIP guidance does NOT definitively prove improved generalizability or proven semantic boundary disambiguation.** The evidence supports only that vision-language score margins can provide a weak prior in highly turbid settings (Bangladesh), but the dominant driver of performance improvement is the spatial CASR adapter architecture itself.
4. **Event-Level Heterogeneity & Degradation**:
   - In **Nepal**, baseline UNet-Mamba achieves 0.8774 Flood IoU. Both CASR variants experience a slight drop on Nepal: E1 drops by **-0.32 pp** (0.8743) and E2 drops by **-0.27 pp** (0.8747), driven by slight over-segmentation along clean mountain water margins.
   - In contrast, in **Bangladesh** (challenging turbid river delta), CASR yields consistent gains: E1 improves by **+0.35 pp** (0.5746) and E2 improves by **+0.43 pp** (0.5754).
5. **No Checkpoint Retuning**:
   - Checkpoints `best_e1_model.pth` and `best_e2_model.pth` were strictly selected at Epoch 5 based on **Validation Set Mean Event Flood IoU**, prior to held-out test evaluation. No checkpoints were selected or modified using test performance.

---

## 6. Action Plan & Deliverables Created (Task 4 & 7)

1. **Preservation**: Original files `CAVE_Flood_Final_Results_Report.md`, `CAVE_Flood_Final_Results_Report.html`, `CAVE_Flood_Final_Results.zip`, `final_test_summary.csv`, and all model checkpoints are preserved untouched.
2. **Versioned Report Created**: `CAVE_Flood_Final_Results_Report_v2.md` and `results/final/CAVE_Flood_Final_Results_Report_v2.md` incorporating the authoritative confusion matrices, true accuracy, true per-event metrics, fixed typos, and corrected scientific framing.
3. **Versioned Metrics Files Created**:
   - `results/final/final_test_metrics_corrected_v2.json`
   - `results/final/final_test_summary_corrected_v2.csv`
   - `results/final/final_test_event_metrics_corrected_v2.csv`
4. **HTML Report Generated**: `CAVE_Flood_Final_Results_Report_v2.html` via `build_html_report_v2.py`.
5. **Reproducibility Guide**: `REPRODUCIBILITY_INSTRUCTIONS_v2.md`.
6. **SHA256 Manifest**: `SHA256SUMS_v2.txt`.
7. **Versioned Release Archive**: `CAVE_Flood_Final_Results_v2.zip` containing all verified and corrected deliverables.
