# CAVE-Flood Stage 7: Read-Only Release Audit Report

**Document**: `results/final/STAGE7_RELEASE_AUDIT.md`  
**Date**: October 9, 2026  
**Auditor**: Senior Computer Vision Research Engineer & Independent PyTorch Experiment Auditor  
**Project**: CAVE-Flood v1 (UNetMamba + Confidence-Aware Semantic Residual Refinement)  
**Status**: COMPLETE — ALL AUDIT CHECKS PASSED. SYSTEM LOCKED FOR HELD-OUT TEST EVALUATION.

---

## 1. Executive Summary & Release Decision

This release audit independently inspects the workspace, model checkpoints, training histories, evaluation engine, and XAI diagnostics before authorizing the single final evaluation pass over the held-out test partition (**Bangladesh, Cambodia, Nepal**).

### Verdict: **`RELEASE APPROVED — READY FOR FINAL TEST EVALUATION`**
- **Zero Test Leakage**: The held-out test partition has not been evaluated, accessed, or used in any parameter tuning or checkpoint selection.
- **Base Model Preservation**: The baseline checkpoint `best_model.pth` has an identical SHA256 checksum to its initial pre-extension audit (`58674C84E97E16C7AB86F2E218BA8738C7AAF69A8C6FD8C384DE00BEEC9EB76D`).
- **Checkpoints Usable**: Both `best_e1_model.pth` and `best_e2_model.pth` load with zero missing/unexpected keys, contain strictly finite weights, and correspond to the best validation epochs.

---

## 2. Checkpoint Inventory & Integrity (Section A)

| Model Name | Relative File Path | Size (Bytes) | SHA-256 Checksum | Total Params | Trainable Params | Status |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **UNetMamba Baseline (E0)** | `UNet-Mamba/checkpoints/best_model.pth` | 171,815,178 | `58674C84E97E16C7AB86F2E218BA8738C7AAF69A8C6FD8C384DE00BEEC9EB76D` | 14,294,018 | 0 (Frozen) | **PASS** |
| **CASR Neutral Adapter (E1)** | `UNet-Mamba/checkpoints/best_e1_model.pth` | 54,430 | `84A02D1F1FE8371BF799EDB0DDCE28F69D20DD90B3DA503C06EAC87F521226E9` | 12,035 | 12,035 (100%) | **PASS** |
| **CASR RemoteCLIP Adapter (E2)** | `UNet-Mamba/checkpoints/best_e2_model.pth` | 54,430 | `19347F5F64385C46F16C27B345A7F143E56B51F23C1FE7E887906B6E7C19D2D9` | 12,035 | 12,035 (100%) | **PASS** |
| **E1 Final Epoch 5** | `UNet-Mamba/checkpoints/final_e1_epoch_5.pth` | 54,426 | `50101F9B3E2893AF9D0CBEC7CFEE81948C3FE7E95D93450DD62FC18C88A0FDA2` | 12,035 | 12,035 | **PASS** |
| **E2 Final Epoch 5** | `UNet-Mamba/checkpoints/final_e2_epoch_5.pth` | 54,426 | `1B2AD9F752AAA4B58D23771CF1879B10C6AC840AC3091E5404AC79B1DA7E3417` | 12,035 | 12,035 | **PASS** |

- **Loading Verification**:
  - `best_model.pth` loads into `UNetMamba` with 0 missing and 0 unexpected keys.
  - `best_e1_model.pth` and `best_e2_model.pth` load into `CASRAdapter` with 0 missing and 0 unexpected keys (all 16 weight and bias tensors match exactly).
  - All parameter tensors across all 3 checkpoints contain 100% finite floating-point values (0 NaNs, 0 Infs).

---

## 3. Training-History Verification (Section B)

Both experiments completed exactly 5 epochs under fixed random seed `2026`:

### E1 (Neutral Guidance $g=0$):
- Epoch 1: Loss = 1.2273 | Mean Event Flood IoU = 0.6516 | Bolivia = 0.5661 | US-Dakota = 0.7370 | Pooled IoU = 0.5973 [* BEST]
- Epoch 2: Loss = 1.1128 | Mean Event Flood IoU = 0.6474 | Bolivia = 0.5571 | US-Dakota = 0.7378 | Pooled IoU = 0.5901
- Epoch 3: Loss = 1.0614 | Mean Event Flood IoU = 0.6518 | Bolivia = 0.5666 | US-Dakota = 0.7370 | Pooled IoU = 0.5976 [* BEST]
- Epoch 4: Loss = 1.0079 | Mean Event Flood IoU = 0.6493 | Bolivia = 0.5616 | US-Dakota = 0.7370 | Pooled IoU = 0.5936
- Epoch 5: Loss = 0.9959 | Mean Event Flood IoU = **0.6533** | Bolivia = **0.5705** | US-Dakota = 0.7361 | Pooled IoU = **0.6007** [* BEST CHECKPOINT SELECTED]

### E2 (RemoteCLIP Guidance $g$):
- Epoch 1: Loss = 1.2198 | Mean Event Flood IoU = 0.6512 | Bolivia = 0.5737 | US-Dakota = 0.7286 | Pooled IoU = 0.6016 [* BEST]
- Epoch 2: Loss = 1.0950 | Mean Event Flood IoU = 0.6480 | Bolivia = 0.5585 | US-Dakota = 0.7375 | Pooled IoU = 0.5912
- Epoch 3: Loss = 1.0393 | Mean Event Flood IoU = 0.6521 | Bolivia = 0.5671 | US-Dakota = 0.7372 | Pooled IoU = 0.5981 [* BEST]
- Epoch 4: Loss = 1.0123 | Mean Event Flood IoU = 0.6521 | Bolivia = 0.5666 | US-Dakota = 0.7376 | Pooled IoU = 0.5978
- Epoch 5: Loss = 0.9773 | Mean Event Flood IoU = **0.6531** | Bolivia = **0.5695** | US-Dakota = 0.7368 | Pooled IoU = **0.5999** [* BEST CHECKPOINT SELECTED]

- **Checkpoint Selection Rule**: Verified. Both experiments selected Epoch 5 because it achieved the highest Mean Event Validation Flood IoU (`0.6533` for E1; `0.6531` for E2).
- **Stability**: Zero gradient anomalies or non-finite losses occurred across the 5 epochs.

---

## 4. Experimental Comparability & Fairness (Section C)

Strict single-variable control was audited:
1. **Splits**: Training (14 events), Validation (Bolivia, US-Dakota), Test (Bangladesh, Cambodia, Nepal). Zero partition overlap.
2. **Candidate Region Selection**: Both use identical geometry (16 $128 \times 128$ grid tiles), valid pixel coverage filter ($\ge 50\%$), and ranking formula ($0.5 \cdot \text{rank}(H) + 0.5 \cdot \text{rank}(\|\nabla P\|)$), selecting the top 4 candidate tiles per patch.
3. **Architecture**: Both use the identical 9-channel `CASRAdapter` (two 3x3 conv blocks with 32 channels, 1-channel sigmoid gate, 2-logit residual head).
4. **Input Tensors**:
   - Channels 0–3: 4-band PlanetScope imagery (Red, Green, Blue, NIR)
   - Channel 4: Baseline flood probability
   - Channel 5: Predictive entropy $H$
   - Channel 6: Boundary gradient magnitude $\|\nabla P\|$
   - Channel 7: Candidate mask $C \in \{0, 1\}$
   - Channel 8: Guidance channel. In E1, this is strictly zero ($g = 0.0$); in E2, this contains RemoteCLIP score margin $g$.
   - **Result**: Channels 0–7 are byte-for-byte identical between E1 and E2. Only Channel 8 differs.
5. **Loss Function**: Identical Cross-Entropy + Flood Dice loss evaluated strictly inside candidate regions on valid pixels.
6. **Optimizer**: Identical AdamW (`lr = 1e-3`, `weight_decay = 1e-4`, `batch_size = 2`, `seed = 2026`).

---

## 5. Frozen-Model & Zero-Residual Invariant (Section D)

- **Backbone Freezing**: UNet-Mamba's 14,294,018 parameters had `requires_grad = False` throughout training and inference.
- **VLM Freezing**: RemoteCLIP ViT-B/32 had `requires_grad = False` throughout.
- **Trainable Parameters**: Strictly 12,035 parameters in the CASR adapter (0.084% of total parameter footprint).
- **Zero-Residual Invariant**:
  $$\max |Z_{\text{final}} - Z_{\text{base}}| = \mathbf{0.0000000000} \quad (\text{torch.equal == True})$$
  At initialization, the residual head is zero-initialized, ensuring exact baseline identity.
- **Bypass Mode**: When `bypass_casr = True`, output reproduces baseline logits to exact floating-point identity ($\max \Delta = 0.0$).

---

## 6. Evaluation Implementation & Reconstruction (Section E)

- **Quadrant Reconstruction**: Source chips of $1024 \times 1024$ are reconstructed from four non-overlapping quadrants:
  - Quadrant 0: $(0, 0)$
  - Quadrant 1: $(0, 512)$
  - Quadrant 2: $(512, 0)$
  - Quadrant 3: $(512, 512)$
- **Valid Pixel Criterion**:
  $$\text{valid} = (\text{raw\_image} \neq 0).\text{any}(\text{axis}=0) \land (\text{raw\_mask} \neq -1)$$
  Sensor NoData (all bands zero) and ignored annotations ($-1$) are completely excluded.
- **Pixel Count Invariant**:
  - Bolivia (100 chips): 102,675,748 valid pixels
  - US-Dakota (14 chips): 14,680,064 valid pixels
  - Total Valid Validation Pixels: **117,355,812**
  - Sum of confusion matrix counts ($TP + FP + FN + TN$) equals **117,355,812** identically. Every valid pixel contributes exactly once.

---

## 7. Metric Arithmetic & Identity Verification (Section F)

### Mathematical Identity Check:
The relation between Flood Dice and Flood IoU:
$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$
For E0 Baseline:
$$\frac{2 \cdot 0.5926787086517884}{1 + 0.5926787086517884} = 0.7442539483101325 \quad (\equiv \text{Reported Dice: } 0.7442539483101325)$$
Verified to the 16th decimal place.

### Reported Differences (in Percentage Points):
- **E1 Mean Event Flood IoU**: `0.6533` vs E0 `0.6490` $\rightarrow$ **+0.43 percentage points (pp)**.
- **E2 Mean Event Flood IoU**: `0.6531` vs E0 `0.6490` $\rightarrow$ **+0.41 percentage points (pp)**.
- **E1 Pooled Flood IoU**: `0.6007` vs E0 `0.5927` $\rightarrow$ **+0.80 percentage points (pp)**.
- **E2 Pooled Flood IoU**: `0.5999` vs E0 `0.5927` $\rightarrow$ **+0.72 percentage points (pp)**.
- **Bolivia E1 IoU Gain**: `0.5705` vs E0 `0.5603` $\rightarrow$ **+1.02 percentage points (pp)**.
- **Bolivia E2 IoU Gain**: `0.5695` vs E0 `0.5603` $\rightarrow$ **+0.92 percentage points (pp)**.

---

## 8. XAI Artifacts & Scientific Framing (Section G)

- **Figures Verified**:
  - `artifacts/CAVE_Flood_Antigravity/figures/xai_diagnostic_Bolivia_BOL_1176.png` (4.89 MB, $3537 \times 2911$)
  - `artifacts/CAVE_Flood_Antigravity/figures/error_comparison_Bolivia_BOL_1176.png` (0.49 MB, $2830 \times 1260$)
  - `artifacts/CAVE_Flood_Antigravity/figures/xai_diagnostic_US-Dakota_RRN_10_46.png` (5.49 MB, $3537 \times 2911$)
  - `artifacts/CAVE_Flood_Antigravity/figures/error_comparison_US-Dakota_RRN_10_46.png` (0.87 MB, $2830 \times 1260$)
  - `artifacts/CAVE_Flood_Antigravity/figures/xai_diagnostic_Bolivia_BOL_1123.png` (4.68 MB, $3537 \times 2911$)
  - `artifacts/CAVE_Flood_Antigravity/figures/error_comparison_Bolivia_BOL_1123.png` (0.50 MB, $2830 \times 1260$)
- **Scientific Disclosures & Interpretations**:
  - *Entropy*: Reflects local Shannon predictive uncertainty across model logits, not a physical hydrological state.
  - *RemoteCLIP Guidance*: Reflects text-image prompt cosine margin, not an absolute calibrated water probability.
  - *LayerCAM*: Reflects backpropagated gradient attribution through `casr.block2`, demonstrating that the adapter activates selectively on flood boundaries without implying causal reasoning.

---

## 9. Final Release Status

All 8 sections of the Stage 7 audit have passed with zero blocking issues. The model architecture, loss functions, score scalers, and checkpoints are **strictly locked**.

**The pipeline is approved to proceed to Stage 8 (Final Held-Out Test Evaluation).**
