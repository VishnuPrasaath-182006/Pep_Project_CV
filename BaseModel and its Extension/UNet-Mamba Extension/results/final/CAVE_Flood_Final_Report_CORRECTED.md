# CAVE-Flood v1: Complete Experimental Results & Final Verification Report (Corrected Edition)

**Project Title**: CAVE-Flood: Confidence-Aware Vision-Language-Guided Residual Refinement for Flood Inundation Segmentation  
**Base Architecture**: Authentic UNet-Mamba (ResT hierarchical vision transformer encoder + chunked checkpointed SS2D Mamba decoder + LSM)  
**Document Purpose**: Final Authoritative Experimental Results & Scientifically Reconciled Technical Report  
**Edition**: Corrected Final Release Edition  
**Date**: October 9, 2026  
**Status**: VERIFIED, MATHEMATICALLY RECONCILED, EVALUATION LOCKED & FULLY REPRODUCIBLE  

---

## 1. Executive Summary

This report presents the complete, verified experimental results for **CAVE-Flood v1**, evaluating the Confidence-Aware Semantic Residual (CASR) adapter on frozen UNet-Mamba backbones for flood inundation segmentation on PlanetScope satellite imagery.

### Core Architectural Invariants:
1. **Zero Backbone Retraining**: The 14.29M-parameter UNet-Mamba baseline checkpoint remained strictly frozen and untouched throughout all stages (`SHA256: 58674C84E97E16C7AB86F2E218BA8738C7AAF69A8C6FD8C384DE00BEEC9EB76D`).
2. **Parameter Footprint**:
   - CASR Adapter Head: **12,035 trainable parameters** ($0.084\%$ of base model; checkpoint size: **54.4 KB**).
   - UNet-Mamba + CASR Head: **14,306,053 parameters** total.
   - RemoteCLIP Vision-Language Guidance Module (E2 only): **87,849,217 frozen parameters** (ViT-B/32 text/image encoder).
   - Full pipeline during E2 inference: **102,155,270 parameters** total (12,035 trainable, 0.012%).
3. **Exact Zero-Residual Equivalence**: In non-candidate regions ($75\%$ of image area) or at initialization ($\lambda = 0$), CASR logits match UNet-Mamba base logits to exact machine precision (`torch.equal == True`, $\max |\Delta Z| \equiv 0.0$).
4. **Validation Performance (114 Chips, 117.36M Unique Pixels across Bolivia & US-Dakota)**:
   - **E1 (Neutral Guidance $g=0$)** achieved **`0.653316`** Mean Event Flood IoU (**+0.43 pp** over baseline `0.648970`).
   - **E2 (RemoteCLIP Guidance $g$)** achieved **`0.653148`** Mean Event Flood IoU (**+0.41 pp** over baseline `0.648970`), winning the tie-breaker Macro mIoU (**`0.667019` vs `0.666783`, +0.50 pp**).
5. **Held-Out Test Performance (44 Chips, 46.14M Unique Pixels across Held-Out Asian Flood Events)**:
   - Evaluated strictly single-pass on quarantined Asian flood events: Bangladesh (15 chips), Cambodia (20 chips), and Nepal (9 chips).
   - **Mean Event Flood IoU**:
     - Baseline E0: `0.752949`
     - Neutral CASR E1: `0.754259` (**+0.13 pp** over E0)
     - RemoteCLIP CASR E2: **`0.754468`** (**+0.15 pp** over E0, **+0.02 pp** over E1)
   - **Pixel-Pooled Flood IoU**:
     - Baseline E0: `0.775913`
     - Neutral CASR E1: **`0.777762`** (**+0.18 pp** over E0)
     - RemoteCLIP CASR E2: `0.777723` (**+0.18 pp** over E0, **-0.004 pp** vs E1)
   - **Pixel Accuracy**: E0 = `0.861027`, E1 = `0.861111` (+0.01 pp), E2 = **`0.861211`** (+0.02 pp).
   - **Recall Recovery**: CASR recovered over **203,000 to 224,000 previously missed flood pixels**, elevating test recall from $83.57\%$ to $84.41\%$ (+0.85 pp for E1, +0.77 pp for E2).

---

## 2. Experimental Rigor & Fair Comparison (E1 vs. E2)

To ensure scientific validity, experiments **E1** and **E2** were executed under identical, controlled conditions:

| Experimental Dimension | Experiment E1 (Neutral Guidance) | Experiment E2 (RemoteCLIP Guidance) | Rigorous Parity Check |
| :--- | :--- | :--- | :---: |
| **Candidate Region Selection** | Top 4 of 16 spatial tiles ($128 \times 128$) ranked by $0.5 \text{rank}(H) + 0.5 \text{rank}(\|\nabla P\|)$ | Identical cached candidate masks | **IDENTICAL** |
| **CASR Adapter Architecture** | 9-channel input, dual Conv2d(32) blocks, sigmoid gate $G$, residual $\Delta Z$ | Identical architecture | **IDENTICAL** |
| **Trainable Parameters** | Exactly 12,035 parameters | Exactly 12,035 parameters | **IDENTICAL** |
| **Training Budget & Epochs** | Fixed 5 epochs, batch size 2, seed 2026 | Fixed 5 epochs, batch size 2, seed 2026 | **IDENTICAL** |
| **Optimization & Loss** | AdamW (`lr=1e-3`, `wd=1e-4`), $\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{Dice}}$ on candidates | Identical AdamW, identical loss | **IDENTICAL** |
| **Dataset Partitions** | 14 training events, 2 validation events (Bolivia, US-Dakota) | Identical partitions and splits | **IDENTICAL** |
| **Non-Guidance Feature Inputs** | 4 multispectral bands, baseline probability, entropy, gradient, candidate mask | Identical 8 non-guidance channels | **IDENTICAL** |
| **Guidance Channel ($g$)** | Uniform zero scalar ($g \equiv 0.0$) | Scaled cosine similarity score from frozen RemoteCLIP | **CONTROLLED ABLATION** |
| **Evaluation Pipeline** | Single-pass source-chip reconstruction | Single-pass source-chip reconstruction | **IDENTICAL** |

---

## 3. Dissecting Contributions: CASR vs. RemoteCLIP Guidance

A central goal of this audit is distinguishing the contribution of the **CASR spatial residual adapter** from the additional contribution of **RemoteCLIP vision-language guidance**:

### 3.1 The CASR Mechanism (E1 vs. E0)
- **Primary Effect**: CASR operates primarily as an **inundation boundary dilator and recall enhancer**.
- By concentrating residual updates strictly within the $25\%$ highest-entropy boundary tiles, CASR recovers missed shallow water and turbid edges, lifting test recall by **+0.85 pp** ($83.57\% \rightarrow 84.41\%$) and adding $+224,499$ true positive pixels.
- **Precision Penalty**: This boundary recovery incurs a consistent precision loss of **-0.74 pp** ($91.56\% \rightarrow 90.82\%$), introducing $+220,624$ false positive pixels on adjacent wet soil and shadows.
- **Net IoU Gain**: Because true positive gains outweigh false alarms, Pixel-Pooled Flood IoU improves by **+0.18 pp** ($0.7759 \rightarrow 0.7778$).

### 3.2 The RemoteCLIP Guidance Effect (E2 vs. E1)
- **Mean Event Flood IoU**: E2 achieves `0.754468` vs E1's `0.754259`. The net improvement attributable to RemoteCLIP guidance is **+0.02 percentage points (+0.00021)**, **NOT +0.15 pp** (which was the gain of E2 over the unrefined baseline E0).
- **Pixel-Pooled Flood IoU**: E2 achieves `0.777723`, which is **0.004 pp lower** than E1 (`0.777762`).
- **Event-Specific Breakdown**:
  - In turbid **Bangladesh**, RemoteCLIP provided a minor boost over neutral guidance (**0.5754 vs 0.5746, +0.08 pp**).
  - In lacustrine **Cambodia**, RemoteCLIP was essentially tied with neutral guidance (**0.8133 vs 0.8139, -0.06 pp**).
  - In mountainous **Nepal**, both models degraded slightly (**0.8747 for E2 vs 0.8743 for E1**, both below E0's 0.8774).
- **Scientific Conclusion**: RemoteCLIP text embeddings provide a subtle, localized prior that slightly assists high-uncertainty river regimes, but **its effect on overall segmentation accuracy is marginal ($\le 0.02\text{ pp}$)**. The primary performance gain of CAVE-Flood stems from spatial residual learning, not vision-language grounding.

---

## 4. Master Reconciled Performance Tables

### 4.1 Held-Out Test Set Summary (44 Chips, 46,137,344 Valid Pixels)

| Metric | E0 Baseline | E1 CASR (Neutral) | E2 CASR (RemoteCLIP) | Test Winner | Δ E1 vs E0 | Δ E2 vs E0 | Δ E2 vs E1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Event Flood IoU** | `0.752949` | `0.754259` | **`0.754468`** | **E2** | **+0.13 pp** | **+0.15 pp** | **+0.02 pp** |
| **Pixel-Pooled Flood IoU** | `0.775913` | **`0.777762`** | `0.777723` | **E1** | **+0.18 pp** | **+0.18 pp** | -0.004 pp |
| **Pixel-Pooled Flood Dice** | `0.873818` | **`0.874990`** | `0.874965` | **E1** | **+0.12 pp** | **+0.11 pp** | -0.002 pp |
| **Pooled Background IoU** | **`0.732127`** | `0.729754` | `0.730187` | **E0** | -0.24 pp | -0.19 pp | +0.04 pp |
| **Pooled Macro mIoU** | **`0.754020`** | `0.753758` | `0.753955` | **E0** | -0.03 pp | -0.01 pp | +0.02 pp |
| **Pooled Flood Precision** | **`0.915601`** | `0.908188` | `0.909053` | **E0** | -0.74 pp | -0.65 pp | +0.09 pp |
| **Pooled Flood Recall** | `0.835683` | **`0.844133`** | `0.843342` | **E1** | **+0.85 pp** | **+0.77 pp** | -0.08 pp |
| **Pooled Specificity** | **`0.895431`** | `0.884158` | `0.885467` | **E0** | -1.13 pp | -1.00 pp | +0.13 pp |
| **Pixel Accuracy** | `0.861027` | `0.861111` | **`0.861211`** | **E2** | **+0.01 pp** | **+0.02 pp** | **+0.01 pp** |
| **True Positives (TP)** | 22,201,266 | **22,425,765** | 22,404,736 | **E1** | +224,499 px | +203,470 px | -21,029 px |
| **False Positives (FP)** | **2,046,482** | 2,267,106 | 2,241,494 | **E0** | +220,624 px | +195,012 px | -25,612 px |
| **False Negatives (FN)** | 4,365,354 | **4,140,855** | 4,161,884 | **E1** | -224,499 px | -203,470 px | +21,029 px |
| **True Negatives (TN)** | **17,524,242** | 17,303,618 | 17,329,230 | **E0** | -220,624 px | -195,012 px | +25,612 px |

### 4.2 Per-Event Breakdown across Held-Out Asian Flood Events

| Event | Chips | Valid Pixels | Model | Flood IoU | Flood Dice | Precision | Recall | Specificity | Pixel Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bangladesh** | 15 | 15,728,640 | **E0** | 0.5711 | 0.7270 | **0.7678** | 0.6903 | **0.9069** | **0.8401** |
| *(Lowland floodplain)* | | | **E1** | 0.5746 (+0.35 pp) | 0.7298 (+0.28 pp) | 0.7577 (-1.01 pp) | **0.7039** (+1.36 pp) | 0.8997 (-0.72 pp) | 0.8393 (-0.08 pp) |
| | | | **E2** | **0.5754** (**+0.43 pp**) | **0.7305** (**+0.35 pp**) | 0.7595 (-0.83 pp) | 0.7036 (+1.33 pp) | 0.9007 (-0.62 pp) | 0.8399 (-0.02 pp) |
| **Cambodia** | 20 | 20,971,520 | **E0** | 0.8103 | 0.8952 | **0.9537** | 0.8434 | **0.8729** | 0.8506 |
| *(Lacustrine basin)* | | | **E1** | **0.8139** (**+0.36 pp**) | **0.8974** (**+0.22 pp**) | 0.9482 (-0.55 pp) | **0.8518** (+0.84 pp) | 0.8553 (-1.76 pp) | **0.8527** (+0.21 pp) |
| | | | **E2** | 0.8133 (+0.30 pp) | 0.8971 (+0.19 pp) | 0.9488 (-0.49 pp) | 0.8507 (+0.73 pp) | 0.8572 (-1.57 pp) | 0.8523 (+0.17 pp) |
| **Nepal** | 9 | 9,437,184 | **E0** | **0.8774** | **0.9347** | **0.9343** | 0.9352 | **0.8927** | **0.9190** |
| *(Mountainous relief)* | | | **E1** | 0.8743 (-0.31 pp) | 0.9329 (-0.18 pp) | 0.9264 (-0.79 pp) | **0.9396** (+0.44 pp) | 0.8782 (-1.45 pp) | 0.9162 (-0.28 pp) |
| | | | **E2** | 0.8747 (-0.27 pp) | 0.9332 (-0.15 pp) | 0.9272 (-0.71 pp) | 0.9393 (+0.41 pp) | 0.8797 (-1.30 pp) | 0.9166 (-0.24 pp) |

---

## 5. Documented Scientific Limitations & Critical Observations

1. **Mountainous Terrain Degradation (Nepal)**:
   In high-relief mountainous topography, mountain shadows and steep valley contours produce high boundary entropy that the candidate selector flags as uncertain. CASR expands these boundaries, erroneously classifying shadows and wet talus as flood water, degrading Flood IoU by **$-0.27\text{ pp}$** (E2) and **$-0.31\text{ pp}$** (E1). Practitioners must not apply CASR indiscriminately in mountainous relief.
2. **Systematic Precision Trade-off**:
   In all evaluated events, CASR reduces precision by $0.49\text{ to }1.01\text{ pp}$. The method is suited for disaster response contexts prioritizing high recall over false alarm avoidance.
3. **Correlational XAI Diagnostics**:
   LayerCAM demonstrates that positive spatial gradients concentrate on water-land boundaries inside candidate regions. This confirms convolutional feature localization, but **does not prove causal semantic grounding**.
4. **No External SOTA Claims**:
   CAVE-Flood is an effective, parameter-efficient residual refinement adapter (12,035 parameters) that improves boundary recall on a frozen backbone, but has not been benchmarked against fine-tuned foundation models (e.g. SegFormer, Prithvi).

---

## 6. Checkpoint Provenance & Manifest Verification

All checkpoints were verified using SHA-256 cryptographic hashes:
- **Baseline UNet-Mamba** (`UNet-Mamba/checkpoints/best_model.pth`):  
  `58674c84e97e16c7ab86f2e218ba8738c7aaf69a8c6fd8c384de00beec9eb76d` (Frozen)
- **CASR E1 Adapter** (`UNet-Mamba/checkpoints/best_e1_model.pth`):  
  `84a02d1f1fe8371bf799edb0ddce28f69d20dd90b3da503c06eac87f521226e9` (Epoch 5)
- **CASR E2 Adapter** (`UNet-Mamba/checkpoints/best_e2_model.pth`):  
  `19347f5f64385c46f16c27b345a7f143e56b51f23c1fe7e887906b6e7c19d2d9` (Epoch 5)

All evaluation scripts, manifests, and test metrics are synchronized in `CAVE_Flood_Audited_Release_v3.zip` with manifest `SHA256SUMS_v3.txt`.
