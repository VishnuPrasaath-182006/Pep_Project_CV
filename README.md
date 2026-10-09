# CAVE-Flood: Confidence-Aware Vision-Language-Guided Residual Refinement for Flood Inundation Segmentation

[![Audit Status](https://img.shields.io/badge/Audit%20Status-VALID%20WITH%20LIMITATIONS-yellow.svg)](#master-component-audit-summary)
[![Release Status](https://img.shields.io/badge/Release-FROZEN%20v1-blue.svg)](#release-packages--cryptographic-checksums)
[![Architecture](https://img.shields.io/badge/Architecture-UNet--Mamba%20%2B%20CASR-6f42c1.svg)](#architecture--residual-refinement-formulation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Trainable Params](https://img.shields.io/badge/Trainable%20Params-12%2C035%20(0.084%25)-green.svg)](#parameter-footprint--efficiency)
[![Sensor Product](https://img.shields.io/badge/Sensor-PlanetScope%204--band%20(3m)-orange.svg)](#dataset-partitioning--sensor-specifications)
[![Automated Audit](https://img.shields.io/badge/Automated%20Audit-100%25%20PASS-brightgreen.svg)](#quickstart--automated-reproduction)

Official repository for **CAVE-Flood v1**, a parameter-efficient residual refinement architecture designed for high-resolution ($3\text{m}$) multispectral satellite flood inundation mapping. CAVE-Flood introduces the **Confidence-Aware Semantic Residual (CASR)** adapter head to refine boundary predictions from a frozen **UNet-Mamba** foundation backbone using spatial uncertainty gating and vision-language guidance from frozen **RemoteCLIP**.

---

## Table of Contents
- [Executive Overview & Release Freeze](#executive-overview--release-freeze)
- [Repository Directory Structure](#repository-directory-structure)
- [Master Component Audit Summary](#master-component-audit-summary)
- [Dataset Partitioning & Sensor Specifications](#dataset-partitioning--sensor-specifications)
- [Architecture & Residual Refinement Formulation](#architecture--residual-refinement-formulation)
- [Parameter Footprint & Computational Efficiency](#parameter-footprint--computational-efficiency)
- [Authoritative Performance Tables](#authoritative-performance-tables)
  - [Held-Out Test Set Performance (46.14M Valid Pixels)](#held-out-test-set-performance-4614m-valid-pixels)
  - [Held-Out Test Confusion Matrices (Exact Physical Pixels)](#held-out-test-confusion-matrices-exact-physical-pixels)
  - [Per-Event Breakdown (Asian Flood Regimes)](#per-event-breakdown-asian-flood-regimes)
  - [Validation Set Performance (117.36M Valid Pixels)](#validation-set-performance-11736m-valid-pixels)
- [Reconciliation of Earlier Reporting Discrepancies](#reconciliation-of-earlier-reporting-discrepancies)
- [Scene-Level XAI Diagnostics & Exact Integer Error Conservation](#scene-level-xai-diagnostics--exact-integer-error-conservation)
- [LayerCAM Formulation & Non-Causal Interpretability Boundaries](#layercam-formulation--non-causal-interpretability-boundaries)
- [Critical Scientific Limitations](#critical-scientific-limitations)
- [Complete Result Folders & Files Inventory](#complete-result-folders--files-inventory)
- [Release Packages & Cryptographic Checksums](#release-packages--cryptographic-checksums)
- [Quickstart & Automated Reproduction](#quickstart--automated-reproduction)
- [Project Changelog](#project-changelog)
- [Citation & Attribution](#citation--attribution)

---

## Executive Overview & Release Freeze

Satellite flood inundation segmentation faces severe boundary ambiguity due to sediment-laden turbid waters, submerged canopies, mixed pixels, and terrain shadows. Traditional fine-tuning of deep vision foundation models incurs high computational overhead and risks catastrophic forgetting.

**CAVE-Flood** addresses this challenge via a parameter-efficient, confidence-gated residual refinement paradigm:
1. **Backbone Invariance**: The 14.29M-parameter UNet-Mamba segmentation model remains **100% frozen** (`best_model.pth`).
2. **Ultra-Lightweight Adapter**: The trainable CASR adapter head adds only **12,035 parameters** ($0.084\%$ overhead; checkpoint size: **54.4 KB**).
3. **Selective Candidate Gating**: An entropy- and gradient-based candidate selector isolates the **top 25% most uncertain boundary tiles**. The remaining $75\%$ of the image strictly bypasses adaptation with exact zero-residual invariance ($\Delta Z \equiv 0$).
4. **Controlled Vision-Language Guidance**: Evaluates frozen RemoteCLIP text-image cosine embeddings (**E2**) against neutral scalar guidance (**E1**).
5. **Rigorous Independent Verification**: Following an exhaustive independent mathematical audit, all reported metrics, confusion matrices, and scene-level error transitions were reconciled against physical ground-truth label arrays across 163.5 million valid pixels.

> [!NOTE]
> **Research Implementation Freeze**:  
> Following the resolution of all sensor descriptions, scene-level error counts, LayerCAM target objectives, and cryptographic checkpoint integrity checks, **the research implementation of CAVE-Flood v1 is formally FROZEN**. No further architectural modifications or experimentation cycles will be initiated.

---

## Repository Directory Structure

```
BaseModel Extension/
├── README.md                                          <- Master authoritative GitHub README (this file)
├── requirements.txt                                   <- Python dependencies (unified master requirements)
├── .gitignore                                         <- Git ignore patterns (unified master gitignore)
│
├── UNet-Mamba/                                        <- Base Model Repository (Original baseline implementation)
│   ├── checkpoints/                                   <- Base model weights snapshots (best_model.pth, etc.)
│   ├── configs/                                       <- Original baseline training configuration
│   ├── manifests/                                     <- Baseline dataset manifests
│   ├── results/                                       <- Baseline training logs, figures & metrics
│   ├── src/                                           <- Core UNet-Mamba model, dataset & training source files
│   ├── Folder_structure_and_details.txt               <- Baseline repository structure inventory
│   ├── environment.yml                                <- Conda environment definition
│   └── run_pipeline.py                                <- Pipeline entrypoint
│
└── UNet-Mamba Extension/                              <- CAVE-Flood Extension Repository (Audited Deliverables)
    ├── Dataset/                                       <- 19 raw geographic event regions (PlanetScope multispectral)
    ├── artifacts/                                     <- Diagnostic figures, summary metrics & candidate cache
    │   └── CAVE_Flood_Antigravity/
    │       ├── stage6_xai_summary.json                <- Authoritative integer confusion counts for XAI scenes
    │       ├── remoteclip_score_scaler.json           <- Min-max calibration config for RemoteCLIP
    │       ├── figures/                               <- Refreshed 8-panel XAI diagnostic dashboards
    │       └── cache/                                 <- Precomputed candidate masks & validation logits
    ├── checkpoints/                                   <- Cryptographically locked PyTorch model weights
    │   ├── best_model.pth                             <- Frozen UNet-Mamba baseline (14.29M params, 171.8 MB)
    │   ├── best_e1_model.pth                          <- Trained CASR E1 adapter (12,035 params, 54.4 KB)
    │   └── best_e2_model.pth                          <- Trained CASR E2 adapter (12,035 params, 54.4 KB)
    ├── configs/
    │   └── train_config.yaml                          <- Model hyperparameters & normalization parameters
    ├── manifests/                                     <- Event split manifests and scene indices
    ├── releases/                                      <- Release archives & cryptographic manifests
    │   ├── CAVE_Flood_Audited_Release_v3.zip          <- Audited release archive (17.74 MB)
    │   ├── Results and Outputs.zip                    <- Complete deliverables & results package (2.71 GB)
    │   └── SHA256SUMS_v3.txt                          <- Master cryptographic checksum ledger (19 entries)
    ├── results/                                       <- Unified results, evaluations & experimental logs
    │   ├── demo_output/                               <- Inference demonstration plots
    │   │   └── demo_inference_Bolivia_BOL_1015_p0.png
    │   ├── final/                                     <- Authoritative test metrics & reconciliation reports
    │   │   ├── CAVE_Flood_Final_Independent_Audit_Report.html
    │   │   ├── CAVE_Flood_Final_Report_v3_CORRECTED.md
    │   │   ├── CAVE_Flood_Final_Report_CORRECTED.md
    │   │   ├── final_test_summary_corrected_v2.csv
    │   │   ├── final_test_event_metrics_corrected_v2.csv
    │   │   ├── final_test_metrics_corrected_v2.json
    │   │   ├── final_test_evaluation.log
    │   │   ├── METRIC_RECONCILIATION_AUDIT.md
    │   │   ├── REPORT_METRIC_RECONCILIATION.md
    │   │   └── SHA256SUMS_v3.txt
    │   ├── figures/                                   <- Baseline curves & sample visualizations
    │   │   ├── confusion_matrix.png
    │   │   ├── training_curves.png
    │   │   ├── processed_comparisons/                 <- 3 preprocessed patch comparison plots
    │   │   └── raw_samples/                           <- 5 raw satellite image inspection plots
    │   ├── logs/                                      <- E1 and E2 epoch-by-epoch training logs
    │   ├── metrics/                                   <- Validation & test metrics
    │   └── reports/                                   <- Dataset inventory & class balance reports
    └── src/
        ├── run_all_audits.py                          <- Master single-command automated test runner
        ├── audit_gt_pixel_counts.py                   <- Independent raw label counter
        ├── audit_reconciliation.py                    <- Metric reconciliation & conservation auditor
        ├── audit_xai_layercam.py                      <- LayerCAM & scene confusion matrix auditor
        ├── build_html_audit_report.py                 <- Interactive HTML audit report generator
        ├── create_release_bundle_v3.py                <- Release archive packaging tool
        ├── casr.py                                    <- CASR adapter PyTorch architecture
        ├── candidate_selector.py                      <- Entropy/gradient candidate tile selector
        ├── train_casr.py                              <- CASR training pipeline
        ├── evaluate_final_test.py                     <- Single-pass held-out test evaluator
        ├── demo_inference.py                          <- End-to-end interactive demo inference script
        └── run_stage6_xai.py                          <- LayerCAM & XAI diagnostic generation script
```

---

## Master Component Audit Summary

| Component | Status | Empirical Methodology | Findings & Action |
| :--- | :---: | :--- | :--- |
| **Dataset Integrity** | **PASS** | `audit_gt_pixel_counts.py` scanned raw ground-truth masks across all 44 test scenes from non-overlapping quadrants. | Exactly **46,137,344** valid pixels counted (**26,566,620** Flood, **19,570,724** Background). Zero NoData or ignored label leakage. |
| **Confusion Matrices** | **PASS** | Forensic audit proved that the initial report's matrix ($\text{TP}=11.21\text{M}$) was an artificial LLM hallucination with a $13.15\text{M}$ flood-pixel deficit. | Evaluator's matrix ($\text{TP}=22,201,266$, $\text{FN}=4,365,354$) matches raw physical labels **to the exact integer**. |
| **Metrics Verification** | **PASS** | `audit_reconciliation.py` recalculated all binary segmentation metrics across validation ($117.36\text{M}$ px) and test ($46.14\text{M}$ px). | Reconciled across CSV, JSON, Markdown, and HTML reports. All division-by-zero safeguards verified. |
| **UNet-Mamba Backbone** | **PASS** | Frozen backbone inspected. SHA-256 hash `58674c84...` verified untouched. | Exactly 0 trainable parameters during adapter training. Strict zero-gradient freeze maintained. |
| **CASR Architecture** | **PASS** | `CASRAdapter` verified: exactly **12,035 trainable parameters** across dual convolution blocks, spatial sigmoid gate, and 2-channel residual head. | Exact zero-residual equivalence verified for $C=0$ or $\lambda=0$ (`torch.equal == True`). |
| **RemoteCLIP Guidance** | **PASS** | Frozen ViT-B/32 text/image cosine similarity score $g$. Both E1 and E2 received identical non-guidance inputs. | Controlled ablation confirmed RemoteCLIP provides marginal gain ($\le 0.02\text{ pp}$) over neutral guidance. |
| **LayerCAM Safety** | **PASS** | Target layer `casr.block2` verified differentiable ($512\times 512\times 32$). Gradients verified 100% finite (0 NaNs, 0 Infs). | Wrapped in `try...finally` with `model_adapter.zero_grad()` to ensure provably zero parameter drift. |
| **XAI Diagnostics** | **PASS** | Orientation, dimensions ($1024\times 1024$), and candidate coverage ($25.0\%$) verified. Edit concentration strictly $100.0\%$. | Exact integer error conservation verified across all 3 representative scenes. |
| **Reproducibility** | **PASS** | All audit scripts execute autonomously and deterministically from clean terminal environments without GPU requirements. | Master audit runner `run_all_audits.py` passes 100% of checks cleanly with exit code 0. |

---

## Dataset Partitioning & Sensor Specifications

The experimental dataset comprises **four-band PlanetScope imagery from the FloodPlanet dataset**, partitioned into strictly disjoint geographic events to evaluate out-of-domain generalization:

- **Physical Sensor**: PlanetScope (PS) constellation multispectral sensors.
- **Spectral Bands**:
  - Band 0: Red ($650\text{ nm}$)
  - Band 1: Green ($560\text{ nm}$)
  - Band 2: Blue ($485\text{ nm}$)
  - Band 3: Near-Infrared (NIR, $860\text{ nm}$)
- **Spatial Resolution**: $3.0\text{ meters per pixel}$ Ground Sampling Distance (GSD).
- **Tile Dimensions**: $512 \times 512$ pixel quadrant patches extracted from $1024 \times 1024$ full scenes.
- **Radiometric Bit Depth**: 16-bit unsigned integers (`uint16`) normalized via training set channel z-score statistics (`mean=[1587.2, 1792.4, 1845.6, 2642.1]`, `std=[842.1, 794.3, 762.5, 1124.6]`).

### Geographic Event Partitions:
1. **Training Split**: 14 flood events across diverse global geographic regimes.
2. **Validation Split (114 full scenes, 456 patches, 117,355,812 valid pixels)**:
   - **Bolivia**: 72 scenes (Amazonian floodplain and meandering channels).
   - **US-Dakota**: 42 scenes (Agricultural plains, glacial prairie potholes).
3. **Held-Out Test Split (44 full scenes, 176 patches, 46,137,344 valid pixels)**:
   - **Bangladesh**: 15 scenes (Lowland delta, extreme turbidity, braided rivers).
   - **Cambodia**: 20 scenes (Tonle Sap lacustrine basin, seasonal expansion).
   - **Nepal**: 9 scenes (Himalayan mountainous terrain, steep valleys, terrain shadow).

---

## Architecture & Residual Refinement Formulation

```
                                  +------------------------------------+
                                  | 4-band PlanetScope Patch (512x512) |
                                  +-----------------+------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
          +------------------------------+                      +------------------------------+
          | Frozen UNet-Mamba (14.29M)   |                      | Frozen RemoteCLIP (ViT-B/32) |
          +--------------+---------------+                      +--------------+---------------+
                         |                                                     |
                         v Logits Z_base [B, 2, 512, 512]                      v Text-Image Sim g
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |     Candidate Selector             |
                                  |  0.5 rank(H) + 0.5 rank(||grad P||)|
                                  |  -> Binary Mask C (Top 4 of 16)    |
                                  +-----------------+------------------+
                                                    |
                                                    v 9-channel Input Tensor
                                  +------------------------------------+
                                  | CASR Adapter Module (12,035 params)|
                                  | - Conv Block 1 (9 -> 32, 3x3)      |
                                  | - Conv Block 2 (32 -> 32, 3x3)     |
                                  |   +-- Gate Head -> Sigmoid Gate G  |
                                  |   +-- Res Head  -> Delta Z [2 ch]  |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Residual Injection:                |
                                  | Z_final = Z_base + lambda*C*G*res  |
                                  +------------------------------------+
```

### Mathematical Formulation:
1. **Predictive Uncertainty & Boundary Gradient**:
   $$H(i, j) = - \sum_{c=0}^1 P_c(i, j) \log_2 P_c(i, j)$$
   $$\|\nabla P\|(i, j) = \sqrt{\left(\frac{\partial P_1}{\partial x}\right)^2 + \left(\frac{\partial P_1}{\partial y}\right)^2}$$
2. **Candidate Region Selection**:
   The $512 \times 512$ image is divided into a $4 \times 4$ grid of $128 \times 128$ non-overlapping tiles. Each tile $t$ is scored by:
   $$S_t = 0.5 \cdot \text{rank}(\bar{H}_t) + 0.5 \cdot \text{rank}(\|\bar{\nabla P}\|_t)$$
   The top 4 tiles ($25\%$ spatial coverage) form the binary candidate mask $C \in \{0, 1\}^{512 \times 512}$.
3. **9-Channel Feature Assembly**:
   $$F = \text{Concat}\big(X_{\text{norm}}, P_1, H, \|\nabla P\|, C, g \cdot \mathbf{1}\big) \in \mathbb{R}^{9 \times 512 \times 512}$$
4. **Logit Injection & Zero-Residual Guarantee**:
   $$Z_{\text{final}} = Z_{\text{base}} + \lambda \cdot C \odot G \odot \Delta Z$$
   where $\lambda = 1.0$, $G = \sigma(\text{Conv}_{1\times 1}(A_2))$, and $\Delta Z = \text{Conv}_{1\times 1}(A_2)$.
   When $C=0$ (outside candidate regions) or at initialization ($\lambda=0$):
   $$Z_{\text{final}} \equiv Z_{\text{base}} \quad (\max |Z_{\text{final}} - Z_{\text{base}}| = 0.0000)$$

---

## Parameter Footprint & Computational Efficiency

| Component | Status | Parameter Count | Footprint vs Base | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: |
| **UNet-Mamba Backbone** | Frozen | 14,294,018 | 100.0% | 171.82 MB |
| **CASR Adapter Head** | Trainable | **12,035** | **0.084%** | **54.43 KB** |
| **RemoteCLIP ViT-B/32** | Frozen (E2 only) | 87,849,217 | 614.6% | ~350 MB |
| **Total Pipeline (E1: Neutral)** | Evaluated | 14,306,053 | 100.08% | ~171.9 MB |
| **Total Pipeline (E2: RemoteCLIP)** | Evaluated | 102,155,270 | 714.7% | ~522 MB |

---

## Authoritative Performance Tables

### Held-Out Test Set Performance (46.14M Valid Pixels)
Evaluated single-pass on 44 source chips across Bangladesh (15 chips), Cambodia (20 chips), and Nepal (9 chips):

| Metric | E0 Baseline | E1 CASR (Neutral) | E2 CASR (RemoteCLIP) | Test Winner | $\Delta$ E1 vs E0 | $\Delta$ E2 vs E0 | $\Delta$ E2 vs E1 |
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

---

### Held-Out Test Confusion Matrices (Exact Physical Pixels)
Derived from the sum of raw physical labels across all 44 test quadrant pairs:
$$\text{Total Valid Pixels} = 46,137,344 \quad (\text{GT Flood} = 26,566,620, \quad \text{GT Background} = 19,570,724)$$

| Model | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Sum Check |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **E0 Baseline** | 22,201,266 | 2,046,482 | 4,365,354 | 17,524,242 | 46,137,344 [PASS] |
| **E1 CASR (Neutral)** | **22,425,765** (+224,499) | 2,267,106 (+220,624) | **4,140,855** (-224,499) | 17,303,618 (-220,624) | 46,137,344 [PASS] |
| **E2 CASR (RemoteCLIP)** | 22,404,736 (+203,470) | 2,241,494 (+195,012) | 4,161,884 (-203,470) | 17,329,230 (-195,012) | 46,137,344 [PASS] |

$$\text{Net Test Pixel Gain}: \quad \text{E1} = 224,499 - 220,624 = \mathbf{+3,875\text{ pixels}}, \quad \text{E2} = 203,470 - 195,012 = \mathbf{+8,458\text{ pixels}}.$$

---

### Per-Event Breakdown (Asian Flood Regimes)

| Event | Chips | Valid Pixels | Model | Flood IoU | Flood Dice | Precision | Recall | Specificity | Pixel Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bangladesh** | 15 | 15,728,640 | **E0** | 0.5711 | 0.7270 | **0.7678** | 0.6903 | **0.9069** | **0.8401** |
| *(Turbid delta)* | | | **E1** | 0.5746 (+0.35 pp) | 0.7298 (+0.28 pp) | 0.7577 (-1.01 pp) | **0.7039** (+1.36 pp) | 0.8997 (-0.72 pp) | 0.8393 (-0.08 pp) |
| | | | **E2** | **0.5754** (**+0.43 pp**) | **0.7305** (**+0.35 pp**) | 0.7595 (-0.83 pp) | 0.7036 (+1.33 pp) | 0.9007 (-0.62 pp) | 0.8399 (-0.02 pp) |
| **Cambodia** | 20 | 20,971,520 | **E0** | 0.8103 | 0.8952 | **0.9537** | 0.8434 | **0.8729** | 0.8506 |
| *(Tonle Sap basin)* | | | **E1** | **0.8139** (**+0.36 pp**) | **0.8974** (**+0.22 pp**) | 0.9482 (-0.55 pp) | **0.8518** (+0.84 pp) | 0.8553 (-1.76 pp) | **0.8527** (+0.21 pp) |
| | | | **E2** | 0.8133 (+0.30 pp) | 0.8971 (+0.19 pp) | 0.9488 (-0.49 pp) | 0.8507 (+0.73 pp) | 0.8572 (-1.57 pp) | 0.8523 (+0.17 pp) |
| **Nepal** | 9 | 9,437,184 | **E0** | **0.8774** | **0.9347** | **0.9343** | 0.9352 | **0.8927** | **0.9190** |
| *(Mountain relief)* | | | **E1** | 0.8743 (-0.31 pp) | 0.9329 (-0.18 pp) | 0.9264 (-0.79 pp) | **0.9396** (+0.44 pp) | 0.8782 (-1.45 pp) | 0.9162 (-0.28 pp) |
| | | | **E2** | 0.8747 (-0.27 pp) | 0.9332 (-0.15 pp) | 0.9272 (-0.71 pp) | 0.9393 (+0.41 pp) | 0.8797 (-1.30 pp) | 0.9166 (-0.24 pp) |

---

### Validation Set Performance (117.36M Valid Pixels)
Evaluated on 114 source scenes across Bolivia (72 scenes) and US-Dakota (42 scenes):

| Metric | E0 Baseline | E1 CASR (Neutral) | E2 CASR (RemoteCLIP) | Val Winner | $\Delta$ E1 vs E0 | $\Delta$ E2 vs E0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Event Flood IoU** | `0.648970` | **`0.653316`** | `0.653148` | **E1** | **+0.43 pp** | **+0.41 pp** |
| **Macro mIoU** | `0.661969` | `0.666783` | **`0.667019`** | **E2** | **+0.48 pp** | **+0.50 pp** |
| **Pixel-Pooled Flood IoU** | `0.592679` | **`0.601524`** | `0.601443` | **E1** | **+0.88 pp** | **+0.87 pp** |
| **Pixel-Pooled Flood Dice** | `0.744254` | **`0.751190`** | `0.751127` | **E1** | **+0.69 pp** | **+0.69 pp** |
| **Pooled Flood Precision** | **`0.775446`** | `0.769970` | `0.769165` | **E0** | -0.55 pp | -0.63 pp |
| **Pooled Flood Recall** | `0.715474` | `0.732890` | **`0.733306`** | **E2** | **+1.74 pp** | **+1.78 pp** |
| **Pixel Accuracy** | `0.806841` | `0.806894` | **`0.806950`** | **E2** | **+0.01 pp** | **+0.01 pp** |

---

## Reconciliation of Earlier Reporting Discrepancies

During independent review, two major reporting anomalies were detected and resolved:

### 1. The Test Accuracy Discrepancy
- **Original Reported Text**: `0.965421` (E0), `0.965128` (E1), `0.965215` (E2).
- **Calculated from Synthetic Matrix**: $\frac{11.21\text{M} + 31.69\text{M}}{46.14\text{M}} = \mathbf{0.929817}$.
- **True Evaluator Ground Truth**: $\frac{22,201,266 + 17,524,242}{46,137,344} = \mathbf{0.861027}$.
- **Resolution**: The text number `0.965421` was an unverified hallucination. The synthetic matrix was reverse-engineered by an LLM to match ratios but omitted $13.15\text{M}$ true flood pixels. The **true evaluator test accuracy is 86.10% (`0.861027`)**, which exactly matches the physical ground-truth labels.

### 2. The Macro mIoU Discrepancy
- **Original Reported Text**: `0.754020` (E0), `0.753758` (E1), `0.753955` (E2).
- **Calculated from Synthetic Matrix**: $\frac{0.775911 + 0.907286}{2} = \mathbf{0.841599}$.
- **Calculated from Authoritative Evaluator Matrix**: $\frac{0.775913 + 0.732127}{2} = \mathbf{0.754020}$.
- **Resolution**: The reported text value (`0.754020`) was genuine and correct from the evaluator. The apparent contradiction arose because an artificial confusion matrix gave `0.841599`. Restoring the true confusion counts restores mathematical identity.

---

## Scene-Level XAI Diagnostics & Exact Integer Error Conservation

Error statistics and pixel-level transitions were verified across three representative validation scenes using `UNet-Mamba/src/audit_xai_layercam.py`. Every scene satisfies exact mathematical conservation:

| Scene Identifier | Event | GT Flood Px | Valid Pixels | Baseline FN ($E_0$) | Refined FN ($E_2$) | Net FN Reduction ($\Delta\text{FN}$) | Gross $\text{FN}\rightarrow\text{TP}$ | Reverse $\text{TP}\rightarrow\text{FN}$ | Baseline FP ($E_0$) | Refined FP ($E_2$) | Net FP Increase ($\Delta\text{FP}$) | Gross $\text{TN}\rightarrow\text{FP}$ | Suppressed $\text{FP}\rightarrow\text{TN}$ | Net Correct Pixel Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Bolivia_BOL_1176`** | Bolivia | 358,861 | 1,048,576 | 93,261 | 67,897 | **-25,364 px** | +25,364 | -0 | 105,341 | 112,534 | **+7,193 px** | +7,195 | -2 | **+18,171 px** |
| **`US-Dakota_RRN_10_46`**| Dakota | 836,574 | 1,048,576 | 282,845 | 268,407 | **-14,438 px** | +14,452 | -14 | 73,473 | 77,679 | **+4,206 px** | +4,212 | -6 | **+10,232 px** |
| **`Bolivia_BOL_1123`** | Bolivia | 280,385 | 1,048,576 | 139,462 | 124,047 | **-15,415 px** | +15,416 | -1 | 32,642 | 37,273 | **+4,631 px** | +4,637 | -6 | **+10,784 px** |
| **Total (3 Scenes)** | — | 1,475,820 | 3,145,728 | 515,568 | 460,351 | **-55,217 px** | **+55,232** | **-15** | 211,456 | 227,486 | **+16,030 px** | **+16,044** | **-14** | **+39,187 px** |

$$\text{Net Correct Gain} = |\Delta\text{FN}| - \Delta\text{FP} = 55,217 - 16,030 = \mathbf{+39,187\text{ pixels strictly conserved}}.$$

### Spatial Gating & Candidate Concentration:
- **Candidate Region Coverage**: Exactly **25.0%** of scene area ($262,144$ of $1,048,576$ px; top 4 of 16 tiles).
- **Edit Concentration**: Strictly **100.0%** of all decision flips occur inside the candidate mask.
- **Effective Gate Outside Candidates**: Identically **0.0000** due to multiplicative bypass masking.

---

## LayerCAM Formulation & Non-Causal Interpretability Boundaries

LayerCAM (Jiang et al., IEEE TIP 2021) targets the deepest convolutional block of the adapter head (`casr.block2`):
1. **Pixel-Wise Spatial Weighting**:
   $$w_k(i, j) = \text{ReLU}\left(\frac{\partial \text{Score}}{\partial A_k(i, j)}\right)$$
2. **Channel Accumulation & Normalization**:
   $$\hat{M}(i, j) = \text{ReLU}\left(\sum_{k=1}^K w_k(i, j) \cdot A_k(i, j)\right)$$
   $$\text{CAM}(i, j) = \frac{\hat{M}(i, j) - \min(\hat{M})}{\max(\hat{M}) - \min(\hat{M}) + \epsilon}$$
3. **Target Objective**:
   $$\text{Score} = \sum_{(i,j) \in C} \Delta Z_1(i, j)$$

> [!IMPORTANT]
> **LayerCAM Target Objective & Interpretability Scope**:
> LayerCAM is computed strictly with respect to the CASR residual flood-logit correction ($\Delta Z_1$) within candidate regions ($C=1$).
> LayerCAM attributions confirm that the adapter's convolutional filters respond with high spatial sensitivity to ambiguous water-land boundaries inside the top-ranked $25\%$ candidate area. However:
> 1. LayerCAM attributions **do not prove causal semantic reasoning or language grounding** by the network.
> 2. LayerCAM attributions **do not explain final segmentation decisions outside candidate regions**, where predictions are determined $100\%$ by the frozen UNet-Mamba backbone.

---

## Critical Scientific Limitations

1. **Marginality of RemoteCLIP Guidance**:
   On held-out test, RemoteCLIP (E2) outperforms neutral guidance (E1) by only **+0.02 percentage points** on Mean Event Flood IoU ($0.754468$ vs $0.754259$), and trails E1 slightly on Pixel-Pooled Flood IoU ($0.777723$ vs $0.777762$). RemoteCLIP text embeddings cannot be claimed as a major driver of segmentation accuracy; the primary performance gain comes from the spatial residual adapter.
2. **Systematic Precision Trade-off**:
   Across both validation and test sets, CASR acts as an inundation boundary expander, trading off precision ($-0.65$ to $-0.74\text{ pp}$) to gain recall ($+0.77$ to $+0.85\text{ pp}$).
3. **Regional Degradation in Mountainous Relief**:
   In Nepal, CASR over-segments mountain shadow and steep rock boundaries, degrading Flood IoU by **$-0.27\text{ pp}$** (E2) and **$-0.31\text{ pp}$** (E1).
4. **Correlational XAI**:
   LayerCAM demonstrates spatial gradient correlation with ambiguous boundaries, but does not prove causal linguistic understanding.
5. **No External SOTA Claims**:
   The model is an effective parameter-efficient adapter against its frozen baseline, but has not been benchmarked against fine-tuned foundation models (e.g. SegFormer, Prithvi).

---

## Complete Result Folders & Files Inventory

| Directory / File Path | Size | Description & Scientific Role |
| :--- | :---: | :--- |
| **`UNet-Mamba/releases/Results and Outputs.zip`** | $2.71\text{ GB}$ | Complete consolidated ZIP archive of all unified results, outputs, figures, logs, reports, and artifacts. |
| **`UNet-Mamba/releases/CAVE_Flood_Audited_Release_v3.zip`** | $17.74\text{ MB}$ | Audited release archive containing final code, configs, logs, metrics, figures, and adapter weights. |
| **`UNet-Mamba/releases/SHA256SUMS_v3.txt`** | $2.15\text{ KB}$ | Master cryptographic SHA-256 verification manifest across release files and checkpoints. |
| **`UNet-Mamba/results/final/CAVE_Flood_Final_Independent_Audit_Report.html`** | $24.79\text{ KB}$ | Publication-grade interactive HTML independent verification and audit report. |
| **`UNet-Mamba/results/demo_output/`** | $798.6\text{ KB}$ | End-to-end forward pass visualization on raw PlanetScope patch `BOL_1015_p0`. |
| **`UNet-Mamba/results/final/`** | $122.3\text{ KB}$ | Authoritative test metrics JSON/CSVs, single-pass test log, HTML reports, and reconciliation audits. |
| **`UNet-Mamba/results/figures/`** | $8.25\text{ MB}$ | Baseline loss curves, confusion matrices, 5 raw scene samples, and 3 processed patch comparisons. |
| **`UNet-Mamba/results/logs/`** | $23.8\text{ KB}$ | Complete training history CSV and JSON files for baseline, E1, and E2 runs. |
| **`UNet-Mamba/results/metrics/`** | $3.5\text{ KB}$ | Validation metrics across 114 source chips ($117.36\text{M}$ pixels) and legacy summaries. |
| **`UNet-Mamba/results/reports/`** | $104.2\text{ KB}$ | Scene inventories, raw data inspection markdown, and class balance reports. |
| **`UNet-Mamba/checkpoints/`** | $171.93\text{ MB}$ | Pretrained baseline (`best_model.pth`, 171.8 MB) and trained adapter checkpoints (E1/E2, 54.4 KB each). |
| **`UNet-Mamba/artifacts/CAVE_Flood_Antigravity/figures/`** | $16.14\text{ MB}$ | 6 high-resolution PNGs (3 4-panel error comparison maps + 3 8-panel diagnostic dashboards). |
| **`UNet-Mamba/artifacts/CAVE_Flood_Antigravity/`** | $17.9\text{ KB}$ | `stage6_xai_summary.json` (exact counts), `remoteclip_score_scaler.json`, and Stage 0-6 audit report. |

---

## Release Packages & Cryptographic Checksums

All checkpoints and release archives are cryptographically locked and verified:

| File Name | SHA-256 Cryptographic Checksum | Size |
| :--- | :--- | :---: |
| `UNet-Mamba/checkpoints/best_model.pth` | `58674c84e97e16c7ab86f2e218ba8738c7aaf69a8c6fd8c384de00beec9eb76d` | 171.82 MB |
| `UNet-Mamba/checkpoints/best_e1_model.pth` | `84a02d1f1fe8371bf799edb0ddce28f69d20dd90b3da503c06eac87f521226e9` | 54.43 KB |
| `UNet-Mamba/checkpoints/best_e2_model.pth` | `19347f5f64385c46f16c27b345a7f143e56b51f23c1fe7e887906b6e7c19d2d9` | 54.43 KB |
| `UNet-Mamba/releases/Results and Outputs.zip` | `989bdd4efaa89f0d29385ccc2ab1c2a9fec1077341c64cbab8d3cf8b759c2c49` | 2.71 GB |
| `UNet-Mamba/releases/CAVE_Flood_Audited_Release_v3.zip` | `ffeaedac4dc53d015221eb6b3455102732b70740d4f6ce7fb8caa1a6193b6a34` | 17.74 MB |

---

## Quickstart & Automated Reproduction

### 1. Run Complete Automated Audit Suite (Zero-GPU, Fast)
Verifies all checkpoints, raw ground-truth label counts, metric formulas, LayerCAM zero-drift invariants, and scene-level error conservation laws in under 30 seconds:
```powershell
python UNet-Mamba/src/run_all_audits.py
```
*Expected Output: Exits with code 0 (`ALL INDEPENDENT AUDITS AND AUTOMATED TESTS PASSED SUCCESSFULLY!`).*

### 2. Run Interactive Demo Inference
Executes end-to-end forward inference on a raw PlanetScope satellite chip:
```powershell
python UNet-Mamba/src/demo_inference.py --patch_path UNet-Mamba/artifacts/CAVE_Flood_Antigravity/cache/candidate_regions/val/Bolivia_BOL_1015_p0.npz --mode E2
```

### 3. Rebuild Publication HTML Report
Recompiles the standalone, interactive HTML edition:
```powershell
python build_html_audit_report.py
```

---

## Project Changelog

- **v1.0.0 (Stage 7 Initial Release)**: Completed CASR E1 and E2 training; held-out test evaluation on Asian flood events.
- **v2.0.0 (Stage 8 Independent Audit)**: Identified LLM-hallucinated test accuracy ($0.9654$) and synthetic confusion matrix. Reconstructed physical ground truth ($46.14\text{M}$ pixels); confirmed evaluator test accuracy ($0.861027$).
- **v3.0.0 (Final Release Freeze)**:
  - Corrected sensor description to 4-band PlanetScope imagery from FloodPlanet; eliminated Sentinel-2 references.
  - Reconciled scene-level XAI counts: documented Net FN Reductions ($-14,438$ Dakota, $-15,415$ Bolivia 1123) alongside gross one-way flips ($+14,452$ and $+15,416$).
  - Clarified LayerCAM target objective ($\Delta Z_1 \cdot C$) and documented non-causal interpretability boundaries.
  - Consolidated documentation into a single master `README.md` and packaged all outputs into `Results and Outputs.zip`.
  - Formally froze the research implementation.

---

## Citation & Attribution

```bibtex
@article{cave_flood_2026,
  title={CAVE-Flood: Confidence-Aware Vision-Language-Guided Residual Refinement for Flood Inundation Segmentation},
  author={CAVE-Flood Research Team},
  year={2026},
  journal={Independent Scientific Verification Report},
  note={Verified with Documented Limitations}
}
```
