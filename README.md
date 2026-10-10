# Satellite Flood Inundation Semantic Segmentation Suite

A unified research and benchmarking suite for high-resolution ($3\text{m}$) multispectral satellite flood inundation mapping using PlanetScope 4-band optical imagery.

This repository consolidates three computer vision and deep learning paradigms for disaster response and flood extent mapping:
1. **Classical CNN Baseline**: 4-Channel ResNet-34 U-Net (`unet/`)
2. **State Space Foundation & Parameter-Efficient Refinement**: UNet-Mamba and CAVE-Flood CASR with RemoteCLIP Vision-Language Guidance (`BaseModel and its Extension/`)
3. **Hierarchical Vision Transformer**: SegFormer-B2 with Leave-One-Event-Out Cross-Validation across 19 global events (`SegFormer/`)

---

## 1. Model Architectures & Approaches

| Model | Architecture Family | Input Channels | Total Parameters | Trainable Parameters | Key Innovation / Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **U-Net** | Convolutional Encoder-Decoder (ResNet-34) | 4 (B, G, R, NIR) | 17.27 M | 17.27 M | Fast, lightweight classical baseline with standard sliding-window inference |
| **UNet-Mamba** | Hybrid CNN + 2D Selective State Space (SS2D) | 4 (B, G, R, NIR) | 14.29 M | 14.29 M | Linear-complexity long-range spatial context modeling for satellite rasters |
| **CAVE-Flood E0** | Frozen UNet-Mamba Backbone | 4 (B, G, R, NIR) | 14.29 M | 0 (Frozen) | Base foundation model reference without residual adaptation |
| **CAVE-Flood E1** | Frozen UNet-Mamba + CASR Adapter (Neutral) | 4 (B, G, R, NIR) | 14.30 M | 12,035 (0.084%) | Confidence-aware boundary refinement gated on top 25% uncertain boundary regions |
| **CAVE-Flood E2** | Frozen UNet-Mamba + CASR Adapter (RemoteCLIP) | 4 (B, G, R, NIR) | 14.30 M | 12,035 (0.084%) | Vision-language guided boundary refinement using satellite-domain text-image cosine embeddings |
| **SegFormer-B2** | Hierarchical Mix Transformer (MiT-B2 + All-MLP) | 3 (R, G, B) | 27.50 M | 27.50 M | Global self-attention benchmark evaluated under rigorous Leave-One-Event-Out (LOEO) cross-validation |

---

## 2. Repository Structure

```
.
├── README.md                          # Master unified documentation (this file)
├── requirements.txt                   # Consolidated Python dependencies for all models
├── .gitignore                         # Master git exclusion rules
│
├── BaseModel and its Extension/       # Mamba & Vision-Language Architectures
│   ├── UNet-Mamba/                    # Base State Space Model implementation
│   │   ├── configs/                   # Model and dataset hyperparameters
│   │   ├── manifests/                 # Geographic event manifests and split indices
│   │   ├── src/                       # UNet-Mamba model, dataset & training modules
│   │   └── run_pipeline.py            # Baseline execution entrypoint
│   │
│   └── UNet-Mamba Extension/          # CAVE-Flood Residual Refinement (Audited)
│       ├── artifacts/                 # XAI summary, RemoteCLIP scaler, and cache
│       ├── checkpoints/               # Trained model weights (best_model.pth, best_e1_model.pth, best_e2_model.pth)
│       ├── configs/                   # Training and evaluation configs (train_config.yaml)
│       ├── manifests/                 # Geographic partition manifests
│       └── src/                       # CASR adapter head, RemoteCLIP guidance, and LayerCAM modules
│
├── SegFormer/                         # Hierarchical Transformer Pipeline
│   ├── configs/                       # LOEO-CV configuration (loeo_config.yaml)
│   ├── manifests/                     # 19-event leave-one-out splits (splits_19_events.json)
│   ├── scripts/                       # Training, evaluation & inference runners
│   └── src/                           # Dataset loaders, SegFormer wrapper & metrics
│
├── Gradio website/                    # Public Web Demonstration (Hugging Face Spaces)
│   ├── app.py                         # Gradio Blocks frontend & inference orchestrator
│   ├── image_adapter.py               # Universal multi-format ingestion & compatibility gating
│   ├── inference.py                   # E0/E1/E2 inference & model manager
│   ├── preprocessing.py               # 4-band satellite preprocessing & RGB preview
│   ├── visualization.py               # Mask, overlay, comparison & XAI visualizers
│   ├── xai_utils.py                   # LayerCAM & ground-truth validation metrics
│   ├── checkpoints/                   # Audited PyTorch model weights
│   ├── examples/                      # Compatible sample GeoTIFF rasters & test images
│   └── tests/                         # Automated deployment verification suite
│
└── unet/                              # Standalone Classical U-Net Baseline
    ├── configs/                       # Base U-Net parameters (base_unet.yaml, paths.yaml)
    ├── scripts/                       # Sliding-window inference (predict_floodplanet.py)
    ├── src/                           # 4-channel model, dataset, dataloader & train loop
    ├── tests/                         # Architecture verification & forward-pass unit tests (test_model.py)
    ├── run_inference.ps1              # 1-click prediction runner for Windows PowerShell
    ├── setup_windows.bat              # Setup batch script for Windows Command Prompt
    └── setup_windows.ps1              # Setup script for Windows PowerShell
```

---

## 3. Dataset & Sensor Specifications

All models evaluate against PlanetScope high-resolution optical satellite imagery from the **FloodPlanet** benchmark:

- **Spatial Resolution**: $3\text{m}$ ground sample distance (GSD).
- **Spectral Bands**:
  - Band 1: Blue ($\sim 455\text{--}515\,\text{nm}$)
  - Band 2: Green ($\sim 500\text{--}590\,\text{nm}$)
  - Band 3: Red ($\sim 590\text{--}670\,\text{nm}$)
  - Band 4: Near-Infrared / NIR ($\sim 780\text{--}860\,\text{nm}$)
- **Target Classes**:
  - `0`: Non-Flood / Land / Background
  - `1`: Flood Inundated Water Body
  - `255` / `-1`: Ignore / Invalid / Nodata
- **Geographic Flood Events**: 19 global events across diverse ecological, hydrological, and urban landscapes (Bangladesh, Bolivia, Cambodia, Colombia, Ghana, Nepal, Nigeria, Paraguay, Somalia, Spain, United States regions, Uzbekistan).

---

## 4. Unified Environment Setup

Ensure Python $\ge 3.10$ is installed, then set up the consolidated environment from the repository root:

### Option A: Standard Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Upgrade pip and install consolidated requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Conda Environment

```bash
conda create -n flood_seg python=3.11 -y
conda activate flood_seg
pip install -r requirements.txt
```

---

## 5. Usage & Execution Guides

### 5.1 Standalone U-Net (`unet/`)

#### Run Verification & Diagnostic Self-Test:
```powershell
cd unet
python tests/test_model.py
```
*Outputs a complete diagnostic table including compute device, parameter footprint ($17,268,034$), simulated tensor passes, and 4 passing unit tests.*

#### Run Sliding-Window Inference:
```powershell
python scripts/predict_floodplanet.py --image "path/to/scene.tif" --checkpoint "checkpoints/best_val_iou.pth" --output-dir "outputs/predictions"
```

---

### 5.2 UNet-Mamba & CAVE-Flood Extension (`BaseModel and its Extension/`)

#### UNet-Mamba Baseline Pipeline:
```powershell
cd "BaseModel and its Extension\UNet-Mamba"
python run_pipeline.py --mode evaluate --config configs/unetmamba_config.py
```

#### CAVE-Flood Residual Refinement (E1 & E2):
```powershell
cd "BaseModel and its Extension\UNet-Mamba Extension"

# Run Audited Verification & Architecture Audit
python src/audit_master.py

# Evaluate E1 (Neutral CASR Residual Head)
python src/evaluate.py --experiment E1 --config configs/train_config.yaml

# Evaluate E2 (RemoteCLIP Vision-Language Guidance)
python src/evaluate.py --experiment E2 --config configs/train_config.yaml
```

---

### 5.3 SegFormer-B2 LOEO Cross-Validation (`SegFormer/`)

#### Train a Leave-One-Event-Out Fold:
```powershell
cd SegFormer
python scripts/train_fold.py --fold 0 --config configs/loeo_config.yaml
```

#### Run Full 19-Fold Evaluation:
```powershell
python scripts/evaluate_all_folds.py --config configs/loeo_config.yaml
```

#### Generate Prediction Overlay:
```powershell
python scripts/inference_single.py --image-path "path/to/chip.tif" --fold 0 --output-dir "outputs/predictions"
```

---

## 6. Key Methodological Features

- **Parameter-Efficient Adaptation**: CAVE-Flood keeps the core 14.29M UNet-Mamba model strictly frozen and trains only 12,035 parameters in the CASR adapter head ($0.084\%$ overhead), preventing catastrophic forgetting while sharpening ambiguous water boundaries.
- **Top-25% Selective Uncertainty Gating**: Pixels with high prediction entropy or gradient magnitude are dynamically selected for adaptation; confident interior pixels bypass refinement with exact zero residual ($\Delta Z \equiv 0$).
- **Strict Event-Level Generalization**: SegFormer evaluated via Leave-One-Event-Out (LOEO) cross-validation to guarantee zero geographic or temporal leakage between train and test distributions.
- **Multispectral Exploitation**: U-Net and UNet-Mamba directly process 4-channel input rasters (including Near-Infrared), maximizing NDWI contrast between water bodies and vegetation.
