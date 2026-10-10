# CAVE-Flood: Intelligent Satellite Flood Inundation Analysis Platform

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg?logo=react)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/AI%20Framework-PyTorch%202.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A full-stack, research-grade geospatial web application for high-resolution satellite flood inundation segmentation. Built around **UNet-Mamba**, **Context-Aware Spatial Refinement (CASR)**, and **RemoteCLIP** vision-language guidance.

---

## 📁 Project Architecture & Folder Structure

```
PEP_Project_CV/
├── Frontend/                           # React.js + Vite Web Application
│   ├── public/                         # Public static assets & sample chips
│   ├── src/
│   │   ├── assets/                     # Icons and vector graphics
│   │   ├── components/                 # UI components (Header, Uploader, Viewer, etc.)
│   │   │   ├── Header.jsx              # System health & global navigation
│   │   │   ├── ImageUploader.jsx       # 3-mode drag-and-drop satellite uploader
│   │   │   ├── ImageMetadataPanel.jsx  # Raster metadata & CRS inspector
│   │   │   ├── ModelSelector.jsx       # Dynamic architecture selector with status
│   │   │   ├── InferenceViewer.jsx     # Side-by-side & opacity slider visualizer
│   │   │   ├── MetricsCard.jsx         # Quantitative flood statistics & delta audit
│   │   │   ├── ModelComparisonView.jsx # E0 vs E1 vs E2 comparative matrix
│   │   │   ├── XAIViewer.jsx           # LayerCAM CASR residual attribution
│   │   │   ├── GroundTruthEvaluator.jsx# Authentic IoU/Dice validation upload
│   │   │   ├── ExportToolbar.jsx       # Download PNGs, GeoTIFF, JSON, CSV, ZIP
│   │   │   └── AboutModal.jsx          # Model formulation & scientific details
│   │   ├── services/                   # Frontend API client (api.js)
│   │   ├── hooks/                      # Custom React hooks
│   │   ├── utils/                      # Formatting & image utilities
│   │   ├── styles/                     # Tailwind CSS & design tokens
│   │   │   └── index.css
│   │   ├── App.jsx                     # Main application layout
│   │   └── main.jsx                    # Vite entry point
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── .env.example
│
├── Backend/                            # FastAPI Python Backend
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py            # REST endpoints (health, predict, compare, XAI)
│   │   ├── core/
│   │   │   ├── config.py               # Settings, constants, and paths
│   │   │   └── registry.py             # Model discovery, checksums & availability
│   │   ├── services/
│   │   │   ├── inference_service.py    # Singleton model runner (E0, E1, E2)
│   │   │   ├── comparison_service.py   # Multi-model comparative runner
│   │   │   └── export_service.py       # Result caching & ZIP/CSV/JSON packager
│   │   ├── schemas/
│   │   │   └── models.py               # Pydantic request/response schemas
│   │   ├── utils/
│   │   │   ├── image_adapter.py        # Universal multi-format loader (uint16/uint8)
│   │   │   ├── geo_inspector.py        # Rasterio/PIL metadata extractor (CRS, bounds)
│   │   │   ├── preprocessing.py        # 4-band PlanetScope normalization & 512x512 tiling
│   │   │   ├── visualization.py        # Fast base64 PNG and heatmap encoders
│   │   │   ├── xai.py                  # LayerCAM gradient attribution engine
│   │   │   └── module_loader.py        # Dynamic namespace-isolated Python loader
│   │   └── main.py                     # FastAPI application setup & CORS middleware
│   ├── tests/                          # Pytest suite
│   │   ├── test_model_registry.py      # Checkpoint integrity & discovery tests
│   │   ├── test_api.py                 # REST endpoint integration tests
│   │   ├── test_inference.py           # E0/E1/E2 parity & zero-residual invariance
│   │   └── fixtures/                   # Sample test chips & optical benchmarks
│   ├── requirements.txt
│   └── .env.example
│
├── Models/                             # Research Models & Checkpoints
│   ├── Unet/                           # Standard U-Net baseline code, configs, results
│   ├── UNet-Mamba/                     # UNet-Mamba base model & trained weights
│   │   ├── checkpoints/                # best_model.pth (171 MB), best_e1_model.pth, best_e2_model.pth
│   │   ├── configs/                    # Model training configurations
│   │   ├── src/                        # UNetMamba state-space implementation
│   │   └── CAVE-Flood/                 # CAVE-Flood CASR Extension & RemoteCLIP weights
│   │       ├── checkpoints/            # best_e1_model.pth, best_e2_model.pth, scaler
│   │       ├── weights/                # RemoteCLIP-ViT-B-32.pt (605 MB)
│   │       ├── configs/                # Prompts config & hyperparameters
│   │       ├── results/                # Research evaluation summaries & logs
│   │       └── src/                    # casr.py, candidate_selector.py, XAI scripts
│   └── SegFormer/                      # SegFormer-B2 implementation, LOEO splits, outputs
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔒 Cryptographic Checkpoint Verification Ledger

All model checkpoints have been cryptographically verified against their original research values:

| Model Architecture | Artifact File | File Size | SHA-256 Checksum | Availability Status |
| :--- | :--- | :--- | :--- | :--- |
| **UNet-Mamba (E0)** | `best_model.pth` | 171,815,178 bytes | `58674c84e97e16c7ab86f2e218ba8738c7aaf69a8c6fd8c384de00beec9eb76d` | ✅ **Verified & Ready** |
| **UNet-Mamba + CASR (E1)** | `best_e1_model.pth` | 54,430 bytes | `84a02d1f1fe8371bf799edb0ddce28f69d20dd90b3da503c06eac87f521226e9` | ✅ **Verified & Ready** |
| **CAVE-Flood Full (E2)** | `best_e2_model.pth` | 54,430 bytes | `19347f5f64385c46f16c27b345a7f143e56b51f23c1fe7e887906b6e7c19d2d9` | ✅ **Verified & Ready** |
| **RemoteCLIP Vision-Language** | `RemoteCLIP-ViT-B-32.pt` | 605,208,421 bytes | `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4` | ✅ **Verified & Ready** |
| **RemoteCLIP Scaler** | `remoteclip_score_scaler.json` | 149 bytes | `e46396fd03cd198f2168d8b36c71bca572ba84d1b3f540ef01898b63c9a9a70e` | ✅ **Verified & Ready** |
| **Standard U-Net** | `best_model.pth` | Pending | — | ℹ️ Preserved (Pending Training) |
| **SegFormer-B2** | `best_model.pth` | Pending | — | ℹ️ Preserved (Pending Training) |

---

## 🚀 Quickstart Guide: Running Locally

### 1. Start the FastAPI Backend

```bash
# From repository root:
python -m uvicorn Backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base: `http://127.0.0.1:8000`
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/api/health`

### 2. Start the React Frontend

```bash
# In a separate terminal:
cd Frontend
npm run dev
```
- Web Application: `http://localhost:5173`

---

## 🧪 Running Automated Tests

```bash
# Run backend unit, integration, and model parity tests:
python -m pytest Backend/tests -v
```

---

## 🛰️ Supported Satellite Input Modes

1. **Mode 1 — Native 4-Band Multispectral Inference (Scientific Standard):**
   - GeoTIFF / TIFF rasters with 4 channels: `[Red, Green, Blue, Near-Infrared (NIR)]`.
   - Uses strict dataset normalization: Mean `[2334.81, 2378.43, 2185.12, 3608.97]`, Std `[1245.52, 1184.21, 1148.65, 1420.33]`.
2. **Mode 2 — RGB Satellite Inspection:**
   - Standard optical imagery (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`).
   - Extracts complete raster dimensions, data types, and channel allocations with optional synthetic NIR demonstration inference.
3. **Mode 3 — RGB + NIR Dual Upload:**
   - Paired high-resolution optical satellite RGB imagery with corresponding calibrated NIR band rasters.

---

## 🔬 Mathematical Formulation

CAVE-Flood implements strict residual spatial refinement:

$$\mathbf{Z}_{\text{final}} = \mathbf{Z}_{\text{base}} + \lambda_{\text{res}} \cdot \mathbf{C} \cdot \mathbf{G} \cdot \Delta\mathbf{Z}$$

Where:
- $\mathbf{Z}_{\text{base}}$: Frozen UNet-Mamba baseline logits.
- $\mathbf{C} \in \{0, 1\}$: Binary candidate-region selection mask isolating Top-4 highest entropy/gradient patches in a 4x4 spatial grid.
- $\mathbf{G} \in [0, 1]$: Effective sigmoid spatial gate.
- $\Delta\mathbf{Z}$: CASR residual correction logits.
- **Invariance Property:** For all pixels outside candidate regions ($\mathbf{C} = 0$), $\mathbf{Z}_{\text{final}} \equiv \mathbf{Z}_{\text{base}}$ with zero residual leakage.
