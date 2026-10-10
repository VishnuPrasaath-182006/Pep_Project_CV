---
title: CAVE-Flood — Satellite Flood Inundation Segmentation Demo
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
---

# CAVE-Flood: Confidence-Aware Vision-Language-Guided Residual Refinement

> 🌐 **Public Live Demonstration Website**: [https://3f615832804328bf5d.gradio.live/](https://3f615832804328bf5d.gradio.live/)
> 📦 **Hugging Face Model Weights**: [https://huggingface.co/VP182006/CAVE-Flood-Weights](https://huggingface.co/VP182006/CAVE-Flood-Weights)
> 🤗 **Hugging Face Space**: [https://huggingface.co/spaces/VP182006/CAVE-Flood](https://huggingface.co/spaces/VP182006/CAVE-Flood)

A research demonstration web application for high-resolution ($3\text{m}$) multispectral satellite flood extent mapping using PlanetScope 4-band optical imagery.

CAVE-Flood introduces parameter-efficient residual adaptation to satellite disaster response:
1. **Foundation Backbone**: Frozen 14.29M-parameter **UNet-Mamba** 2D selective scan model.
2. **Residual Adapter**: 12,035-parameter **Confidence-Aware Semantic Residual (CASR)** adapter head ($0.084\%$ parameter overhead).
3. **Selective Uncertainty Gating**: Isolates the top 25% most uncertain boundary tiles, while strictly enforcing zero-residual invariance ($\Delta Z \equiv 0$) across 75% of confident interior pixels.
4. **Vision-Language Guidance**: Semantic margin guidance derived from frozen **RemoteCLIP** (ViT-B/32) text-image embeddings.

---

## 1. Features

- **Multispectral GeoTIFF Upload**: Native ingestion of 4-band PlanetScope satellite imagery (Blue, Green, Red, Near-Infrared).
- **True-Color RGB Contrast Preview**: Dynamic 2%–98% percentile optical stretch for terrain inspection.
- **Three Experimental Paradigms**:
  - `E0`: Original UNet-Mamba Baseline
  - `E1`: CASR Residual Refinement (Neutral Guidance)
  - `E2`: CASR + RemoteCLIP Vision-Language Guidance
- **Interactive Visualizations**: High-contrast binary flood masks, semi-transparent satellite overlays, and multi-model comparative transition panels.
- **Explainable AI (XAI)**:
  - Predictive Entropy (Uncertainty)
  - Top 25% Candidate Selection Mask
  - Dynamic CASR Gate Activations
  - RemoteCLIP Guidance Margins
  - Residual Probability Delta ($\Delta P$)
  - **LayerCAM Attribution Heatmap** on CASR residual block2.
- **Optional Ground-Truth Verification**: Physical pixel confusion matrix (TP, FP, FN, TN), IoU, Dice/F1, Precision, and Recall.
- **Geospatial Export**: Download predicted flood masks as georeferenced GeoTIFFs preserving Coordinate Reference System (CRS) and affine transform.

---

## 2. Local Execution

To run the demonstration locally:

```bash
cd "Gradio website"

# Install dependencies
pip install -r requirements.txt

# Run verification tests
python -B tests/test_deployment.py

# Launch Gradio server
python app.py
```
Open your browser at `http://localhost:7860`.

---

## 3. Deploying to Hugging Face Spaces

### Method A: Direct Git Push to Hugging Face Space

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) selecting **Gradio** as the SDK.
2. Clone your newly created Space repository:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   ```
3. Copy all contents of this `Gradio website/` directory into your Space folder:
   ```bash
   # Copy app.py, requirements.txt, README.md, deployment_config.py, preprocessing.py, image_adapter.py, inference.py, visualization.py, xai_utils.py, src/, assets/, checkpoints/, examples/
   ```
4. Push to Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy CAVE-Flood demonstration app"
   git push origin main
   ```

### Method B: Model Weights Storage on Hugging Face Hub

The base foundation model `best_model.pth` is 163.86 MB. On Hugging Face Spaces, you can either:
- Store `best_model.pth` via Git LFS in the Space repository, or
- Download it dynamically on first startup from a model repository on Hugging Face Hub:
  ```python
  from huggingface_hub import hf_hub_download
  ckpt = hf_hub_download("VishnuPrasaath-182006/CAVE-Flood-Weights", "best_model.pth")
  ```

---

## 4. Hardware Requirements & CPU/GPU Compatibility

- **Pure PyTorch Selective Scan**: The UNet-Mamba architecture uses a checkpointed chunked scan operator implemented in pure PyTorch (no custom Triton or CUDA-only C++ compilation required).
- **Execution Compatibility**: Runs on standard free CPU Space environments (takes $\sim 6\text{--}8\,\text{s}$ per 512×512 patch) as well as T4/A10G GPU instances for near-instant inference ($< 0.5\,\text{s}$).

---

## 5. Citation & Attribution

If you use CAVE-Flood in your research or project:

```bibtex
@article{caveflood2026,
  title={CAVE-Flood: Confidence-Aware Vision-Language-Guided Residual Refinement for Flood Inundation Segmentation},
  author={Vishnu Prasaath and Collaborators},
  year={2026}
}
```
