# Dataset Integrity & Multi-Sensor Audit Report

## 1. Summary Statistics
- Total Flood Events: 19
- Total PlanetScope (Modified 8-bit) Chips: 366
- Total Paired Ground Truth Masks: 366
- Missing Image-Mask Pairs: 0
- Corrupted Files: 0
- Original Coincident Observations Audited: 880 (S1, S2, L8)

## 2. Sensor Comparison & Characteristics

| Sensor | Product Type | Spatial Resolution | Dimensions | Bands | Dtype | Data Range |
|---|---|---|---|---|---|---|
| PlanetScope (PS) | Modified Display 8-bit | 3 meters | 1024 x 1024 | 4 (B, G, R, NIR) | uint16 | [0, 255] |
| Sentinel-1 (S1) | Original SAR GRD (dB) | 10 meters | ~334 x 334 | 2 (VV, VH) | float32 | [-50, +25] dB |
| Sentinel-2 (S2) | Original Optical L2A | 10–20 meters | ~334 x 334 | 10 multispectral | float32 | [0, 4000] |
| Landsat-8 (L8) | Original Optical Surface Refl | 30 meters | ~111 x 111 | 7 multispectral | float32 | [0, 30000] |
| Labels (Masks) | Ground Truth Inundation | 3 meters | 1024 x 1024 | 1 (Integer Mask) | uint8 / float32 | {1: Land, 2: Water} |

## 3. Pairing Verification
- Exactly 366 / 366 PlanetScope chips have matching 1024x1024 ground truth masks.
- Zero shape mismatches detected across all 19 flood events.
- Zero NaN or infinite values detected in PlanetScope optical rasters.
