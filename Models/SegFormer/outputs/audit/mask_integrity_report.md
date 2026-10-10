# Mask Pixel Value Integrity & Class Mapping Verification Report

## 1. Executive Summary
- Total Masks Audited: 366
- Unique Raw Label Values Across Entire Dataset: [np.int64(1), np.int64(2)]
- Unique Processed Target Values: [np.int64(0), np.int64(1)]
- Pixel Conservation Mismatches: 0
- Mask Integrity Gate Status: **PASSED**

## 2. Verified Class Mapping Policy

| Raw Value | Target Class | Semantic Category | Pixel Count | Dataset Percentage |
|---|---|---|---|---|
| Raw 1 | Class 0 | Non-Flood / Land / Background | 209,156,698 | 54.50% |
| Raw 2 | Class 1 | Flood / Inundation | 174,622,118 | 45.50% |
| Raw 0 | Class 255 | Ignore / Nodata / Invalid | 0 | 0.00% |

## 3. Strict Invariants Verified
1. **Zero Raw 0 to Background conversion**: Raw nodata (0) pixels are strictly mapped to `ignore_index = 255` and never to background (0).
2. **Exact Pixel Conservation**: Every single flood pixel (raw 2) maps to target 1; every non-flood pixel (raw 1) maps to target 0.
3. **Nearest-Neighbor Interpolation**: Semantic masks are downscaled or cropped strictly with nearest-neighbor indexing, preventing fractional intermediate logits.
4. **Data Type Harmonization**: Both uint8 and float32 masks (e.g. US-Arkansas, Uzbekistan) are safely rounded to int64 before class assignment.
