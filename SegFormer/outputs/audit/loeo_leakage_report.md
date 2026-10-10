# Leave-One-Event-Out (LOEO) Cross-Validation Leakage Audit Report

## 1. Summary of Invariant Checks
- Total Outer Folds: 19
- Distinct Events Evaluated: 19 / 19
- Leakage Violations Detected: 0
- Duplicate Chips Across Events: 0
- Leakage Gate Status: **PASSED**

## 2. Mathematical Partition Proofs

For every fold $i \in \{0, 1, \dots, 18\}$:
$$\text{Train Events} \cap \text{Val Events} = \emptyset$$
$$\text{Train Events} \cap \text{Test Events} = \emptyset$$
$$\text{Val Events} \cap \text{Test Events} = \emptyset$$
$$\text{Train Events} \cup \text{Val Events} \cup \text{Test Events} = \mathcal{E}_{\text{all}}$$

## 3. Data Normalization & Hyperparameter Isolation
1. **Data Normalization**: Fixed ImageNet constants $(\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225])$ are applied. No normalization statistics are computed from or fitted on the held-out test event.
2. **Checkpoint Selection**: Epoch-level checkpoint selection uses exclusively inner validation event metrics (`val_flood_iou` on 3 disjoint events). The held-out test event is evaluated strictly ONCE after training concludes.
3. **Spatial Tiling Containment**: Patches strictly inherit their parent chip's event identifier. No patch crosses geographic event boundaries.

## 4. Fold Manifest Summary Table

| Fold ID | Held-Out Test Event | Train Events | Val Events | Test Chips | Train Chips | Val Chips |
|---|---|---|---|---|---|---|
| fold_00 | **Bangladesh** | 15 events | 3 events | 15 | 308 | 43 |
| fold_01 | **Bolivia** | 15 events | 3 events | 100 | 212 | 54 |
| fold_02 | **Cambodia** | 15 events | 3 events | 20 | 296 | 50 |
| fold_03 | **Colombia** | 15 events | 3 events | 12 | 313 | 41 |
| fold_04 | **Ghana** | 15 events | 3 events | 20 | 321 | 25 |
| fold_05 | **Nepal** | 15 events | 3 events | 9 | 302 | 55 |
| fold_06 | **Nigeria** | 15 events | 3 events | 15 | 300 | 51 |
| fold_07 | **Paraguay** | 15 events | 3 events | 18 | 316 | 32 |
| fold_08 | **Somalia** | 15 events | 3 events | 20 | 304 | 42 |
| fold_09 | **Spain** | 15 events | 3 events | 20 | 222 | 124 |
| fold_10 | **US-Alabama** | 15 events | 3 events | 16 | 310 | 40 |
| fold_11 | **US-Arkansas** | 15 events | 3 events | 4 | 234 | 128 |
| fold_12 | **US-Carolina** | 15 events | 3 events | 15 | 326 | 25 |
| fold_13 | **US-Dakota** | 15 events | 3 events | 14 | 316 | 36 |
| fold_14 | **US-Kansas** | 15 events | 3 events | 20 | 302 | 44 |
| fold_15 | **US-Nebraska** | 15 events | 3 events | 16 | 313 | 37 |
| fold_16 | **US-Oklahoma** | 15 events | 3 events | 11 | 326 | 29 |
| fold_17 | **US-Texas** | 15 events | 3 events | 9 | 310 | 47 |
| fold_18 | **Uzbekistan** | 15 events | 3 events | 12 | 298 | 56 |
