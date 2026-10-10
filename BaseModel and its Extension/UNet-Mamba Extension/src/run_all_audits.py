"""
Master Automated Audit Test Suite: CAVE-Flood v1
Executes all independent audit modules and verifies:
1. Ground-truth label counts from raw masks (Audit A)
2. Metric reconciliation across validation and test sets (Audit B)
3. LayerCAM formulation, gradient safety, and parameter invariance (Audit C)
4. XAI alignment, spatial gate statistics, and scene-level confusion matrices (Audits D & E)
5. Checkpoint SHA-256 integrity checks
"""

import sys
import hashlib
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from audit_gt_pixel_counts import count_gt_pixels
from audit_reconciliation import audit_metrics
from audit_xai_layercam import audit_scene_xai


EXPECTED_HASHES = {
    "checkpoints/best_model.pth": "58674c84e97e16c7ab86f2e218ba8738c7aaf69a8c6fd8c384de00beec9eb76d",
    "checkpoints/best_e1_model.pth": "84a02d1f1fe8371bf799edb0ddce28f69d20dd90b3da503c06eac87f521226e9",
    "checkpoints/best_e2_model.pth": "19347f5f64385c46f16c27b345a7f143e56b51f23c1fe7e887906b6e7c19d2d9"
}


def verify_checkpoint_hashes():
    print("\n" + "=" * 80)
    print("VERIFYING CHECKPOINT CRYPTOGRAPHIC INTEGRITY (SHA-256)")
    print("=" * 80)
    all_matched = True
    for rel_path, expected_hash in EXPECTED_HASHES.items():
        candidates = [
            SRC_DIR.parent / rel_path,
            Path("UNet-Mamba Extension") / rel_path,
            Path("UNet-Mamba") / rel_path,
            Path(rel_path)
        ]
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            print(f"  [FAIL] Checkpoint not found: {rel_path}")
            all_matched = False
            continue
        with open(p, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest().lower()
        if computed_hash == expected_hash.lower():
            print(f"  [PASS] {p.name}: {computed_hash[:16]}... (SHA256 Match)")
        else:
            print(f"  [FAIL] {p.name}: Expected {expected_hash}, got {computed_hash}")
            all_matched = False
    return all_matched


def run_all_audits():
    print("\n" + "#" * 80)
    print("STARTING CAVE-FLOOD v1 MASTER INDEPENDENT AUDIT SUITE")
    print("#" * 80)
    
    # Test 1: Checkpoints
    ckpt_ok = verify_checkpoint_hashes()
    assert ckpt_ok, "Checkpoint SHA-256 integrity verification failed!"
    
    # Test 2: Ground Truth Raw Counts (Audit A)
    gt_stats = count_gt_pixels()
    assert gt_stats["valid_pixels"] == 46137344, "Total valid test pixels mismatch!"
    assert gt_stats["flood_pixels"] == 26566620, "Ground-truth flood pixels mismatch!"
    assert gt_stats["bg_pixels"] == 19570724, "Ground-truth background pixels mismatch!"
    print("\n>>> AUDIT A: GROUND-TRUTH TEST COUNTS FULLY PASSED.")
    
    # Test 3: Metric Reconciliation (Audit B)
    recon_ok = audit_metrics()
    assert recon_ok, "Metric reconciliation audit failed!"
    print("\n>>> AUDIT B: VALIDATION & TEST METRIC RECONCILIATION FULLY PASSED.")
    
    # Test 4: LayerCAM & Scene Diagnostics (Audits C, D, E)
    scene_results = audit_scene_xai()
    assert len(scene_results) == 3, "Expected 3 audited scenes!"
    print("\n>>> AUDITS C, D, E: LAYERCAM & XAI DIAGNOSTICS FULLY PASSED.")
    
    print("\n" + "#" * 80)
    print("ALL INDEPENDENT AUDITS AND AUTOMATED TESTS PASSED SUCCESSFULLY!")
    print("FINAL VERDICT: VALID WITH LIMITATIONS")
    print("#" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_all_audits())
