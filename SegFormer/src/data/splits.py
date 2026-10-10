"""Leave-One-Event-Out (LOEO) Cross-Validation Split Generator.

Implements strict event-disjoint splitting:
- Each flood event is held out as the test set exactly once.
- The remaining events form the training candidate pool.
- A deterministic inner subset (~15-20%, 3 events) is chosen for validation.
- The remaining events form the training set.

Guarantees:
  Train events ∩ Val events = ∅
  Train events ∩ Test events = ∅
  Val events ∩ Test events = ∅
"""

import json
import os
import random
from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("LOEOSplits")


def generate_loeo_splits(
    dataset_manifest_path: str = "outputs/dataset_manifest.csv",
    splits_output_dir: str = "outputs/splits",
    val_ratio: float = 0.17,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate and persist strict event-disjoint LOEO splits.
    
    Args:
        dataset_manifest_path: Path to dataset manifest CSV generated during inspection.
        splits_output_dir: Directory to save split JSON files.
        val_ratio: Fraction of non-test candidate events to allocate for inner validation (~0.17 for 3 of 18).
        seed: Random seed for deterministic inner validation event selection.
        
    Returns:
        List of fold specification dictionaries.
    """
    if not os.path.exists(dataset_manifest_path):
        raise FileNotFoundError(f"Dataset manifest not found at: {dataset_manifest_path}. Run inspection first.")

    os.makedirs(splits_output_dir, exist_ok=True)
    manifest_df = pd.read_csv(dataset_manifest_path)
    
    events = sorted(manifest_df["event"].unique().tolist())
    num_events = len(events)
    logger.info(f"Generating LOEO cross-validation splits for {num_events} events: {events}")

    folds = []
    
    # Calculate chip count per event for tracking
    event_chip_counts = manifest_df["event"].value_counts().to_dict()

    for fold_idx, test_event in enumerate(events):
        fold_id = f"fold_{fold_idx:02d}"
        candidate_events = [e for e in events if e != test_event]
        
        # Deterministic inner validation selection using fixed seeded RNG
        rng = random.Random(seed + fold_idx * 100)
        num_val_events = max(1, round(len(candidate_events) * val_ratio))
        
        # Deterministically sample val_events from sorted candidates
        shuffled_candidates = list(candidate_events)
        rng.shuffle(shuffled_candidates)
        val_events = sorted(shuffled_candidates[:num_val_events])
        train_events = sorted(shuffled_candidates[num_val_events:])

        # Strict leakage assertions
        train_set = set(train_events)
        val_set = set(val_events)
        test_set = {test_event}

        assert not (train_set & val_set), f"Leakage detected in {fold_id}: Train and Val overlap!"
        assert not (train_set & test_set), f"Leakage detected in {fold_id}: Train and Test overlap!"
        assert not (val_set & test_set), f"Leakage detected in {fold_id}: Val and Test overlap!"
        assert train_set | val_set | test_set == set(events), f"Event mismatch in {fold_id}!"

        # Count chips
        train_chips = sum(event_chip_counts[e] for e in train_events)
        val_chips = sum(event_chip_counts[e] for e in val_events)
        test_chips = event_chip_counts[test_event]

        fold_data = {
            "fold_id": fold_id,
            "fold_index": fold_idx,
            "test_event": test_event,
            "train_events": train_events,
            "val_events": val_events,
            "num_train_events": len(train_events),
            "num_val_events": len(val_events),
            "num_test_events": 1,
            "train_chips_count": train_chips,
            "val_chips_count": val_chips,
            "test_chips_count": test_chips,
            "total_chips_count": train_chips + val_chips + test_chips,
            "random_seed": seed,
        }

        # Save individual fold JSON
        fold_file = os.path.join(splits_output_dir, f"{fold_id}.json")
        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(fold_data, f, indent=2)

        folds.append(fold_data)

    # Save overall master LOEO splits JSON
    master_splits_file = os.path.join(splits_output_dir, "loeo_splits.json")
    with open(master_splits_file, "w", encoding="utf-8") as f:
        json.dump({
            "num_folds": len(folds),
            "events": events,
            "folds": folds,
        }, f, indent=2)

    logger.info(f"Successfully generated and validated {len(folds)} LOEO splits in {splits_output_dir}")
    return folds


if __name__ == "__main__":
    generate_loeo_splits()
