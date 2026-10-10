import os
import glob
import pandas as pd
from pathlib import Path

from config import (
    load_config,
    TRAIN_EVENTS,
    VAL_EVENTS,
    TEST_EVENTS,
    ALL_19_EVENTS,
    BASE_DIR
)
from utils import get_logger

def generate_split_manifests(config=None, manifests_dir=None):
    if config is None:
        config = load_config()
    logger = get_logger("SplitManager")
    
    raw_dataset_dir = Path(config['data']['raw_dataset_dir'])
    if not raw_dataset_dir.is_absolute():
        raw_dataset_dir = BASE_DIR / raw_dataset_dir
        
    sensor = config['data']['sensor']
    
    if manifests_dir is None:
        manifests_dir = BASE_DIR / config['data']['manifests_dir']
    else:
        manifests_dir = Path(manifests_dir)
        
    os.makedirs(manifests_dir, exist_ok=True)
    
    # Assert disjointness of event splits
    set_train = set(TRAIN_EVENTS)
    set_val = set(VAL_EVENTS)
    set_test = set(TEST_EVENTS)
    
    assert set_train.isdisjoint(set_val), "Data leakage! Train and Val events overlap!"
    assert set_train.isdisjoint(set_test), "Data leakage! Train and Test events overlap!"
    assert set_val.isdisjoint(set_test), "Data leakage! Val and Test events overlap!"
    
    all_specified = sorted(list(set_train | set_val | set_test))
    assert all_specified == ALL_19_EVENTS, f"Mismatch in event list! Expected {ALL_19_EVENTS}, got {all_specified}"
    
    records = {"train": [], "val": [], "test": []}
    
    event_split_map = {}
    for evt in TRAIN_EVENTS:
        event_split_map[evt] = "train"
    for evt in VAL_EVENTS:
        event_split_map[evt] = "val"
    for evt in TEST_EVENTS:
        event_split_map[evt] = "test"
        
    for event_name in ALL_19_EVENTS:
        event_dir = raw_dataset_dir / event_name
        split_name = event_split_map[event_name]
        
        ps_dir = event_dir / sensor
        mask_dir = event_dir / "labels"
        if not mask_dir.exists():
            mask_dir = event_dir / "Masks"
        
        ps_files = sorted(glob.glob(str(ps_dir / "*.tif"))) + sorted(glob.glob(str(ps_dir / "*.png")))
        mask_files = sorted(glob.glob(str(mask_dir / "*.tif"))) + sorted(glob.glob(str(mask_dir / "*.png")))
        
        ps_map = {Path(p).stem: p for p in ps_files}
        mask_map = {Path(m).stem: m for m in mask_files}
        
        matched_stems = sorted(list(set(ps_map.keys()).intersection(set(mask_map.keys()))))
        
        for stem in matched_stems:
            records[split_name].append({
                "event": event_name,
                "stem": stem,
                "img_path": ps_map[stem],
                "mask_path": mask_map[stem],
                "sensor": sensor,
                "width": 1024,
                "height": 1024,
                "split": split_name
            })
            
    # Save CSVs and verify counts
    for split_name in ["train", "val", "test"]:
        df_split = pd.DataFrame(records[split_name])
        csv_path = manifests_dir / f"{split_name}.csv"
        df_split.to_csv(csv_path, index=False)
        logger.info(f"Saved {split_name} manifest with {len(df_split)} scenes to {csv_path}")
        
    logger.info("Fixed event-level split verification complete! No event leakage detected.")

if __name__ == "__main__":
    generate_split_manifests()
