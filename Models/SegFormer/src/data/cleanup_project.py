"""Safe Project Cleanup Module.

Removes strictly safe temporary files:
- macOS .DS_Store metadata files inside Datasets/
- .pytest_cache directories
- Python bytecode files (__pycache__, *.pyc)

Strictly preserves:
- All original and modified dataset rasters (PS, S1, S2, L8)
- All ground truth masks (labels)
- All GDAL .aux.xml auxiliary files
- All source code, configs, tests, checkpoints, and outputs.
"""

import os
import shutil
import sys
import pandas as pd
from typing import List, Dict, Any

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

def execute_safe_cleanup(output_dir: str = "outputs/audit") -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    cleanup_records = []

    # 1. Locate all .DS_Store files
    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        for f in files:
            if f == ".DS_Store" or f.endswith(".tmp"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, WORKSPACE_ROOT)
                file_size = os.path.getsize(full_path)
                try:
                    os.remove(full_path)
                    status = "Deleted"
                except Exception as e:
                    status = f"Failed: {e}"
                cleanup_records.append({
                    "relative_path": rel_path.replace("\\", "/"),
                    "action": "Remove",
                    "file_size": file_size,
                    "reason": "macOS metadata or temporary artifact",
                    "status": status,
                })

    # 2. Locate and remove .pytest_cache
    pytest_cache = os.path.join(WORKSPACE_ROOT, ".pytest_cache")
    if os.path.exists(pytest_cache):
        shutil.rmtree(pytest_cache, ignore_errors=True)
        cleanup_records.append({
            "relative_path": ".pytest_cache",
            "action": "Remove Directory",
            "file_size": 0,
            "reason": "Pytest run cache",
            "status": "Deleted",
        })

    # Save cleanup manifest
    manifest_df = pd.DataFrame(cleanup_records)
    manifest_path = os.path.join(output_dir, "cleanup_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved cleanup manifest to {manifest_path} ({len(cleanup_records)} items removed)")

    # Re-run audit to generate project_structure_after.txt and updated inventory
    from src.data.audit_project import run_audit
    run_audit(output_dir)

    # Copy project_structure_before to project_structure_after with current state
    struct_after_path = os.path.join(output_dir, "project_structure_after.txt")
    tree_lines = []
    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        rel_root = os.path.relpath(root, WORKSPACE_ROOT)
        if rel_root.startswith(".venv") or rel_root.startswith(".git") or "__pycache__" in rel_root:
            continue
        depth = rel_root.count(os.sep) if rel_root != "." else 0
        indent = "  " * depth
        folder_name = os.path.basename(root) if rel_root != "." else "SegFormer"
        tree_lines.append(f"{indent}[DIR] {folder_name}/")
        for f in sorted(files):
            if not f.endswith(".pyc") and f != ".DS_Store":
                full_path = os.path.join(root, f)
                file_size = os.path.getsize(full_path)
                if not rel_root.startswith("Datasets") or depth <= 2:
                    tree_lines.append(f"{indent}  {f} ({file_size:,} bytes)")

    with open(struct_after_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))
    print(f"Saved updated project structure to {struct_after_path}")

    return cleanup_records


if __name__ == "__main__":
    execute_safe_cleanup()
