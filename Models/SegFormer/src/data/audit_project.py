"""Project Audit and Inventory Generator.

Scans the repository, classifies every file into Categories A-J,
and generates:
- outputs/audit/project_inventory.csv
- outputs/audit/project_structure_before.txt
- outputs/audit/project_audit_report.md
"""

import os
import pandas as pd
from typing import Dict, Any, List

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def categorize_file(rel_path: str, filename: str) -> Dict[str, Any]:
    norm_path = rel_path.replace("\\", "/")
    
    # Check ignore/virtualenv
    if norm_path.startswith(".venv/") or norm_path.startswith(".git/"):
        return {
            "category": "G",
            "category_name": "Virtualenv / Version Control Internal",
            "referenced_by_code": True,
            "required_for_training": True,
            "recommended_action": "Retain",
            "reason": "Execution environment or git directory",
        }
    
    # Dataset files
    if norm_path.startswith("Datasets/"):
        if filename == ".DS_Store":
            return {
                "category": "G",
                "category_name": "Temporary or redundant files",
                "referenced_by_code": False,
                "required_for_training": False,
                "recommended_action": "Remove",
                "reason": "macOS metadata artifact from archive extraction",
            }
        elif filename.endswith(".tif.aux.xml"):
            return {
                "category": "D",
                "category_name": "Essential reproducibility metadata",
                "referenced_by_code": True,
                "required_for_training": True,
                "recommended_action": "Retain",
                "reason": "GDAL raster geospatial metadata auxiliary file",
            }
        elif filename.endswith(".tif"):
            return {
                "category": "B",
                "category_name": "Essential dataset files",
                "referenced_by_code": True,
                "required_for_training": True,
                "recommended_action": "Retain",
                "reason": "Core multi-sensor satellite imagery or flood ground truth mask",
            }
        else:
            return {
                "category": "J",
                "category_name": "Unused code requiring manual review",
                "referenced_by_code": False,
                "required_for_training": False,
                "recommended_action": "Review",
                "reason": "Unrecognized dataset file",
            }

    # Python source code
    if norm_path.startswith("src/"):
        return {
            "category": "A",
            "category_name": "Essential source code",
            "referenced_by_code": True,
            "required_for_training": True,
            "recommended_action": "Retain",
            "reason": "Core modular pipeline source code",
        }

    # Python scripts
    if norm_path.startswith("scripts/"):
        return {
            "category": "A",
            "category_name": "Essential source code",
            "referenced_by_code": True,
            "required_for_training": True,
            "recommended_action": "Retain",
            "reason": "CLI entrypoints for inspection, testing, and training",
        }

    # Tests
    if norm_path.startswith("tests/"):
        return {
            "category": "A",
            "category_name": "Essential source code",
            "referenced_by_code": True,
            "required_for_training": False,
            "recommended_action": "Retain",
            "reason": "Automated verification test suite",
        }

    # Configurations
    if norm_path.startswith("configs/"):
        return {
            "category": "C",
            "category_name": "Essential configurations",
            "referenced_by_code": True,
            "required_for_training": True,
            "recommended_action": "Retain",
            "reason": "Experiment hyperparameters and environment configurations",
        }

    # Outputs
    if norm_path.startswith("outputs/"):
        if norm_path.startswith("outputs/checkpoints/"):
            return {
                "category": "E",
                "category_name": "Useful experiment outputs",
                "referenced_by_code": True,
                "required_for_training": False,
                "recommended_action": "Retain",
                "reason": "Trained SegFormer weights and resume checkpoints",
            }
        elif norm_path.startswith("outputs/splits/"):
            return {
                "category": "D",
                "category_name": "Essential reproducibility metadata",
                "referenced_by_code": True,
                "required_for_training": True,
                "recommended_action": "Retain",
                "reason": "Verified zero-leakage 19-fold LOEO cross-validation manifests",
            }
        elif norm_path.startswith("outputs/results/") or norm_path.startswith("outputs/visualizations/"):
            return {
                "category": "E",
                "category_name": "Useful experiment outputs",
                "referenced_by_code": False,
                "required_for_training": False,
                "recommended_action": "Retain",
                "reason": "Verified evaluation metrics and qualitative visualization figures",
            }
        elif norm_path.endswith(".csv") or norm_path.endswith(".json"):
            return {
                "category": "F",
                "category_name": "Generated files that can be regenerated",
                "referenced_by_code": True,
                "required_for_training": True,
                "recommended_action": "Retain",
                "reason": "Audited dataset manifest and statistics",
            }
        else:
            return {
                "category": "F",
                "category_name": "Generated files that can be regenerated",
                "referenced_by_code": False,
                "required_for_training": False,
                "recommended_action": "Retain",
                "reason": "Audit or visualization output",
            }

    # Project metadata root files
    if filename in ["README.md", "requirements.txt", "pyproject.toml", ".gitignore"]:
        return {
            "category": "D",
            "category_name": "Essential reproducibility metadata",
            "referenced_by_code": True,
            "required_for_training": True,
            "recommended_action": "Retain",
            "reason": "Root project configuration, packaging, and documentation",
        }

    # Temporary build caches
    if "__pycache__" in norm_path or filename.endswith(".pyc") or filename.endswith(".pyo") or ".pytest_cache" in norm_path:
        return {
            "category": "G",
            "category_name": "Temporary or redundant files",
            "referenced_by_code": False,
            "required_for_training": False,
            "recommended_action": "Remove",
            "reason": "Python or pytest compiled bytecode cache",
        }

    if ".egg-info" in norm_path:
        return {
            "category": "G",
            "category_name": "Temporary or redundant files",
            "referenced_by_code": False,
            "required_for_training": False,
            "recommended_action": "Retain",
            "reason": "Editable pip package metadata",
        }

    return {
        "category": "J",
        "category_name": "Unused code requiring manual review",
        "referenced_by_code": False,
        "required_for_training": False,
        "recommended_action": "Review",
        "reason": "Unclassified workspace file",
    }


def run_audit(output_dir: str = "outputs/audit"):
    os.makedirs(output_dir, exist_ok=True)
    inventory = []
    tree_lines = []

    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        # Exclude .venv from tree lines to prevent 30,000+ line files
        rel_root = os.path.relpath(root, WORKSPACE_ROOT)
        if rel_root.startswith(".venv") or rel_root.startswith(".git"):
            continue

        depth = rel_root.count(os.sep) if rel_root != "." else 0
        indent = "  " * depth
        folder_name = os.path.basename(root) if rel_root != "." else "SegFormer"
        tree_lines.append(f"{indent}[DIR] {folder_name}/")

        for f in sorted(files):
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, WORKSPACE_ROOT)
            file_size = os.path.getsize(full_path)
            _, ext = os.path.splitext(f)
            
            cat_info = categorize_file(rel_path, f)
            inventory.append({
                "relative_path": rel_path.replace("\\", "/"),
                "filename": f,
                "file_type": ext if ext else "no_ext",
                "size_bytes": file_size,
                "category": cat_info["category"],
                "category_name": cat_info["category_name"],
                "referenced_by_code": cat_info["referenced_by_code"],
                "required_for_training": cat_info["required_for_training"],
                "recommended_action": cat_info["recommended_action"],
                "reason": cat_info["reason"],
            })
            
            # Keep tree readable: summarize datasets folders
            if not rel_root.startswith("Datasets") or depth <= 2:
                tree_lines.append(f"{indent}  {f} ({file_size:,} bytes)")

    # Save project structure before
    struct_before_path = os.path.join(output_dir, "project_structure_before.txt")
    with open(struct_before_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))
    print(f"Saved structure before to {struct_before_path}")

    # Save inventory CSV
    inventory_df = pd.DataFrame(inventory)
    csv_path = os.path.join(output_dir, "project_inventory.csv")
    inventory_df.to_csv(csv_path, index=False)
    print(f"Saved inventory to {csv_path} ({len(inventory_df)} files)")

    # Summary by category
    summary = inventory_df.groupby("category").agg(
        file_count=("filename", "count"),
        total_size_mb=("size_bytes", lambda s: round(s.sum() / (1024 * 1024), 2)),
    ).reset_index()

    # Generate Markdown Report
    report_path = os.path.join(output_dir, "project_audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Project Audit & File Inventory Report\n\n")
        f.write(f"- Total Files Audited: {len(inventory_df)}\n")
        f.write(f"- Total Storage Footprint: {inventory_df['size_bytes'].sum() / (1024*1024):.2f} MB\n\n")
        f.write("## Category Breakdown\n\n")
        f.write("| Category | Description | File Count | Size (MB) |\n")
        f.write("|---|---|---|---|\n")
        cat_desc = {
            "A": "Essential source code",
            "B": "Essential dataset files",
            "C": "Essential configurations",
            "D": "Essential reproducibility metadata",
            "E": "Useful experiment outputs",
            "F": "Generated files that can be regenerated",
            "G": "Temporary or redundant files",
            "H": "Duplicates requiring investigation",
            "I": "Potentially dangerous or corrupted files",
            "J": "Unused code requiring manual review",
        }
        for _, row in summary.iterrows():
            c = row["category"]
            f.write(f"| {c} | {cat_desc.get(c, 'Other')} | {row['file_count']} | {row['total_size_mb']} MB |\n")
        
        f.write("\n## Safe Cleanup Candidates\n\n")
        cleanup_candidates = inventory_df[inventory_df["recommended_action"] == "Remove"]
        f.write(f"Identified {len(cleanup_candidates)} candidates for safe removal:\n\n")
        for _, r in cleanup_candidates.head(25).iterrows():
            f.write(f"- `{r['relative_path']}`: {r['reason']}\n")
        if len(cleanup_candidates) > 25:
            f.write(f"- ... and {len(cleanup_candidates) - 25} more (see project_inventory.csv)\n")

    print(f"Saved audit report to {report_path}")


if __name__ == "__main__":
    run_audit()
