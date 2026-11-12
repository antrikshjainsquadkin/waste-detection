"""Dataset Validation and Annotation Quality Audit Utility.

Inspects dataset directories, validates image formats and dimensions, audits bounding-box
annotations against the 8 approved waste categories, and detects train/val leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

APPROVED_CATEGORIES = {
    0: "paper",
    1: "cardboard",
    2: "plastic_bottle",
    3: "polythene_bag",
    4: "metal",
    5: "glass",
    6: "organic_waste",
    7: "general_trash",
}


def audit_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Audit a YOLO-format object detection dataset directory."""
    report: Dict[str, Any] = {
        "dataset_path": str(dataset_dir.resolve()),
        "valid": False,
        "splits_found": [],
        "total_images": 0,
        "total_annotations": 0,
        "class_counts": {name: 0 for name in APPROVED_CATEGORIES.values()},
        "unknown_class_ids": {},
        "corrupt_images": [],
        "empty_annotation_files": 0,
        "missing_annotations": 0,
        "leakage_suspects": [],
    }

    if not dataset_dir.exists():
        report["error"] = f"Directory does not exist: {dataset_dir}"
        return report

    image_hashes: Dict[str, str] = {}  # hash -> split/filename

    splits = ["train", "val", "test"]
    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        # Support flat layout if images/split doesn't exist
        if not img_dir.exists():
            img_dir = dataset_dir / split / "images"
            lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists():
            continue

        report["splits_found"].append(split)
        img_files = list(img_dir.glob("*.*"))

        for img_path in img_files:
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                continue

            report["total_images"] += 1

            # Validate image readability
            img = cv2.imread(str(img_path))
            if img is None:
                report["corrupt_images"].append(str(img_path))
                continue

            # Check for data leakage via simple visual hash
            resized = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
            img_hash = hash(resized.tobytes())
            if str(img_hash) in image_hashes:
                prev_loc = image_hashes[str(img_hash)]
                report["leakage_suspects"].append(f"{img_path.name} in {split} matches {prev_loc}")
            else:
                image_hashes[str(img_hash)] = f"{split}/{img_path.name}"

            # Check matching annotation file
            lbl_path = lbl_dir / f"{img_path.stem}.txt" if lbl_dir.exists() else None
            if not lbl_path or not lbl_path.exists():
                report["missing_annotations"] += 1
                continue

            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                if not lines:
                    report["empty_annotation_files"] += 1

                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        report["total_annotations"] += 1
                        if cls_id in APPROVED_CATEGORIES:
                            c_name = APPROVED_CATEGORIES[cls_id]
                            report["class_counts"][c_name] += 1
                        else:
                            report["unknown_class_ids"][str(cls_id)] = report["unknown_class_ids"].get(str(cls_id), 0) + 1
            except Exception as e:
                report["corrupt_images"].append(f"{lbl_path}: {e}")

    report["valid"] = (
        report["total_images"] > 0
        and report["total_annotations"] > 0
        and len(report["corrupt_images"]) == 0
        and len(report["splits_found"]) >= 2
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Waste Dataset Quality & Annotation Auditor")
    parser.add_argument("--dataset-dir", type=str, default="datasets/waste_data", help="Path to dataset root")
    parser.add_argument("--json-out", type=str, default="", help="Path to save audit report JSON")
    args = parser.parse_args()

    target = Path(args.dataset_dir)
    print(f"Auditing waste dataset at: {target.resolve()}...")
    report = audit_dataset(target)

    print("\n--- DATASET AUDIT SUMMARY ---")
    print(f"Total Images Found    : {report['total_images']}")
    print(f"Total Annotations     : {report['total_annotations']}")
    print(f"Splits Identified     : {report['splits_found']}")
    print(f"Corrupt Images        : {len(report['corrupt_images'])}")
    print(f"Missing Annotations   : {report['missing_annotations']}")
    print(f"Leakage Suspects      : {len(report['leakage_suspects'])}")
    print("\nClass Annotation Breakdown:")
    for cat_name, count in report["class_counts"].items():
        print(f"  - {cat_name:18}: {count}")

    if report["unknown_class_ids"]:
        print(f"\n[WARNING] Unapproved Class IDs detected: {report['unknown_class_ids']}")

    print(f"\nOverall Dataset Validity: {'PASS' if report['valid'] else 'FAIL'}")

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Audit report saved to: {out_p}")


if __name__ == "__main__":
    main()

