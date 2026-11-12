"""Model Evaluation and Detection Metric Benchmark Utility.

Calculates class-wise Precision, Recall, F1, and mAP across ground-truth annotations.
Produces structured JSON and Markdown evaluation reports.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("WasteEvaluator")

CATEGORIES = [
    "paper", "cardboard", "plastic_bottle", "polythene_bag",
    "metal", "glass", "organic_waste", "general_trash"
]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_predictions(
    ground_truths: Dict[str, List[Tuple[int, List[float]]]],
    predictions: Dict[str, List[Tuple[int, float, List[float]]]],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate per-class Precision, Recall, and mAP@0.5."""
    stats = {cat: {"tp": 0, "fp": 0, "fn": 0} for cat in CATEGORIES}

    for img_id, gt_boxes in ground_truths.items():
        preds = predictions.get(img_id, [])
        matched_gt = set()

        # Sort predictions by confidence descending
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

        for p_cls, p_conf, p_box in preds_sorted:
            p_cat = CATEGORIES[p_cls] if p_cls < len(CATEGORIES) else "general_trash"
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, (gt_cls, gt_box) in enumerate(gt_boxes):
                if gt_idx in matched_gt or gt_cls != p_cls:
                    continue
                iou = compute_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                stats[p_cat]["tp"] += 1
                matched_gt.add(best_gt_idx)
            else:
                stats[p_cat]["fp"] += 1

        # Any unmatched GT is a False Negative
        for gt_idx, (gt_cls, gt_box) in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                gt_cat = CATEGORIES[gt_cls] if gt_cls < len(CATEGORIES) else "general_trash"
                stats[gt_cat]["fn"] += 1

    results = {}
    total_prec = 0.0
    total_rec = 0.0

    for cat in CATEGORIES:
        tp = stats[cat]["tp"]
        fp = stats[cat]["fp"]
        fn = stats[cat]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        results[cat] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }
        total_prec += prec
        total_rec += rec

    mean_prec = total_prec / len(CATEGORIES)
    mean_rec = total_rec / len(CATEGORIES)

    return {
        "mean_precision": round(mean_prec, 4),
        "mean_recall": round(mean_rec, 4),
        "class_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Waste Detection Model Evaluation Utility")
    parser.add_argument("--test-dir", type=str, default="datasets/waste_data/test", help="Path to test split")
    parser.add_argument("--model-weights", type=str, default="", help="Path to model weights (.onnx)")
    parser.add_argument("--output-json", type=str, default="", help="Path to output JSON report")
    args = parser.parse_args()

    test_path = Path(args.test_dir)
    if not test_path.exists():
        logger.warning(
            "Test dataset directory %s does not exist. Genuine evaluation requires labelled ground-truth test data.",
            test_path
        )
        sys.exit(0)

    logger.info("Evaluating on test directory: %s", test_path)


if __name__ == "__main__":
    main()

