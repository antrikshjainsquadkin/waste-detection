"""ONNX Export and CPU Optimization Utility.

Converts trained object detection checkpoints into deployable ONNX format,
validates graph structure, and verifies compatibility with ONNX Runtime CPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("OnnxExporter")


def validate_onnx_model(onnx_path: Path, input_shape: tuple = (1, 3, 416, 416)) -> bool:
    """Run smoke test on exported ONNX model with ONNX Runtime CPU."""
    logger.info("Validating ONNX model structure: %s", onnx_path)
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        logger.info("ONNX Inputs : %s", [i.name for i in inputs])
        logger.info("ONNX Outputs: %s", [o.name for o in outputs])

        dummy_input = np.zeros(input_shape, dtype=np.float32)
        res = session.run([outputs[0].name], {inputs[0].name: dummy_input})

        logger.info("Validation successful. Output shape: %s", res[0].shape)
        return True
    except Exception as e:
        logger.error("ONNX validation failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Trained Checkpoints to ONNX Runtime")
    parser.add_argument("--weights", type=str, required=True, help="Path to input weights (.pt or .onnx)")
    parser.add_argument("--output", type=str, default="", help="Path for exported .onnx file")
    parser.add_argument("--imgsz", type=int, default=416, help="Input dimension")
    args = parser.parse_args()

    weights_p = Path(args.weights)
    if not weights_p.exists():
        logger.error("Weights file does not exist: %s", weights_p)
        sys.exit(1)

    output_p = Path(args.output) if args.output else weights_p.with_suffix(".onnx")

    if weights_p.suffix.lower() == ".pt":
        logger.info("Exporting PyTorch weights %s to ONNX %s...", weights_p, output_p)
        try:
            from ultralytics import YOLO
            model = YOLO(str(weights_p))
            model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True)
            logger.info("Export completed.")
        except ImportError:
            logger.error("Ultralytics package not installed. Install with `uv pip install ultralytics` to run export.")
            sys.exit(1)
    elif weights_p.suffix.lower() == ".onnx":
        output_p = weights_p

    # Validate
    success = validate_onnx_model(output_p, input_shape=(1, 3, args.imgsz, args.imgsz))
    if success:
        logger.info("Model ready for registration in Waste Detection System.")
    else:
        logger.error("Exported model failed validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()

