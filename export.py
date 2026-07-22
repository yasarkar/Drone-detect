import argparse
import sys
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the model export script.
    """
    parser = argparse.ArgumentParser(
        description="Compile YOLOv8 model weights to ONNX/TensorRT for high-speed inference."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to pre-trained PyTorch weights (.pt file)."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="engine",
        choices=["engine", "onnx"],
        help="Format to export model into: 'engine' (TensorRT) or 'onnx'."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Target input image width/height for inference."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    weights_path = Path(args.weights)

    if not weights_path.exists():
        logger.error(f"Pre-trained weights file not found at: {weights_path.absolute()}")
        logger.error("Please make sure you have trained a model first or placed best.pt in the weights/ directory.")
        sys.exit(1)

    logger.info(f"Loading YOLO model: {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    logger.info(f"Starting model compilation to '{args.format}' format with imgsz={args.imgsz}...")
    try:
        # Execute YOLO model export
        # YOLOv8 export API supports dynamic formats like 'onnx', 'engine' (TensorRT), etc.
        exported_path = model.export(
            format=args.format,
            imgsz=args.imgsz,
            device=0  # Use CUDA device 0 for engine compilation if available
        )
        
        logger.info("Model compilation completed successfully!")
        logger.info(f"Compiled model is saved at: {exported_path}")

        # Construct instructions for config update
        logger.info("\n" + "="*80)
        logger.info("SYSTEM CONFIGURATION INSTRUCTIONS:")
        logger.info("To use this optimized engine for inference:")
        logger.info("1. Open the file 'config/config.yaml'")
        logger.info("2. Locate the 'weights_path' configuration key")
        logger.info("3. Update it to point to your new compiled model path:")
        
        if args.format == "engine":
            # Usually YOLO saves best.pt to best.engine in the same directory
            suggested_path = weights_path.with_suffix(".engine")
            logger.info(f"   weights_path: \"{suggested_path.as_posix()}\"")
            logger.info("Note: TensorRT (.engine) is GPU-specific. Do not copy it to other GPUs.")
        else:
            suggested_path = weights_path.with_suffix(".onnx")
            logger.info(f"   weights_path: \"{suggested_path.as_posix()}\"")
            
        logger.info("="*80 + "\n")

    except Exception as e:
        logger.error(f"An error occurred during model export: {e}")
        logger.error("Make sure your system has the correct CUDA/TensorRT environment variables configured if compiling to engine.")
        sys.exit(1)


if __name__ == "__main__":
    main()
