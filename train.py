import os
import sys
import shutil
import logging
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# Setup logging configuration
# Log to both stdout and a log file in the logs directory
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs/training.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Drone Detection YOLOv8 Training Pipeline.")

    # 1. Load configuration from config/config.yaml
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.error(f"Configuration file not found at: {config_path.absolute()}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
    except Exception as e:
        logger.error(f"Failed to read/parse configuration file: {e}")
        sys.exit(1)

    # 2. Check and validate paths
    data_yaml_path = Path(config.get("data_yaml_path", "drone_dataset/data.yaml"))
    if not data_yaml_path.exists():
        logger.error(
            f"Dataset configuration file not found at: '{data_yaml_path.absolute()}'.\n"
            f"Please ensure your dataset is placed correctly and data.yaml exists before training."
        )
        sys.exit(1)

    # 3. Check GPU availability
    configured_device = config.get("device", 0)
    device = "cpu"

    if configured_device != "cpu":
        if torch.cuda.is_available():
            device = int(configured_device) if isinstance(configured_device, (int, float)) else configured_device
            logger.info(f"GPU acceleration is available. Running training on CUDA device: {device}")
        else:
            logger.warning("GPU was requested in configuration, but CUDA is not available. Falling back to CPU.")
            device = "cpu"
    else:
        logger.info("Running training on CPU (as configured).")

    # 4. Instantiate the YOLO model
    model_type = config.get("model_type", "yolov8m.pt")
    logger.info(f"Initializing YOLO model: {model_type}")
    try:
        model = YOLO(model_type)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    # 5. Extract training parameters
    epochs = config.get("epochs", 50)
    imgsz = config.get("imgsz", 640)
    batch_size = config.get("batch_size", 16)
    project_dir = config.get("project_dir", "runs/detect")
    run_name = config.get("name", "drone_model_exp")

    logger.info("=== Training Parameters ===")
    logger.info(f"Model Type   : {model_type}")
    logger.info(f"Dataset YAML : {data_yaml_path}")
    logger.info(f"Epochs       : {epochs}")
    logger.info(f"Image Size   : {imgsz}")
    logger.info(f"Batch Size   : {batch_size}")
    logger.info(f"Project Dir  : {project_dir}")
    logger.info(f"Run Name     : {run_name}")
    logger.info(f"Device       : {device}")
    logger.info("===========================")

    # 6. Execute model training
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=0,
            project=project_dir,
            name=run_name,
            exist_ok=True  # Overwrite/reuse the run folder instead of auto-incrementing
        )
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        sys.exit(1)

    # 7. Print metric summaries
    if results is not None:
        logger.info("=== Final Evaluation Metrics ===")
        # Safely extract metrics from the Results / DetMetrics object
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            for key, val in results.results_dict.items():
                metric_name = key.replace("metrics/", "")
                if isinstance(val, float):
                    logger.info(f"  {metric_name:<20}: {val:.5f}")
                else:
                    logger.info(f"  {metric_name:<20}: {val}")
        elif hasattr(results, "mean_results"):
            try:
                metrics_list = results.mean_results()
                logger.info(f"  Precision           : {metrics_list[0]:.5f}")
                logger.info(f"  Recall              : {metrics_list[1]:.5f}")
                logger.info(f"  mAP50               : {metrics_list[2]:.5f}")
                logger.info(f"  mAP50-95            : {metrics_list[3]:.5f}")
            except Exception as ex:
                logger.warning(f"Could not parse mean_results: {ex}")
        else:
            logger.info(f"Raw metrics object: {results}")

    # 8. Copy/Save best weights to the weights/ directory
    # Determine the actual save directory of the run
    actual_save_dir = None
    if hasattr(model, "trainer") and model.trainer is not None:
        if hasattr(model.trainer, "save_dir"):
            actual_save_dir = Path(model.trainer.save_dir)

    if not actual_save_dir:
        actual_save_dir = Path(project_dir) / run_name

    best_weights_source = actual_save_dir / "weights" / "best.pt"
    weights_dest_dir = Path("weights")
    weights_dest_dir.mkdir(parents=True, exist_ok=True)
    best_weights_dest = weights_dest_dir / "best.pt"

    if best_weights_source.exists():
        logger.info(f"Locating best weights at: {best_weights_source}")
        try:
            shutil.copy(best_weights_source, best_weights_dest)
            logger.info(f"Successfully copied best weights to: {best_weights_dest.absolute()}")
        except Exception as e:
            logger.error(f"Failed to copy weights to destination: {e}")
    else:
        logger.warning(f"Best weights file not found at expected path: {best_weights_source.absolute()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception in training pipeline: {e}")
        sys.exit(1)
