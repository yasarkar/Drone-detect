import logging
from pathlib import Path
import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO

# Setup module logging
logger = logging.getLogger(__name__)


class DroneDetector:
    """
    A modular object detector for detecting drones using YOLOv8 models.
    Supports GPU-accelerated inference and custom visualization options.
    """

    def __init__(self, config_path: str | Path):
        """
        Initializes the DroneDetector with parameters loaded from config.yaml.

        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            logger.error(f"Configuration file not found at: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration settings from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load or parse configuration YAML: {e}")
            raise

        # Parse inference settings
        self.weights_path = Path(self.config.get("weights_path", "weights/best.pt"))
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.50))
        self.iou_threshold = float(self.config.get("iou_threshold", 0.45))
        self.imgsz = int(self.config.get("imgsz", 640))

        # Device selection & check
        configured_device = self.config.get("device", 0)
        self.device = "cpu"
        if configured_device != "cpu":
            if torch.cuda.is_available():
                self.device = int(configured_device) if isinstance(configured_device, (int, float)) else configured_device
                logger.info(f"GPU device '{self.device}' detected. Running inference on GPU.")
            else:
                logger.warning("GPU was requested, but CUDA is not available. Falling back to CPU.")
                self.device = "cpu"
        else:
            logger.info("Running inference on CPU.")

        # Visual settings
        visuals = self.config.get("visuals", {})
        self.bbox_color = tuple(visuals.get("bbox_color", [0, 255, 0]))
        self.bbox_thickness = int(visuals.get("bbox_thickness", 2))
        self.font_scale = float(visuals.get("font_scale", 0.6))

        # Validate weights existence and load YOLO model
        if not self.weights_path.exists():
            logger.error(f"Weights file not found at: {self.weights_path.absolute()}")
            raise FileNotFoundError(f"Model weights file not found at: {self.weights_path.absolute()}")

        try:
            self.model = YOLO(self.weights_path)
            logger.info(f"YOLO model successfully loaded from {self.weights_path}")
        except Exception as e:
            logger.error(f"Failed to instantiate YOLO model with weights from {self.weights_path}: {e}")
            raise

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Runs object detection inference on a single BGR image/frame.

        Args:
            frame: Input BGR image as a NumPy array (OpenCV format).

        Returns:
            A list of dictionary objects representing the detections:
            [
                {
                    "bbox": [x1, y1, x2, y2],  # Bounding box coordinates (integers)
                    "confidence": 0.89,        # Detection confidence float
                    "class_id": 0,             # Class ID integer
                    "class_name": "drone"      # Name of the detected class
                },
                ...
            ]
        """
        if frame is None:
            logger.warning("Received empty/None frame for detection.")
            return []

        try:
            # Perform inference
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                for box in boxes:
                    # Extract coordinates and transfer to CPU NumPy
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = result.names[class_id]

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })

            return detections

        except Exception as e:
            logger.error(f"An error occurred during frame detection: {e}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draws annotated bounding boxes and labels onto a copy of the input frame.

        Args:
            frame: Input BGR frame.
            detections: List of detection dictionaries as returned by self.detect().

        Returns:
            Annotated copy of the frame.
        """
        if frame is None:
            return np.array([])

        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']*100:.1f}%"

            # Draw outer bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                self.bbox_color,
                self.bbox_thickness
            )

            # Get label text size
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                1
            )

            # Draw label background box above bbox, or inside if too close to top
            text_y = y1 - baseline - 2
            if text_y < 0:
                text_y = y1 + text_h + baseline + 2
                bg_top = y1
                bg_bottom = y1 + text_h + baseline + 4
            else:
                bg_top = y1 - text_h - baseline - 4
                bg_bottom = y1

            cv2.rectangle(
                annotated_frame,
                (x1, bg_top),
                (x1 + text_w + 4, bg_bottom),
                self.bbox_color,
                cv2.FILLED
            )

            # Render text on top of the label background
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated_frame
