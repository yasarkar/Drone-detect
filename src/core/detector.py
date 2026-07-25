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
    def __init__(self, config_path: str | Path):
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

        # SAHI (Slicing Aided Hyper Inference) configuration
        sahi_config = self.config.get("sahi", {})
        self.sahi_enabled = sahi_config.get("enabled", False)
        self.sahi_slice_height = int(sahi_config.get("slice_height", 512))
        self.sahi_slice_width = int(sahi_config.get("slice_width", 512))
        self.sahi_overlap_height_ratio = float(sahi_config.get("overlap_height_ratio", 0.2))
        self.sahi_overlap_width_ratio = float(sahi_config.get("overlap_width_ratio", 0.2))
        self.sahi_postprocess_type = sahi_config.get("postprocess_type", "NMS")
        self.sahi_postprocess_match_threshold = float(sahi_config.get("postprocess_match_threshold", 0.5))

        # Visual settings
        visuals = self.config.get("visuals", {})
        self.bbox_color = tuple(visuals.get("bbox_color", [0, 255, 0]))
        self.bbox_thickness = int(visuals.get("bbox_thickness", 2))
        self.font_scale = float(visuals.get("font_scale", 0.6))

        # Validate weights existence
        if not self.weights_path.exists():
            logger.error(f"Weights file not found at: {self.weights_path.absolute()}")
            raise FileNotFoundError(f"Model weights file not found at: {self.weights_path.absolute()}")

        # Initialize SAHI AutoDetectionModel wrapper if enabled
        if self.sahi_enabled:
            logger.info("SAHI is enabled. Initializing SAHI AutoDetectionModel wrapper.")
            try:
                from sahi import AutoDetectionModel
                sahi_device = self.device
                if isinstance(sahi_device, int):
                    sahi_device = f"cuda:{sahi_device}"

                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8',
                    model_path=str(self.weights_path),
                    confidence_threshold=self.confidence_threshold,
                    device=sahi_device,
                )
                logger.info("SAHI AutoDetectionModel initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize SAHI AutoDetectionModel: {e}")
                logger.warning("Falling back to standard YOLOv8 model for inference.")
                self.sahi_enabled = False

        # Always initialize standard model as fallback
        try:
            self.model = YOLO(self.weights_path)
            logger.info(f"YOLO model successfully loaded from {self.weights_path}")
        except Exception as e:
            logger.error(f"Failed to instantiate YOLO model with weights from {self.weights_path}: {e}")
            raise

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None:
            logger.warning("Received empty/None frame for detection.")
            return []

        # Run SAHI sliced inference if enabled
        if self.sahi_enabled:
            try:
                from sahi.predict import get_sliced_prediction
                sliced_result = get_sliced_prediction(
                    frame,
                    self.sahi_model,
                    slice_height=self.sahi_slice_height,
                    slice_width=self.sahi_slice_width,
                    overlap_height_ratio=self.sahi_overlap_height_ratio,
                    overlap_width_ratio=self.sahi_overlap_width_ratio,
                    postprocess_type=self.sahi_postprocess_type,
                    postprocess_match_threshold=self.sahi_postprocess_match_threshold,
                    verbose=0
                )
                
                detections = []
                for obj in sliced_result.object_prediction_list:
                    x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
                    confidence = float(obj.score.value)
                    class_id = int(obj.category.id)
                    class_name = str(obj.category.name)
                    
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })
                return detections
            except Exception as e:
                logger.error(f"SAHI sliced inference failed: {e}. Falling back to standard YOLO inference.")

        # Fallback / standard YOLOv8 inference
        try:
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
            logger.error(f"YOLOv8 inference failed: {e}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
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
