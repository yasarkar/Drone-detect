import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import yaml

# Module logger
logger = logging.getLogger(__name__)


class SnapshotManager:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            logger.error(f"Configuration file not found at: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read/parse configuration: {e}")
            raise

        # Parse snapshot settings
        snapshot_config = self.config.get("snapshot", {})
        self.enabled = snapshot_config.get("enabled", True)
        self.min_confidence = float(snapshot_config.get("min_confidence", 0.80))
        self.save_dir = Path(snapshot_config.get("save_dir", "captures"))
        self.debounce_seconds = float(snapshot_config.get("debounce_seconds", 5.0))
        self.draw_boxes_on_snapshot = snapshot_config.get("draw_boxes_on_snapshot", True)

        # Parse visual settings for annotation
        visuals = self.config.get("visuals", {})
        self.bbox_color = tuple(visuals.get("bbox_color", [0, 255, 0]))
        self.bbox_thickness = int(visuals.get("bbox_thickness", 2))
        self.font_scale = float(visuals.get("font_scale", 0.6))

        # In-memory dictionary tracking: {track_id: last_saved_time_float}
        self.last_saved_times = {}

    def save_snapshot(
        self,
        frame: np.ndarray,
        track_id: int,
        confidence: float,
        bbox: list[int] | None = None,
        class_name: str = "drone"
    ) -> Path | None:
        if not self.enabled:
            return None

        # 1. Validate confidence threshold
        if confidence < self.min_confidence:
            return None

        # 2. Apply debouncing check per track_id
        current_time = time.time()
        last_saved = self.last_saved_times.get(track_id, 0.0)
        if current_time - last_saved < self.debounce_seconds:
            # Debounce active; skip saving
            return None

        if frame is None or frame.size == 0:
            logger.warning(f"Attempted to save snapshot for track ID {track_id} but frame was empty.")
            return None

        # 3. Create daily subdirectory path: captures/YYYY-MM-DD/
        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        target_dir = self.save_dir / date_str
        target_dir.mkdir(parents=True, exist_ok=True)

        # 4. Prepare frame image (draw bbox on a copy if enabled)
        if self.draw_boxes_on_snapshot and bbox is not None:
            image_to_save = frame.copy()
            x1, y1, x2, y2 = map(int, bbox)
            label = f"{class_name} #{track_id} {confidence*100:.1f}%"

            # Draw bounding box
            cv2.rectangle(
                image_to_save,
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

            # Set label background position
            text_y = y1 - baseline - 2
            if text_y < 0:
                text_y = y1 + text_h + baseline + 2
                bg_top = y1
                bg_bottom = y1 + text_h + baseline + 4
            else:
                bg_top = y1 - text_h - baseline - 4
                bg_bottom = y1

            # Draw background box and text
            cv2.rectangle(
                image_to_save,
                (x1, bg_top),
                (x1 + text_w + 4, bg_bottom),
                self.bbox_color,
                cv2.FILLED
            )
            cv2.putText(
                image_to_save,
                label,
                (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        else:
            image_to_save = frame

        # 5. Build safe date-time filename
        # Format: drone_id_{track_id}_conf_{conf}_time_{YYYYMMDD_HHMMSS_mmmZ}.jpg
        time_fn_str = now_utc.strftime("%Y%m%d_%H%M%S_%f")[:-3] + "Z"
        conf_percent = int(confidence * 100)
        filename = f"drone_id_{track_id}_conf_{conf_percent}_time_{time_fn_str}.jpg"
        file_path = target_dir / filename

        # 6. Save image using OpenCV
        try:
            success = cv2.imwrite(str(file_path), image_to_save)
            if success:
                # Update debounce timestamp
                self.last_saved_times[track_id] = current_time
                logger.info(f"Snapshot saved for Track ID {track_id} to: {file_path}")
                return file_path
            else:
                logger.error(f"Failed to write snapshot image to path: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error saving snapshot image: {e}")
            return None
