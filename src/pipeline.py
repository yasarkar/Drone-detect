import logging
from pathlib import Path
import cv2
import numpy as np
import yaml

from src.core.detector import DroneDetector
from src.core.tracker import DroneTracker
from src.core.zone_logic import GeofenceManager
from src.utils.snapshot import SnapshotManager
from src.utils.logger import AuditLogger
from src.core.satellite import SatelliteTracker

# Setup module logging
logger = logging.getLogger(__name__)


class DronePipeline:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            logger.error(f"Configuration file not found at: {self.config_path}")
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info("DronePipeline configuration loaded.")
        except Exception as e:
            logger.error(f"Failed to parse configuration YAML: {e}")
            raise

        # 1. Initialize Detector
        try:
            self.detector = DroneDetector(self.config_path)
        except Exception as e:
            logger.error(f"Failed to initialize DroneDetector: {e}")
            raise

        # 2. Initialize Tracker if enabled
        self.tracker = None
        tracking_enabled = self.config.get("tracking", {}).get("enabled", True)
        if tracking_enabled:
            try:
                self.tracker = DroneTracker(self.config_path)
            except Exception as e:
                logger.error(f"Failed to initialize DroneTracker: {e}")
                raise

        # 3. Initialize Geofencing if enabled
        self.geofence = None
        geofence_enabled = self.config.get("geofencing", {}).get("enabled", True)
        if geofence_enabled:
            try:
                self.geofence = GeofenceManager(self.config_path)
            except Exception as e:
                logger.error(f"Failed to initialize GeofenceManager: {e}")
                raise

        # 4. Initialize Snapshot Manager if enabled
        self.snapshot_mgr = None
        snapshot_enabled = self.config.get("snapshot", {}).get("enabled", True)
        if snapshot_enabled:
            try:
                self.snapshot_mgr = SnapshotManager(self.config_path)
            except Exception as e:
                logger.error(f"Failed to initialize SnapshotManager: {e}")
                raise

        # 5. Initialize Audit Logger if enabled
        self.audit_logger = None
        logging_enabled = self.config.get("logging", {}).get("enabled", True)
        if logging_enabled:
            try:
                self.audit_logger = AuditLogger(self.config_path)
            except Exception as e:
                logger.error(f"Failed to initialize AuditLogger: {e}")
                raise

        # 6. Initialize Satellite Tracker if enabled
        self.satellite_tracker = None
        satellite_enabled = self.config.get("satellite", {}).get("enabled", True)
        if satellite_enabled:
            try:
                self.satellite_tracker = SatelliteTracker(self.config)
            except Exception as e:
                logger.error(f"Failed to initialize SatelliteTracker: {e}")
                raise

    def process_frame(self, frame: np.ndarray, current_fps: float = 0.0) -> tuple[np.ndarray, list[dict]]:
        if frame is None or frame.size == 0:
            return frame, []

        # Step 1: Detect targets
        raw_detections = self.detector.detect(frame)

        # Step 2: Update target tracker states
        if self.tracker is not None:
            tracked_detections = self.tracker.update(raw_detections, frame)
        else:
            tracked_detections = raw_detections

        # Step 3: Evaluate geofencing violations
        if self.geofence is not None:
            final_detections = self.geofence.check_violations(tracked_detections, frame.shape)
        else:
            final_detections = tracked_detections

        # Step 4: Evaluate Snapshot capturing & Audit logging per target
        for det in final_detections:
            track_id = det.get("track_id", -1)
            in_zone = det.get("in_zone", False)
            zone_name = det.get("zone_name", "OUTSIDE")

            # Debounced Snapshot Save
            snapshot_path = None
            if self.snapshot_mgr is not None:
                snapshot_path = self.snapshot_mgr.save_snapshot(
                    frame=frame,
                    track_id=track_id,
                    confidence=det["confidence"],
                    bbox=det["bbox"],
                    class_name=det["class_name"]
                )

            # Audit Event Log
            if self.audit_logger is not None:
                zone_status = f"INSIDE: {zone_name}" if in_zone else "OUTSIDE"
                self.audit_logger.log_event(
                    track_id=track_id,
                    confidence=det["confidence"],
                    bbox=det["bbox"],
                    zone_status=zone_status,
                    snapshot_path=snapshot_path
                )

        # Step 5: Render overlays and visualizations
        annotated_frame = self._draw_visuals(frame, final_detections, current_fps)

        return annotated_frame, final_detections

    def _draw_visuals(self, frame: np.ndarray, detections: list[dict], current_fps: float) -> np.ndarray:
        annotated_frame = frame.copy()

        # 1. Render Geofencing Zones
        if self.geofence is not None:
            annotated_frame = self.geofence.draw_zones(annotated_frame)

        # 2. Draw tracker trajectory trails
        if self.tracker is not None and self.tracker.draw_trail:
            for track in self.tracker.tracks:
                trail = track["trail"]
                if len(trail) < 2:
                    continue
                for i in range(1, len(trail)):
                    pt1 = trail[i - 1]
                    pt2 = trail[i]
                    thickness = max(1, int(self.detector.bbox_thickness * (i / len(trail))))
                    cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), thickness, cv2.LINE_AA)

        # 3. Draw target bounding boxes and "Drone Detected" label tags
        total_targets = len(detections)
        for idx, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = det["bbox"]

            # Standard green bounding box
            color = self.detector.bbox_color

            # Draw bbox
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                self.detector.bbox_thickness
            )

            # Build clean label text
            if total_targets > 1:
                label = f"Drone Detected #{idx} {det['confidence']*100:.1f}%"
            else:
                label = f"Drone Detected {det['confidence']*100:.1f}%"

            # Calculate label text bounding box
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.detector.font_scale,
                1
            )

            # Adjust background box vertical alignment
            text_y = y1 - baseline - 2
            if text_y < 0:
                text_y = y1 + text_h + baseline + 2
                bg_top = y1
                bg_bottom = y1 + text_h + baseline + 4
            else:
                bg_top = y1 - text_h - baseline - 4
                bg_bottom = y1

            # Render background shape and text
            cv2.rectangle(
                annotated_frame,
                (x1, bg_top),
                (x1 + text_w + 4, bg_bottom),
                color,
                cv2.FILLED
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.detector.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 4. Render top warning banner if any geofence zone is actively violated
        if self.geofence is not None and len(self.geofence.violated_zones) > 0:
            banner_h = 45
            banner_overlay = annotated_frame.copy()

            cv2.rectangle(
                banner_overlay,
                (0, 0),
                (annotated_frame.shape[1], banner_h),
                (0, 0, 255),
                cv2.FILLED
            )

            cv2.addWeighted(banner_overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)

            warning_text = "WARNING: RESTRICTED ZONE VIOLATION!"
            (tw, th), tb = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = (annotated_frame.shape[1] - tw) // 2
            ty = (banner_h + th) // 2
            cv2.putText(
                annotated_frame,
                warning_text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # 5. Render FPS overlay on the top-left
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(
            annotated_frame,
            fps_text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # Green FPS text
            2,
            cv2.LINE_AA
        )

        # 6. Draw Satellite Overlay if enabled and configured to "overlay" mode
        if self.satellite_tracker is not None and self.satellite_tracker.enabled:
            if self.satellite_tracker.display_mode == "overlay":
                annotated_frame = self.satellite_tracker.overlay_on_frame(annotated_frame)

        return annotated_frame
