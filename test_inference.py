import argparse
import sys
import time
import logging
from pathlib import Path
import cv2
import yaml
import numpy as np

from src.core.detector import DroneDetector
from src.core.tracker import DroneTracker
from src.core.zone_logic import GeofenceManager
from src.utils.snapshot import SnapshotManager
from src.utils.logger import AuditLogger

# Configure script logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the inference test script.
    """
    parser = argparse.ArgumentParser(description="Integrated Drone Detection, Tracking & Geofencing System.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source: video file path or webcam index (e.g. 0). If not set, falls back to config."
    )
    return parser.parse_args()


def draw_pipeline_visuals(
    frame: np.ndarray,
    detections: list[dict],
    tracker: DroneTracker | None,
    geofence: GeofenceManager | None,
    detector: DroneDetector
) -> np.ndarray:
    """
    Draws geofencing zones, movement trails, colored bounding boxes, labels,
    and warning banners on the output frame.

    Args:
        frame: Input raw BGR frame.
        detections: List of enriched tracked detections.
        tracker: Active DroneTracker object or None.
        geofence: Active GeofenceManager object or None.
        detector: Active DroneDetector object.

    Returns:
        Annotated copy of the frame.
    """
    annotated_frame = frame.copy()

    # 1. Render Geofencing Zones
    if geofence is not None:
        annotated_frame = geofence.draw_zones(annotated_frame)

    # 2. Render Trajectory Trails
    if tracker is not None and tracker.draw_trail:
        for track in tracker.tracks:
            trail = track["trail"]
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                pt1 = trail[i - 1]
                pt2 = trail[i]
                # Scale thickness based on age
                thickness = max(1, int(detector.bbox_thickness * (i / len(trail))))
                # Draw blue trail line
                cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), thickness, cv2.LINE_AA)

    # 3. Draw Bounding Boxes and Labels
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = det.get("track_id", None)
        in_zone = det.get("in_zone", False)

        # RED bounding box if inside zone, else GREEN (detector configured color)
        color = (0, 0, 255) if in_zone else detector.bbox_color

        # Draw bbox rectangle
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            detector.bbox_thickness
        )

        # Format label text
        id_str = f" #{track_id}" if track_id is not None else ""
        label = f"{det['class_name']}{id_str} {det['confidence']*100:.1f}%"
        if in_zone:
            label += f" [ALERT: {det['zone_name']}]"

        # Calculate label background box size
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            detector.font_scale,
            1
        )

        # Place label above bounding box or inside if too close to top
        text_y = y1 - baseline - 2
        if text_y < 0:
            text_y = y1 + text_h + baseline + 2
            bg_top = y1
            bg_bottom = y1 + text_h + baseline + 4
        else:
            bg_top = y1 - text_h - baseline - 4
            bg_bottom = y1

        # Draw background and label text
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
            detector.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # 4. Render top warning banner if any geofence zone is actively violated
    if geofence is not None and len(geofence.violated_zones) > 0:
        banner_h = 45
        banner_overlay = annotated_frame.copy()
        
        # Red warning banner
        cv2.rectangle(
            banner_overlay,
            (0, 0),
            (annotated_frame.shape[1], banner_h),
            (0, 0, 255),
            cv2.FILLED
        )
        
        # Alpha blending for a premium warning screen bar overlay
        cv2.addWeighted(banner_overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)

        # Centered white text on the banner
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

    return annotated_frame


def main():
    args = parse_arguments()
    config_path = Path(args.config)

    # 1. Load config settings
    if not config_path.exists():
        logger.error(f"Configuration file not found at: {config_path.absolute()}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse config file: {e}")
        sys.exit(1)

    # 2. Select input source
    source_raw = args.source or config.get("test_source", "0")
    if isinstance(source_raw, str) and source_raw.isdigit():
        source = int(source_raw)
    else:
        source = source_raw

    # 3. Instantiate modules
    logger.info("Initializing system modules...")
    
    # Detector
    try:
        detector = DroneDetector(config_path)
    except Exception as e:
        logger.error(f"Failed to initialize DroneDetector: {e}")
        sys.exit(1)

    # Tracker
    tracker = None
    tracking_enabled = config.get("tracking", {}).get("enabled", True)
    if tracking_enabled:
        try:
            tracker = DroneTracker(config_path)
        except Exception as e:
            logger.error(f"Failed to initialize DroneTracker: {e}")
            sys.exit(1)

    # Geofencing
    geofence = None
    geofence_enabled = config.get("geofencing", {}).get("enabled", True)
    if geofence_enabled:
        try:
            geofence = GeofenceManager(config_path)
        except Exception as e:
            logger.error(f"Failed to initialize GeofenceManager: {e}")
            sys.exit(1)

    # Snapshot Manager
    snapshot_mgr = None
    snapshot_enabled = config.get("snapshot", {}).get("enabled", True)
    if snapshot_enabled:
        try:
            snapshot_mgr = SnapshotManager(config_path)
        except Exception as e:
            logger.error(f"Failed to initialize SnapshotManager: {e}")
            sys.exit(1)

    # Audit Logger
    audit_logger = None
    logging_enabled = config.get("logging", {}).get("enabled", True)
    if logging_enabled:
        try:
            audit_logger = AuditLogger(config_path)
        except Exception as e:
            logger.error(f"Failed to initialize AuditLogger: {e}")
            sys.exit(1)

    # 4. Setup video capture
    logger.info(f"Opening video/stream source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Could not open source: {source}")
        sys.exit(1)

    window_name = "Drone Detection, Tracking & Geofencing System"
    logger.info("Press 'q' in the display window to exit.")

    # Time tracking for FPS calculation
    prev_time = time.time()
    fps_smoothed = 0.0
    alpha = 0.9  # Smoothing factor for Exponential Moving Average FPS

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or cannot fetch frame.")
                break

            # 5. Core pipeline step execution
            # Detection
            raw_detections = detector.detect(frame)

            # Tracking
            if tracker is not None:
                tracked_detections = tracker.update(raw_detections, frame)
            else:
                tracked_detections = raw_detections

            # Geofencing violations check
            if geofence is not None:
                final_detections = geofence.check_violations(tracked_detections, frame.shape)
            else:
                final_detections = tracked_detections

            # 6. Evaluation for Event Logging & Snapshot Capture
            for det in final_detections:
                track_id = det.get("track_id", -1)
                in_zone = det.get("in_zone", False)
                zone_name = det.get("zone_name", "OUTSIDE")

                # Snapshot check (triggered if drone is detected, min conf met & debounced)
                snapshot_path = None
                if snapshot_mgr is not None:
                    # Save a snapshot (uses configured min_confidence & debounce internally)
                    snapshot_path = snapshot_mgr.save_snapshot(
                        frame=frame,
                        track_id=track_id,
                        confidence=det["confidence"],
                        bbox=det["bbox"],
                        class_name=det["class_name"]
                    )

                # Event logging
                if audit_logger is not None:
                    zone_status = f"INSIDE: {zone_name}" if in_zone else "OUTSIDE"
                    audit_logger.log_event(
                        track_id=track_id,
                        confidence=det["confidence"],
                        bbox=det["bbox"],
                        zone_status=zone_status,
                        snapshot_path=snapshot_path
                    )

            # 7. Render annotations and visual layouts
            annotated_frame = draw_pipeline_visuals(
                frame, final_detections, tracker, geofence, detector
            )

            # 8. Calculate real-time FPS
            curr_time = time.time()
            time_delta = curr_time - prev_time
            prev_time = curr_time

            if time_delta > 0:
                fps_instant = 1.0 / time_delta
                if fps_smoothed == 0.0:
                    fps_smoothed = fps_instant
                else:
                    fps_smoothed = (alpha * fps_smoothed) + ((1.0 - alpha) * fps_instant)

            # 9. Draw status dashboard on the frame (top-left area)
            # Render smoothed FPS
            fps_text = f"FPS: {fps_smoothed:.1f}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),  # Red
                2,
                cv2.LINE_AA
            )

            # Status overlays
            # SAHI
            sahi_text = f"SAHI: {'ON' if detector.sahi_enabled else 'OFF'}"
            sahi_color = (0, 255, 0) if detector.sahi_enabled else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                sahi_text,
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                sahi_color,
                2,
                cv2.LINE_AA
            )

            # Tracking
            track_text = f"TRACK: {'ON' if tracker is not None else 'OFF'}"
            track_color = (0, 255, 0) if tracker is not None else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                track_text,
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                track_color,
                2,
                cv2.LINE_AA
            )

            # Geofencing
            geofence_text = f"GEOFENCE: {'ON' if geofence is not None else 'OFF'}"
            geofence_color = (0, 255, 0) if geofence is not None else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                geofence_text,
                (15, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                geofence_color,
                2,
                cv2.LINE_AA
            )

            # Snapshots
            snap_text = f"SNAPSHOTS: {'ON' if snapshot_mgr is not None else 'OFF'}"
            snap_color = (0, 255, 0) if snapshot_mgr is not None else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                snap_text,
                (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                snap_color,
                2,
                cv2.LINE_AA
            )

            # Logging
            log_text = f"LOGGING: {'ON' if audit_logger is not None else 'OFF'}"
            log_color = (0, 255, 0) if audit_logger is not None else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                log_text,
                (15, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                log_color,
                2,
                cv2.LINE_AA
            )

            # Log frame stats to console/logs
            logger.info(
                f"Targets: {len(final_detections)} | Violated Zones: {len(geofence.violated_zones) if geofence else 0} | "
                f"Speed: {fps_smoothed:.1f} FPS"
            )

            # 10. Display real-time result window
            try:
                cv2.imshow(window_name, annotated_frame)
                # Exit check on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exit key 'q' pressed. Terminating.")
                    break
            except cv2.error as cv_err:
                logger.warning(f"Display window could not be initialized: {cv_err}")
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user.")
    finally:
        # Resource cleanup
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("Video resources released cleanly.")


if __name__ == "__main__":
    main()
