import argparse
import sys
import time
import logging
from pathlib import Path
import cv2
import yaml
from src.core.detector import DroneDetector
from src.core.tracker import DroneTracker

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
    parser = argparse.ArgumentParser(description="Test Inference for Drone Detector and Tracker System.")
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


def main():
    args = parse_arguments()
    config_path = Path(args.config)

    # 1. Load config to determine fallback source
    if not config_path.exists():
        logger.error(f"Configuration file not found at: {config_path.absolute()}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse config file: {e}")
        sys.exit(1)

    # 2. Select input source (CLI Arg -> Config File -> Default Webcam '0')
    source_raw = args.source or config.get("test_source", "0")
    
    # Try converting numeric strings to integer webcam indices
    if isinstance(source_raw, str) and source_raw.isdigit():
        source = int(source_raw)
    else:
        source = source_raw

    # Initialize Detector
    logger.info(f"Initializing DroneDetector with config: {config_path}")
    try:
        detector = DroneDetector(config_path)
    except FileNotFoundError as fnf:
        logger.error(f"Initialization failed: {fnf}")
        logger.error("Please make sure you have trained a model and saved the weights at the configured weights_path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize DroneDetector: {e}")
        sys.exit(1)

    # Initialize Tracker if enabled
    tracker = None
    tracking_config = config.get("tracking", {})
    tracking_enabled = tracking_config.get("enabled", True)

    if tracking_enabled:
        logger.info("Initializing DroneTracker...")
        try:
            tracker = DroneTracker(config_path)
        except Exception as e:
            logger.error(f"Failed to initialize DroneTracker: {e}")
            sys.exit(1)
    else:
        logger.info("Tracking is disabled in configuration.")

    # 3. Setup video capture
    logger.info(f"Opening video/stream source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Could not open source: {source}")
        sys.exit(1)

    window_name = "Drone Detector & Tracker Real-time Test"
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

            # 4. Perform detection
            detections = detector.detect(frame)

            # 5. Run tracking and annotation
            if tracker is not None:
                tracked_dets = tracker.update(detections, frame)
                annotated_frame = tracker.draw_tracks(frame, tracked_dets)
                count = len(tracked_dets)
            else:
                annotated_frame = detector.draw_detections(frame, detections)
                count = len(detections)

            # 6. Calculate real-time FPS
            curr_time = time.time()
            time_delta = curr_time - prev_time
            prev_time = curr_time

            if time_delta > 0:
                fps_instant = 1.0 / time_delta
                if fps_smoothed == 0.0:
                    fps_smoothed = fps_instant
                else:
                    fps_smoothed = (alpha * fps_smoothed) + ((1.0 - alpha) * fps_instant)

            # 7. Render overlays (FPS, SAHI and Tracking statuses) on top-left
            # Render FPS
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

            # Render SAHI status
            sahi_text = f"SAHI: {'ON' if detector.sahi_enabled else 'OFF'}"
            sahi_color = (0, 255, 0) if detector.sahi_enabled else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                sahi_text,
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                sahi_color,
                2,
                cv2.LINE_AA
            )

            # Render Tracker status
            track_text = f"TRACK: {'ON' if tracker is not None else 'OFF'}"
            track_color = (0, 255, 0) if tracker is not None else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                track_text,
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                track_color,
                2,
                cv2.LINE_AA
            )

            # Log frame statistics
            logger.info(
                f"Status: SAHI={detector.sahi_enabled}, Track={tracker is not None} | "
                f"Active Targets: {count} | Speed: {fps_smoothed:.1f} FPS"
            )

            # 8. Display real-time result window
            try:
                cv2.imshow(window_name, annotated_frame)
                # Check for 'q' key press to break out of loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exit key 'q' pressed. Terminating.")
                    break
            except cv2.error as cv_err:
                logger.warning(f"Display window could not be initialized (headless mode?): {cv_err}")
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.warning("Inference execution interrupted by user.")
    finally:
        # Cleanup video resources
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("Video resources released cleanly.")


if __name__ == "__main__":
    main()
