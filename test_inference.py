import argparse
import sys
import time
import logging
from pathlib import Path
import cv2

from src.pipeline import DronePipeline

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
    parser = argparse.ArgumentParser(description="Integrated Drone Detection, Tracking & Geofencing System Test.")
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
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Runs test pipeline in headless mode without GUI window."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config_path = Path(args.config)

    # 1. Instantiate unified orchestrator pipeline
    logger.info("Initializing DronePipeline orchestrator...")
    try:
        pipeline = DronePipeline(config_path)
    except Exception as e:
        logger.error(f"Failed to initialize DronePipeline: {e}")
        sys.exit(1)

    # 2. Select input source
    source_raw = args.source or pipeline.config.get("test_source", "0")
    if isinstance(source_raw, str) and source_raw.isdigit():
        source = int(source_raw)
    else:
        source = source_raw

    # 3. Setup video capture
    logger.info(f"Opening video/stream source: {source}")
    if isinstance(source, int):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30.0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Could not open source: {source}")
        sys.exit(1)

    window_name = "Drone Detection, Tracking & Geofencing System (Test Mode)"
    if not args.no_display:
        logger.info("Press 'q' inside display window to exit cleanly.")
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except Exception as e:
            logger.warning(f"Could not initialize display window settings: {e}")

    prev_tick = cv2.getTickCount()
    tick_frequency = cv2.getTickFrequency()
    fps_smoothed = 0.0
    alpha = 0.9

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream or cannot fetch next frame. Terminating.")
                break

            frame_count += 1

            # 4. Delegate single frame execution to DronePipeline
            annotated_frame, detections = pipeline.process_frame(frame, fps_smoothed)

            # High precision FPS calculation
            curr_tick = cv2.getTickCount()
            time_delta = (curr_tick - prev_tick) / tick_frequency
            prev_tick = curr_tick

            if time_delta > 0:
                fps_instant = 1.0 / time_delta
                fps_smoothed = fps_instant if fps_smoothed == 0.0 else (alpha * fps_smoothed) + ((1.0 - alpha) * fps_instant)

            if frame_count % 30 == 0:
                logger.info(
                    f"Frame {frame_count} | Detections: {len(detections)} | "
                    f"Speed: {fps_smoothed:.1f} FPS"
                )

            # 5. Display frame if GUI window is active
            if not args.no_display:
                try:
                    cv2.imshow(window_name, annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Exit command 'q' received. Exiting.")
                        break
                except cv2.error as cv_err:
                    logger.warning(f"Display rendering error: {cv_err}")
                    time.sleep(0.01)

    except KeyboardInterrupt:
        logger.warning("Test pipeline execution interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info("Test execution completed and resources released cleanly.")


if __name__ == "__main__":
    main()
