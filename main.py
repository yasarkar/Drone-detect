import argparse
import sys
import time
import logging
from pathlib import Path
import cv2
import yaml

from src.pipeline import DronePipeline

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments for the main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Modular Drone Detection, Tracking & Geofencing System CLI."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input video source: file path, RTSP URL, or integer webcam index (e.g. 0)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Runs the pipeline in headless mode (no OpenCV GUI window)."
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Saves annotated output video stream to captures/output_video.mp4."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config_path = Path(args.config)

    # 1. Load config for fallback configurations
    if not config_path.exists():
        logger.error(f"Configuration file not found at: {config_path.absolute()}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse config file: {e}")
        sys.exit(1)

    # 2. Select source (CLI -> Config -> Webcam '0')
    source_raw = args.source or config.get("test_source", "0")
    if isinstance(source_raw, str) and source_raw.isdigit():
        source = int(source_raw)
    else:
        source = source_raw

    # 3. Instantiate unified pipeline
    logger.info("Initializing DronePipeline...")
    try:
        pipeline = DronePipeline(config_path)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline orchestrator: {e}")
        sys.exit(1)

    # 4. Open video/stream capture
    logger.info(f"Connecting to video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        sys.exit(1)

    # Resolve frame metrics for writing output files
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or fps_in != fps_in:  # check for NaN
        fps_in = 30.0

    # 5. Initialize Video Writer if output saving is requested
    writer = None
    if args.save_output:
        save_path = Path("captures/output_video.mp4")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # BGR mp4v format
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps_in, (width, height))
        logger.info(f"Saving output video stream to: {save_path.absolute()}")

    # 6. High-precision timing metrics for FPS
    prev_tick = cv2.getTickCount()
    tick_frequency = cv2.getTickFrequency()
    fps_smoothed = 0.0
    alpha = 0.9  # Exponential smoothing factor

    window_name = "Drone Detection, Tracking & Geofencing System"
    if not args.no_display:
        logger.info("Press 'q' inside display window to exit cleanly.")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream or cannot fetch next frame. Terminating.")
                break

            frame_count += 1

            # 7. Process frame through the orchestrator pipeline
            annotated_frame, detections = pipeline.process_frame(frame, fps_smoothed)

            # 8. High precision FPS update
            curr_tick = cv2.getTickCount()
            time_delta = (curr_tick - prev_tick) / tick_frequency
            prev_tick = curr_tick

            if time_delta > 0:
                fps_instant = 1.0 / time_delta
                if fps_smoothed == 0.0:
                    fps_smoothed = fps_instant
                else:
                    fps_smoothed = (alpha * fps_smoothed) + ((1.0 - alpha) * fps_instant)

            # 9. Save annotated frame to disk if requested
            if writer is not None:
                writer.write(annotated_frame)

            # 10. Display GUI window if not in headless mode
            if not args.no_display:
                try:
                    cv2.imshow(window_name, annotated_frame)
                    # Graceful exit on pressing 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Exit command 'q' received. Exiting.")
                        break
                except cv2.error as cv_err:
                    logger.warning(f"Display rendering error (running headless?): {cv_err}")
                    time.sleep(0.01)

            # Keep developer informed in the terminal
            if frame_count % 30 == 0:
                violation_state = len(pipeline.geofence.violated_zones) if pipeline.geofence else 0
                logger.info(
                    f"Frame {frame_count} | Targets: {len(detections)} | "
                    f"Violations: {violation_state} | Speed: {fps_smoothed:.1f} FPS"
                )

    except KeyboardInterrupt:
        logger.warning("\nPipeline execution interrupted by system signal (Ctrl+C).")
    except Exception as e:
        logger.error(f"Unhandled runtime exception in frame loop: {e}", exc_info=True)
    finally:
        # 11. Clean up and release system/video resources
        cap.release()
        if writer is not None:
            writer.release()
            logger.info("Output video writer released.")
        
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        
        logger.info("System shut down cleanly.")


if __name__ == "__main__":
    main()
