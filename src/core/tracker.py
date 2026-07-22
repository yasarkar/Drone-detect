import logging
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

# Setup module logging
logger = logging.getLogger(__name__)


def compute_iou(box_a: list[int], box_b: list[int]) -> float:
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box_a: Bounding box [x1, y1, x2, y2]
        box_b: Bounding box [x1, y1, x2, y2]

    Returns:
        IoU value between 0.0 and 1.0.
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / float(union_area)


class DroneTracker:
    """
    A modular multi-object tracker for tracking drones across video frames.
    Uses Hungarian algorithm for IoU-based association and manages track life cycle.
    Supports persistent track IDs and movement trail rendering.
    """

    def __init__(self, config_path: str | Path):
        """
        Initializes the DroneTracker with parameters loaded from config.yaml.

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

        # Parse tracking configurations
        tracking_config = self.config.get("tracking", {})
        self.tracking_enabled = tracking_config.get("enabled", True)
        self.track_thresh = float(tracking_config.get("track_thresh", 0.5))
        self.track_buffer = int(tracking_config.get("track_buffer", 30))
        self.match_thresh = float(tracking_config.get("match_thresh", 0.8))  # Max cost (1 - IoU) allowed for match
        self.draw_trail = tracking_config.get("draw_trail", True)
        self.trail_length = int(tracking_config.get("trail_length", 30))

        # Parse visuals for rendering
        visuals = self.config.get("visuals", {})
        self.bbox_color = tuple(visuals.get("bbox_color", [0, 255, 0]))
        self.bbox_thickness = int(visuals.get("bbox_thickness", 2))
        self.font_scale = float(visuals.get("font_scale", 0.6))

        # Tracker state variables
        self.next_track_id = 1
        self.tracks = []  # List of dictionaries, each representing an active track
        # track dict structure:
        # {
        #     "track_id": int,
        #     "bbox": [x1, y1, x2, y2],
        #     "confidence": float,
        #     "class_id": int,
        #     "class_name": str,
        #     "lost_count": int,
        #     "trail": list[tuple[int, int]]
        # }

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """
        Updates the tracker with new detections and manages active tracks.

        Args:
            detections: List of detection dictionaries from the detector.
            frame: Current frame (used for dimensions or visual reference).

        Returns:
            A list of tracked detection dictionaries, containing an added "track_id".
        """
        if not self.tracking_enabled:
            # If tracking is disabled, simply return detections unchanged
            return detections

        # Filter detections below tracking threshold
        filtered_dets = [d for d in detections if d["confidence"] >= self.track_thresh]

        # Partition existing tracks into active (lost_count == 0) and lost (lost_count > 0)
        # We will match against all tracks that are within the buffer limit
        matched_tracks = []
        unmatched_tracks = list(self.tracks)
        unmatched_dets = list(filtered_dets)

        if len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            # Build cost matrix based on IoU distance (1.0 - IoU)
            cost_matrix = np.zeros((len(unmatched_tracks), len(unmatched_dets)), dtype=np.float32)
            for i, track in enumerate(unmatched_tracks):
                for j, det in enumerate(unmatched_dets):
                    cost_matrix[i, j] = 1.0 - compute_iou(track["bbox"], det["bbox"])

            # Solve linear assignment problem
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            # Process assignments
            # Note: we must iterate backwards or keep track of indices to remove from unmatched lists safely
            assignments = []
            for t_idx, d_idx in zip(track_indices, det_indices):
                cost = cost_matrix[t_idx, d_idx]
                # If cost is below match_thresh (i.e. IoU is above 1 - match_thresh)
                if cost <= self.match_thresh:
                    assignments.append((t_idx, d_idx))

            # Apply matched assignments
            # Sort assignments in descending order of track index to remove safely
            assignments.sort(key=lambda x: x[0], reverse=True)
            for t_idx, d_idx in assignments:
                track = unmatched_tracks.pop(t_idx)
                det = unmatched_dets.pop(d_idx)

                # Update track details
                track["bbox"] = det["bbox"]
                track["confidence"] = det["confidence"]
                track["lost_count"] = 0
                
                # Update trail
                x1, y1, x2, y2 = det["bbox"]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                track["trail"].append((cx, cy))
                if len(track["trail"]) > self.trail_length:
                    track["trail"].pop(0)

                matched_tracks.append(track)

        # For remaining unmatched tracks, increment lost count
        for track in unmatched_tracks:
            track["lost_count"] += 1
            # Keep the track if it hasn't exceeded the buffer limit
            if track["lost_count"] <= self.track_buffer:
                matched_tracks.append(track)

        # For unmatched detections, create new tracks
        for det in unmatched_dets:
            x1, y1, x2, y2 = det["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            new_track = {
                "track_id": self.next_track_id,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "lost_count": 0,
                "trail": [(cx, cy)]
            }
            self.next_track_id += 1
            matched_tracks.append(new_track)

        # Save all currently active/lost tracks for the next frame
        self.tracks = matched_tracks

        # Return only the currently active tracks (lost_count == 0) in the formatted output
        output_detections = []
        for track in self.tracks:
            if track["lost_count"] == 0:
                output_det = {
                    "bbox": track["bbox"],
                    "confidence": track["confidence"],
                    "class_id": track["class_id"],
                    "class_name": track["class_name"],
                    "track_id": track["track_id"]
                }
                output_detections.append(output_det)

        return output_detections

    def draw_tracks(self, frame: np.ndarray, tracked_detections: list[dict]) -> np.ndarray:
        """
        Draws bounding boxes, track IDs, and movement trails onto the frame.

        Args:
            frame: Input BGR frame.
            tracked_detections: List of tracked detections containing "track_id".

        Returns:
            Annotated frame.
        """
        if frame is None:
            return np.array([])

        annotated_frame = frame.copy()

        # 1. Draw trails for all active tracks in history
        if self.draw_trail:
            for track in self.tracks:
                trail = track["trail"]
                if len(trail) < 2:
                    continue
                # Draw lines between consecutive points in the trail
                for i in range(1, len(trail)):
                    pt1 = trail[i - 1]
                    pt2 = trail[i]
                    # Make older trail segments slightly thinner/faded
                    thickness = max(1, int(self.bbox_thickness * (i / len(trail))))
                    # Draw trail line (in blue or using bbox_color)
                    cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), thickness, cv2.LINE_AA)

        # 2. Draw bounding boxes and labels
        for det in tracked_detections:
            x1, y1, x2, y2 = det["bbox"]
            track_id = det["track_id"]
            label = f"Drone #{track_id} {det['confidence']*100:.1f}%"

            # Draw bbox
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

            # Determine background box placement
            text_y = y1 - baseline - 2
            if text_y < 0:
                text_y = y1 + text_h + baseline + 2
                bg_top = y1
                bg_bottom = y1 + text_h + baseline + 4
            else:
                bg_top = y1 - text_h - baseline - 4
                bg_bottom = y1

            # Draw background label box
            cv2.rectangle(
                annotated_frame,
                (x1, bg_top),
                (x1 + text_w + 4, bg_bottom),
                self.bbox_color,
                cv2.FILLED
            )

            # Render text
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
