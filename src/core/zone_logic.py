import logging
from pathlib import Path
import cv2
import numpy as np
import yaml

# Module logger
logger = logging.getLogger(__name__)


class GeofenceManager:
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

        geofence_config = self.config.get("geofencing", {})
        self.enabled = geofence_config.get("enabled", True)
        
        # Load zones list
        self.zones = []
        raw_zones = geofence_config.get("zones", [])
        for z in raw_zones:
            polygon_pts = np.array(z.get("polygon", []), dtype=np.float32)
            color = tuple(z.get("color", [0, 0, 255])) # BGR Red default
            self.zones.append({
                "name": z.get("name", "Unnamed_Zone"),
                "normalized_polygon": polygon_pts,
                "pixel_polygon": None,
                "color": color
            })

        # Frame dimension caching to avoid recalculating pixel values on every frame
        self.last_width = 0
        self.last_height = 0

        # In-memory tracking of currently violated zone names
        self.violated_zones = set()

    def _update_pixel_polygons(self, frame_width: int, frame_height: int):
        if self.last_width == frame_width and self.last_height == frame_height:
            return  # Dimensions did not change

        self.last_width = frame_width
        self.last_height = frame_height

        for zone in self.zones:
            norm_poly = zone["normalized_polygon"]
            if len(norm_poly) == 0:
                continue
            
            # Map normalized [x, y] to pixel coordinate values
            pixel_poly = np.zeros_like(norm_poly, dtype=np.int32)
            pixel_poly[:, 0] = (norm_poly[:, 0] * frame_width).astype(np.int32)
            pixel_poly[:, 1] = (norm_poly[:, 1] * frame_height).astype(np.int32)
            
            # OpenCV pointPolygonTest requires shape (N, 1, 2) or (N, 2) of type int32
            zone["pixel_polygon"] = pixel_poly

    def check_violations(self, tracked_detections: list[dict], frame_shape: tuple) -> list[dict]:
        if not self.enabled or len(self.zones) == 0:
            # If disabled, fill in default safe values
            for det in tracked_detections:
                det["in_zone"] = False
                det["zone_name"] = "OUTSIDE"
            self.violated_zones.clear()
            return tracked_detections

        # Extract dimensions from frame shape (height, width)
        fh, fw = frame_shape[0], frame_shape[1]
        self._update_pixel_polygons(fw, fh)

        self.violated_zones.clear()
        enriched_detections = []

        for det in tracked_detections:
            x1, y1, x2, y2 = det["bbox"]
            # Calculate drone center coordinate
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            in_zone = False
            detected_zone_name = "OUTSIDE"

            # Check center coordinate against each zone polygon
            for zone in self.zones:
                pixel_poly = zone.get("pixel_polygon", None)
                if pixel_poly is None:
                    continue

                # cv2.pointPolygonTest returns positive for inside, 0 for boundary, negative for outside
                dist = cv2.pointPolygonTest(pixel_poly, (float(cx), float(cy)), False)
                if dist >= 0:
                    in_zone = True
                    detected_zone_name = zone["name"]
                    self.violated_zones.add(zone["name"])
                    break  # Associate with the first matching restricted zone

            # Create enriched detection copy
            det_copy = dict(det)
            det_copy["in_zone"] = in_zone
            det_copy["zone_name"] = detected_zone_name
            enriched_detections.append(det_copy)

        return enriched_detections

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or len(self.zones) == 0 or frame is None or frame.size == 0:
            return frame

        annotated_frame = frame.copy()
        overlay = frame.copy()

        for zone in self.zones:
            pixel_poly = zone.get("pixel_polygon", None)
            if pixel_poly is None:
                continue

            zone_name = zone["name"]
            is_violated = zone_name in self.violated_zones
            
            # Violation highlights the boundary in highly visible Red
            color = (0, 0, 255) if is_violated else zone["color"]
            thickness = 3 if is_violated else 2

            # Draw semi-transparent filled zone region
            cv2.fillPoly(overlay, [pixel_poly], color)

            # Draw solid boundary line
            cv2.polylines(annotated_frame, [pixel_poly], isClosed=True, color=color, thickness=thickness)

            # Draw text label near topmost vertex of the polygon
            top_vertex_idx = np.argmin(pixel_poly[:, 1])
            tx = int(pixel_poly[top_vertex_idx][0])
            ty = int(pixel_poly[top_vertex_idx][1]) - 6

            label = f"{zone_name}"
            if is_violated:
                label += " - ALERT!"

            cv2.putText(
                annotated_frame,
                label,
                (tx, max(15, ty)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

        # Blend original frame and colored overlay
        # Alpha of 0.12 gives a soft tinted look on the security feed
        cv2.addWeighted(overlay, 0.12, annotated_frame, 0.88, 0, annotated_frame)
        return annotated_frame
