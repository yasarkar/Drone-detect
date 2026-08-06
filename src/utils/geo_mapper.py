"""
GeoMapper: Converts pixel coordinates of detected targets into real-world
geographic coordinates (WGS84 lat/lon) for a fixed/static camera setup.

Approach (monocular ground-plane intersection):

    1. Pixel (u, v)  ->  normalized camera ray via horizontal/vertical FOV
    2. Camera ray    ->  ENU frame using camera heading (yaw), pitch, roll
    3. ENU ray       ->  intersect with the configured target altitude plane
                         (target_altitude_amsl_m) to obtain North/East offsets
    4. ENU offsets   ->  WGS84 lat/lon via equirectangular approximation
                         around the camera GPS position

Notes:
    - Monocular depth is ambiguous in general; this module assumes the target
      flies at the configured `target_altitude_amsl_m` altitude plane.
    - If the ray points away from the target plane the result is None.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Mean Earth radius in meters (WGS84 semi-major axis)
EARTH_RADIUS_M = 6378137.0


class GeoMapper:
    """
    Maps pixel coordinates to WGS84 latitude/longitude for a static camera.
    """

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

        gm_config = self.config.get("geo_mapping", {})
        self.enabled = gm_config.get("enabled", False)

        cam_config = gm_config.get("camera", {})
        self.cam_lat = float(cam_config.get("latitude", 41.0082))
        self.cam_lon = float(cam_config.get("longitude", 28.9784))
        self.cam_alt_m = float(cam_config.get("altitude_m", 5.0))
        self.heading_deg = float(cam_config.get("heading_deg", 0.0))
        self.pitch_deg = float(cam_config.get("pitch_deg", -10.0))
        self.roll_deg = float(cam_config.get("roll_deg", 0.0))
        self.fov_h_deg = float(cam_config.get("fov_h_deg", 70.0))

        ref_size = cam_config.get("reference_size", [1280, 720])
        if isinstance(ref_size, (list, tuple)) and len(ref_size) == 2:
            self.ref_w = int(ref_size[0])
            self.ref_h = int(ref_size[1])
        else:
            self.ref_w, self.ref_h = 1280, 720

        # Optional: derive vertical FOV from horizontal FOV + aspect ratio
        self.fov_v_deg = float(cam_config.get("fov_v_deg", 0.0) or 0.0)
        if self.fov_v_deg <= 0.0:
            self.fov_v_deg = math.degrees(
                2.0 * math.atan(
                    math.tan(math.radians(self.fov_h_deg) / 2.0) * (self.ref_h / self.ref_w)
                )
            )

        self.target_alt_amsl_m = float(gm_config.get("target_altitude_amsl_m", 0.0))
        self.show_on_screen = gm_config.get("show_on_screen", True)

        # Precompute reference focal lengths (pixels) at the reference resolution
        self.ref_fx = (self.ref_w / 2.0) / math.tan(math.radians(self.fov_h_deg) / 2.0)
        self.ref_fy = (self.ref_h / 2.0) / math.tan(math.radians(self.fov_v_deg) / 2.0)

        # Precompute camera -> ENU rotation matrix at init (static camera)
        self._rotation = self._build_rotation_matrix()
        self._inv_rotation = self._rotation.T

        logger.info(
            f"GeoMapper initialized: "
            f"cam=({self.cam_lat:.5f}, {self.cam_lon:.5f}, {self.cam_alt_m:.1f}m) "
            f"heading={self.heading_deg:.1f}° pitch={self.pitch_deg:.1f}° "
            f"roll={self.roll_deg:.1f}° fov=({self.fov_h_deg:.1f}x{self.fov_v_deg:.1f})° "
            f"target_alt={self.target_alt_amsl_m:.1f}m"
        )

    # ------------------------------------------------------------------ #
    # Spatial helpers
    # ------------------------------------------------------------------ #

    def _build_rotation_matrix(self) -> np.ndarray:
        """
        Builds the 3x3 rotation matrix that transforms camera-frame unit
        vectors (x=right, y=down, z=forward) into the local ENU frame
        (E=x, N=y, U=z).

        The camera boresight is derived from heading (azimuth measured
        clockwise from North) and pitch (negative = looking downward).
        """
        az = math.radians(self.heading_deg)
        # Negative pitch means looking down; positive pitch means looking up.
        # The elevation angle of the boresight equals the pitch directly.
        elev = math.radians(self.pitch_deg)

        # Forward (boresight) unit vector in ENU coordinates
        forward = np.array([
            math.sin(az) * math.cos(elev),
            math.cos(az) * math.cos(elev),
            math.sin(elev),
        ])

        # Camera-right horizontal vector in ENU coordinates
        right = np.array([
            math.cos(az),
            -math.sin(az),
            0.0,
        ])

        # Camera-down vector: for an orthonormal frame, right x down = forward,
        # so down = forward x right.
        down = np.cross(forward, right)

        # Normalize for safety (roll handled below rotates these basis vectors)
        right = right / np.linalg.norm(right)
        down = down / np.linalg.norm(down)
        forward = forward / np.linalg.norm(forward)

        # Apply roll rotation around the camera forward axis (x_c stays right,
        # y_c stays down; rotating by roll_deg).
        if abs(self.roll_deg) > 1e-9:
            roll = math.radians(self.roll_deg)
            cos_r, sin_r = math.cos(roll), math.sin(roll)
            # Rotate the (right, down) basis vectors about the forward axis
            right_new = cos_r * right + sin_r * down
            down_new = -sin_r * right + cos_r * down
            right, down = right_new, down_new

        # Columns of R are the camera basis expressed in ENU coordinates
        R = np.column_stack((right, down, forward))
        return R

    def _effective_focal(self, frame_w: int, frame_h: int) -> tuple[float, float]:
        """Scales reference focal lengths to the actual frame resolution."""
        scale_x = frame_w / self.ref_w
        scale_y = frame_h / self.ref_h
        return self.ref_fx * scale_x, self.ref_fy * scale_y

    def _ray_in_enu(self, u: float, v: float, frame_w: int, frame_h: int) -> np.ndarray | None:
        """
        Converts a pixel to a camera ray expressed in the local ENU frame.
        Returns a direction vector or None when the pixel is outside the frame.
        """
        if u < 0 or v < 0 or u > frame_w or v > frame_h:
            return None

        fx, fy = self._effective_focal(frame_w, frame_h)
        cx, cy = frame_w / 2.0, frame_h / 2.0

        # Normalized camera coordinates (pinhole projection)
        n_x = (u - cx) / fx
        n_y = (v - cy) / fy

        # Camera frame ray: x right, y down, z forward
        d_cam = np.array([n_x, n_y, 1.0], dtype=float)

        # Transform to ENU: R * d_cam
        d_enu = self._rotation @ d_cam
        return d_enu

    def pixel_to_world(
        self,
        u: float,
        v: float,
        frame_w: int,
        frame_h: int,
        target_alt_amsl: float | None = None,
    ) -> dict | None:
        """
        Converts a pixel coordinate into real-world WGS84 coordinates.

        Args:
            u, v: Pixel coordinates (image center of the target).
            frame_w, frame_h: Current frame resolution.
            target_alt_amsl: Target altitude above mean sea level. Defaults to
                the configured ``target_altitude_amsl_m``.

        Returns:
            dict with keys {lat, lon, alt_amsl, dist_m, bearing_deg} or None
            when the ray does not intersect the target altitude plane.
        """
        if not self.enabled:
            return None

        if target_alt_amsl is None:
            target_alt_amsl = self.target_alt_amsl_m

        d_enu = self._ray_in_enu(u, v, frame_w, frame_h)
        if d_enu is None:
            return None

        d_e, d_n, d_u = d_enu

        # Relative altitude of target plane w.r.t. camera (up positive)
        delta_h = target_alt_amsl - self.cam_alt_m

        # Ray-plane intersection: t > 0 ensures the plane is ahead of the camera
        if abs(d_u) < 1e-12:
            return None
        t = delta_h / d_u
        if t <= 0.0:
            return None

        east_m = t * d_e
        north_m = t * d_n
        up_m = t * d_u  # should equal delta_h

        dist_m = math.sqrt(east_m * east_m + north_m * north_m + up_m * up_m)
        ground_dist_m = math.hypot(east_m, north_m)
        bearing_deg = math.degrees(math.atan2(east_m, north_m)) % 360.0

        # Equirectangular approximation around the camera GPS position
        lat_rad = math.radians(self.cam_lat)
        d_lat = north_m / EARTH_RADIUS_M
        d_lon = east_m / (EARTH_RADIUS_M * math.cos(lat_rad))

        lat = self.cam_lat + math.degrees(d_lat)
        lon = self.cam_lon + math.degrees(d_lon)

        return {
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "alt_amsl": round(float(target_alt_amsl), 2),
            "dist_m": round(dist_m, 2),
            "ground_dist_m": round(ground_dist_m, 2),
            "bearing_deg": round(bearing_deg, 2),
        }

    def world_to_pixel(
        self,
        lat: float,
        lon: float,
        alt_amsl: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float] | None:
        """
        Inverse mapping (for testing/debug): WGS84 -> pixel.

        Returns (u, v) when the point is in front of the camera, else None.
        """
        cam_lat_rad = math.radians(self.cam_lat)

        # ENU deltas (DEGREES -> RADIANS before multiplying by Earth radius)
        d_n = math.radians(lat - self.cam_lat) * EARTH_RADIUS_M
        d_e = math.radians(lon - self.cam_lon) * EARTH_RADIUS_M * math.cos(cam_lat_rad)
        d_u = alt_amsl - self.cam_alt_m

        d_enu = np.array([d_e, d_n, d_u], dtype=float)

        # Transform ENU -> camera frame
        d_cam = self._inv_rotation @ d_enu
        x_c, y_c, z_c = d_cam

        # Point must be in front of the camera
        if z_c <= 0.0:
            return None

        fx, fy = self._effective_focal(frame_w, frame_h)
        cx, cy = frame_w / 2.0, frame_h / 2.0

        u = fx * (x_c / z_c) + cx
        v = fy * (y_c / z_c) + cy

        # Clamp check: if the point maps outside the frame return None
        if u < 0 or v < 0 or u >= frame_w or v >= frame_h:
            return None

        return u, v