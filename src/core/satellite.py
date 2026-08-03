import time
import math
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Skyfield imports for real orbital mechanics
try:
    from skyfield.api import load, wgs84, EarthSatellite
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False
    logger.warning("skyfield not installed. Satellite tracking will be disabled. Install with: pip install skyfield")


class SatelliteTracker:
    """
    Real-time satellite tracking using TLE (Two-Line Element) data from CelesTrak
    and the Skyfield library for SGP4 orbital propagation.

    Computes actual lat/lon/alt/azimuth/elevation for each configured satellite
    relative to the observer's geographic position.
    """
    def __init__(self, config: dict):
        self.config = config.get("satellite", {})
        self.enabled = self.config.get("enabled", True)
        self.display_mode = self.config.get("display_mode", "overlay")
        self.position = self.config.get("position", "top-right")
        self.sat_configs = self.config.get("list", [])

        # Observer location coordinates (Default: Istanbul)
        self.obs_lat = self.config.get("observer_lat", 41.0082)
        self.obs_lon = self.config.get("observer_lon", 28.9784)

        # Performance: throttle TLE position recalculation
        self.update_interval = self.config.get("update_interval_sec", 1.0)
        self._last_update_time = 0.0
        self._cached_states = []

        # Size configuration for the dashboard layout
        self.width = 380
        self.height = 200
        self.radar_radius = 65

        # Initialize Skyfield components
        self._satellites = []  # list of (name, EarthSatellite) tuples
        self._observer = None
        self._ts = None
        self._init_ok = False

        if not SKYFIELD_AVAILABLE:
            logger.error("Skyfield is not available. Satellite panel will show no data.")
            self.enabled = False
            return

        try:
            self._ts = load.timescale()
            self._observer = wgs84.latlon(
                latitude_degrees=self.obs_lat,
                longitude_degrees=self.obs_lon,
                elevation_m=0.0
            )
            logger.info(f"Satellite observer position: {self.obs_lat:.4f}°N, {self.obs_lon:.4f}°E")
        except Exception as e:
            logger.error(f"Failed to initialize Skyfield timescale/observer: {e}")
            self.enabled = False
            return

        # Download TLE data for each configured satellite
        self._load_tle_data()

        if len(self._satellites) > 0:
            self._init_ok = True
            logger.info(f"Satellite tracker initialized with {len(self._satellites)} satellites (real TLE data).")
        else:
            logger.warning("No satellites loaded. Satellite panel will be empty.")

    def _load_tle_data(self):
        """
        Downloads TLE data from CelesTrak for each satellite by NORAD catalog number.
        Uses the GP API endpoint with TLE format.
        """
        import urllib.request
        import urllib.error

        for sat_cfg in self.sat_configs:
            name = sat_cfg.get("name", "UNKNOWN")
            norad_id = sat_cfg.get("norad_id")

            if norad_id is None:
                logger.warning(f"Satellite '{name}' has no norad_id, skipping.")
                continue

            url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
            try:
                logger.info(f"Fetching TLE for {name} (NORAD {norad_id})...")
                req = urllib.request.Request(url, headers={"User-Agent": "DroneDetect/1.0"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    tle_text = response.read().decode("utf-8").strip()

                lines = tle_text.splitlines()
                if len(lines) < 2:
                    logger.warning(f"Invalid TLE response for {name} (NORAD {norad_id}): not enough lines.")
                    continue

                # CelesTrak TLE format returns 3 lines: name, line1, line2
                if len(lines) >= 3:
                    tle_name = lines[0].strip()
                    tle_line1 = lines[1].strip()
                    tle_line2 = lines[2].strip()
                else:
                    # 2LE format (no name line)
                    tle_name = name
                    tle_line1 = lines[0].strip()
                    tle_line2 = lines[1].strip()

                # Validate TLE line format
                if not tle_line1.startswith("1 ") or not tle_line2.startswith("2 "):
                    logger.warning(f"Malformed TLE data for {name} (NORAD {norad_id}). Skipping.")
                    continue

                sat = EarthSatellite(tle_line1, tle_line2, tle_name, self._ts)
                self._satellites.append((name, sat))
                logger.info(f"  ✓ Loaded TLE for {name} (epoch: {sat.epoch.utc_strftime('%Y-%m-%d %H:%M UTC')})")

            except urllib.error.URLError as e:
                logger.warning(f"Network error fetching TLE for {name} (NORAD {norad_id}): {e}")
            except Exception as e:
                logger.warning(f"Failed to load TLE for {name} (NORAD {norad_id}): {e}")

    def get_satellite_states(self) -> list[dict]:
        """
        Computes real-time satellite positions using SGP4 orbital propagation.
        Results are cached and refreshed every `update_interval_sec` seconds.

        Returns a list of dicts with:
            name, lat, lon, alt (km), azimuth (deg), elevation (deg), visible (bool)
        """
        now = time.time()

        # Return cached results if within update interval
        if now - self._last_update_time < self.update_interval and self._cached_states:
            return self._cached_states

        if not self._init_ok or self._ts is None:
            return []

        t = self._ts.now()
        states = []

        for name, sat in self._satellites:
            try:
                # Compute geocentric position
                geocentric = sat.at(t)

                # Extract geographic coordinates (lat, lon, altitude)
                lat = wgs84.latlon_of(geocentric)
                alt = wgs84.height_of(geocentric)
                lat_deg = lat[0].degrees
                lon_deg = lat[1].degrees
                alt_km = alt.km

                # Compute topocentric position (azimuth/elevation from observer)
                difference = sat - self._observer
                topocentric = difference.at(t)
                el, az, distance = topocentric.altaz()

                elevation_deg = el.degrees
                azimuth_deg = az.degrees
                distance_km = distance.km

                # Satellite is visible if elevation > 0 (above horizon)
                is_visible = elevation_deg > 0.0

                states.append({
                    "name": name,
                    "lat": lat_deg,
                    "lon": lon_deg,
                    "alt": alt_km,
                    "azimuth": azimuth_deg,
                    "elevation": elevation_deg,
                    "distance_km": distance_km,
                    "visible": is_visible,
                })

            except Exception as e:
                logger.debug(f"Error computing position for {name}: {e}")
                states.append({
                    "name": name,
                    "lat": 0.0,
                    "lon": 0.0,
                    "alt": 0.0,
                    "azimuth": 0.0,
                    "elevation": -90.0,
                    "distance_km": 0.0,
                    "visible": False,
                })

        self._cached_states = states
        self._last_update_time = now
        return states

    def draw_dashboard(self, bg_color=(20, 20, 20), text_color=(255, 255, 255), radar_color=(0, 255, 0)) -> np.ndarray:
        """
        Generates a standalone frame containing the satellite tracker panel (solid background).
        Used when displaying in a separate OpenCV window.
        """
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = bg_color

        # Draw border
        cv2.rectangle(frame, (0, 0), (self.width - 1, self.height - 1), (50, 50, 50), 1)

        # Left Panel - Radar display
        rcx, rcy = 75, 100
        cv2.circle(frame, (rcx, rcy), self.radar_radius, (0, 100, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (rcx, rcy), int(self.radar_radius * 0.66), (0, 70, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (rcx, rcy), int(self.radar_radius * 0.33), (0, 45, 0), 1, cv2.LINE_AA)

        # Radar Crosshairs
        cv2.line(frame, (rcx - self.radar_radius, rcy), (rcx + self.radar_radius, rcy), (0, 70, 0), 1)
        cv2.line(frame, (rcx, rcy - self.radar_radius), (rcx, rcy + self.radar_radius), (0, 70, 0), 1)

        # Rotating radar sweep line
        sweep_angle = (time.time() * 2.0) % (2.0 * math.pi)
        sx = int(rcx + self.radar_radius * math.cos(sweep_angle))
        sy = int(rcy - self.radar_radius * math.sin(sweep_angle))
        cv2.line(frame, (rcx, rcy), (sx, sy), (0, 180, 0), 1, cv2.LINE_AA)

        # Label radar
        cv2.putText(frame, "SATELLITE SKYVIEW", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw Satellites on Radar
        states = self.get_satellite_states()
        for sat in states:
            if sat["visible"]:
                # Project satellite sky coordinates onto radar
                r = self.radar_radius * (90.0 - sat["elevation"]) / 90.0
                theta = math.radians(sat["azimuth"])
                sx = int(rcx + r * math.sin(theta))
                sy = int(rcy - r * math.cos(theta))

                if 0 <= sx < self.width and 0 <= sy < self.height:
                    cv2.circle(frame, (sx, sy), 3, (0, 255, 255), -1)  # Yellow blip
                    short_name = sat["name"].split(" ")[0][:8]
                    cv2.putText(frame, short_name, (sx + 4, sy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 250, 200), 1, cv2.LINE_AA)

        # Right Panel - Tabular listing of active systems
        start_x = 160
        cv2.putText(frame, "NAME", (start_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "LAT / LON", (start_x + 90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "ALT", (start_x + 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        y_offset = 42
        for sat in states:
            name = sat["name"][:12]
            coord_str = f"{sat['lat']:.2f} / {sat['lon']:.2f}"
            alt_str = f"{sat['alt']:.0f}km"

            # Differentiate visual color by status (visible = active)
            color = (255, 255, 255) if sat["visible"] else (120, 120, 120)

            cv2.putText(frame, name, (start_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            cv2.putText(frame, coord_str, (start_x + 90, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            cv2.putText(frame, alt_str, (start_x + 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            y_offset += 26
            if y_offset > self.height - 10:
                break

        return frame

    def overlay_on_frame(self, main_frame: np.ndarray) -> np.ndarray:
        """
        Overlays the tracker HUD directly onto the main frame in the configured position.
        Uses translucency for premium styling.
        """
        if not self.enabled:
            return main_frame

        fh, fw = main_frame.shape[:2]

        # Stop overlay if main frame size cannot accommodate the panel
        if fw < self.width + 20 or fh < self.height + 20:
            return main_frame

        # Determine panel coordinates
        if self.position == "top-left":
            x_start, y_start = 10, 10
        else:  # top-right
            x_start, y_start = fw - self.width - 10, 10

        # 1. Create a translucent background card
        sub_img = main_frame[y_start:y_start+self.height, x_start:x_start+self.width]
        overlay = sub_img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width - 1, self.height - 1), (15, 22, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.82, sub_img, 0.18, 0, sub_img)

        # 2. Draw active components on the card area
        # Radar circles
        rcx, rcy = 75, 100
        cv2.circle(sub_img, (rcx, rcy), self.radar_radius, (0, 130, 0), 1, cv2.LINE_AA)
        cv2.circle(sub_img, (rcx, rcy), int(self.radar_radius * 0.66), (0, 90, 0), 1, cv2.LINE_AA)
        cv2.circle(sub_img, (rcx, rcy), int(self.radar_radius * 0.33), (0, 55, 0), 1, cv2.LINE_AA)
        cv2.line(sub_img, (rcx - self.radar_radius, rcy), (rcx + self.radar_radius, rcy), (0, 90, 0), 1)
        cv2.line(sub_img, (rcx, rcy - self.radar_radius), (rcx, rcy + self.radar_radius), (0, 90, 0), 1)

        # Sweep line
        sweep_angle = (time.time() * 2.2) % (2.0 * math.pi)
        sx = int(rcx + self.radar_radius * math.cos(sweep_angle))
        sy = int(rcy - self.radar_radius * math.sin(sweep_angle))
        cv2.line(sub_img, (rcx, rcy), (sx, sy), (0, 240, 0), 1, cv2.LINE_AA)

        # Title text
        cv2.putText(sub_img, "SAT STATUS DISPLAY", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Satellite data calculations using real TLE data
        states = self.get_satellite_states()
        for sat in states:
            if sat["visible"]:
                r = self.radar_radius * (90.0 - sat["elevation"]) / 90.0
                theta = math.radians(sat["azimuth"])
                sx = int(rcx + r * math.sin(theta))
                sy = int(rcy - r * math.cos(theta))
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    cv2.circle(sub_img, (sx, sy), 3, (0, 255, 255), -1)  # Yellow blip
                    short_name = sat["name"].split(" ")[0][:8]
                    cv2.putText(sub_img, short_name, (sx + 4, sy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 255, 200), 1, cv2.LINE_AA)

        # Coordinates List Panel (Table columns)
        start_x = 160
        cv2.putText(sub_img, "SAT SYSTEM", (start_x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sub_img, "LAT / LON", (start_x + 90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sub_img, "ALT", (start_x + 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        y_offset = 42
        for sat in states:
            name = sat["name"][:12]
            coord_str = f"{sat['lat']:.2f}/{sat['lon']:.2f}"
            alt_str = f"{sat['alt']:.0f}km"

            # Active (White) vs. Inactive (Gray) colors for table rows
            color = (255, 255, 255) if sat["visible"] else (110, 110, 110)

            cv2.putText(sub_img, name, (start_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            cv2.putText(sub_img, coord_str, (start_x + 90, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            cv2.putText(sub_img, alt_str, (start_x + 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            # Elevation indicator bar (higher elevation = longer bar)
            el = max(0.0, sat["elevation"])
            bar_w = int(30 * (el / 90.0))
            cv2.rectangle(sub_img, (start_x + 180, y_offset + 4), (start_x + 210, y_offset + 7), (40, 40, 40), -1)
            # Green if high elevation, yellow if medium, red if low/below horizon
            if sat["visible"]:
                bar_color = (0, 255, 0) if el >= 45 else ((0, 255, 255) if el >= 15 else (0, 140, 255))
            else:
                bar_color = (0, 0, 180)
                bar_w = 2  # minimal indicator for below-horizon
            cv2.rectangle(sub_img, (start_x + 180, y_offset + 4), (start_x + 180 + bar_w, y_offset + 7), bar_color, -1)

            y_offset += 28
            if y_offset > self.height - 10:
                break

        # Outer green card border
        cv2.rectangle(sub_img, (0, 0), (self.width - 1, self.height - 1), (0, 255, 0), 1, cv2.LINE_AA)

        return main_frame
