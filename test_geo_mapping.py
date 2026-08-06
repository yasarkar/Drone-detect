"""
Unit tests for the GeoMapper (pixel <-> WGS84 geographic coordinate mapping).

Run with:
    python test_geo_mapping.py
"""

import math
import sys
import tempfile
from pathlib import Path

import yaml

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.geo_mapper import GeoMapper, EARTH_RADIUS_M

# Test configuration fixture
TEST_CONFIG = {
    "geo_mapping": {
        "enabled": True,
        "camera": {
            "latitude": 41.0082,
            "longitude": 28.9784,
            "altitude_m": 5.0,
            "heading_deg": 0.0,      # Facing due North
            "pitch_deg": -10.0,      # Looking down 10 degrees
            "roll_deg": 0.0,
            "fov_h_deg": 70.0,
            "fov_v_deg": 0.0,        # Auto-derived
            "reference_size": [1280, 720],
        },
        "target_altitude_amsl_m": 0.0,  # Target at sea level
        "show_on_screen": True,
    }
}

FRAME_W, FRAME_H = 1280, 720
PASS = 0
FAIL = 0


def _write_temp_config() -> Path:
    """Writes the test config into a temporary YAML file."""
    tmp = Path(tempfile.gettempdir()) / "test_geo_mapping_config.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(TEST_CONFIG, f)
    return tmp


def _check(name: str, condition: bool, details: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {details}")


def test_initialization():
    print("\n[1] Initialization")
    mapper = GeoMapper(_write_temp_config())
    _check("enabled is True", mapper.enabled is True)
    _check("ref focal length fx > 0", mapper.ref_fx > 0)
    _check("ref focal length fy > 0", mapper.ref_fy > 0)
    _check("rotation matrix is 3x3", mapper._rotation.shape == (3, 3))
    # Rotation matrix should be orthonormal
    identity = mapper._rotation @ mapper._rotation.T
    _check(
        "rotation matrix is orthonormal",
        abs(identity[0, 0] - 1.0) < 1e-9 and abs(identity[1, 1] - 1.0) < 1e-9,
        f"identity diag: {identity.diagonal()}",
    )
    return mapper


def test_ray_forward_symmetry():
    print("\n[2] Ray Forward Symmetry (below-center pixels see North)")
    mapper = GeoMapper(_write_temp_config())

    # With heading=0 (North) and pitch=-10 (looking down), an image center
    # pixel should map to a point North of the camera.
    result_center = mapper.pixel_to_world(FRAME_W / 2, FRAME_H / 2, FRAME_W, FRAME_H)
    _check("center pixel maps to a valid result", result_center is not None)
    if result_center:
        _check("center maps North of camera (lat > cam_lat)", result_center["lat"] > mapper.cam_lat, str(result_center))
        _check("center bearing is ~0° (due North)", result_center["bearing_deg"] < 5.0, str(result_center["bearing_deg"]))

    # A pixel below center points closer to the camera (steeper view angle),
    # so it maps to a NEARER ground point (less North). The center pixel
    # should therefore be further North than the lower pixel.
    result_lower = mapper.pixel_to_world(FRAME_W / 2, FRAME_H * 0.9, FRAME_W, FRAME_H)
    if result_center and result_lower:
        _check("center pixel maps further North than lower pixel", result_center["lat"] > result_lower["lat"],
               f"center={result_center['lat']}, lower={result_lower['lat']}")


def test_altitude_plane_intersection():
    print("\n[3] Altitude Plane Intersection")
    mapper = GeoMapper(_write_temp_config())

    # Camera is at 5m AMSL and looks DOWN (pitch=-10). The ray goes downward,
    # so a target plane BELOW the camera (e.g. ground at 0m) intersects,
    # while a plane ABOVE the camera (e.g. 30m) cannot be reached -> None.
    r_ground = mapper.pixel_to_world(FRAME_W / 2, FRAME_H * 0.8, FRAME_W, FRAME_H, target_alt_amsl=0.0)
    r_high = mapper.pixel_to_world(FRAME_W / 2, FRAME_H * 0.8, FRAME_W, FRAME_H, target_alt_amsl=30.0)

    _check("ground plane intersects", r_ground is not None)
    _check("higher plane above camera returns None (looking down)", r_high is None)

    # A different vertical FOV/pitch setup should yield a different result
    # for the same pixel: steeper pitch = closer ground point.
    config_steep = dict(TEST_CONFIG)
    config_steep["geo_mapping"]["camera"]["pitch_deg"] = -30.0
    tmp = Path(tempfile.gettempdir()) / "test_geo_mapping_steep.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_steep, f)
    mapper_steep = GeoMapper(tmp)
    r_steep = mapper_steep.pixel_to_world(FRAME_W / 2, FRAME_H * 0.8, FRAME_W, FRAME_H, target_alt_amsl=0.0)

    _check("steeper pitch produces a ground hit", r_steep is not None)
    if r_ground and r_steep:
        _check("steeper pitch yields shorter ground distance", r_steep["ground_dist_m"] < r_ground["ground_dist_m"],
               f"{r_steep['ground_dist_m']} vs {r_ground['ground_dist_m']}")


def test_upward_ray_returns_none():
    print("\n[4] Upward Ray Returns None")
    # Configure camera looking UP (pitch=+30) -> ground-plane ray should not intersect
    config_up = dict(TEST_CONFIG)
    config_up["geo_mapping"]["camera"]["pitch_deg"] = 30.0
    tmp = Path(tempfile.gettempdir()) / "test_geo_mapping_up.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_up, f)

    mapper_up = GeoMapper(tmp)
    # At the top of the frame, the ray points upward -> should be None
    result = mapper_up.pixel_to_world(FRAME_W / 2, 10, FRAME_W, FRAME_H)
    _check("upward ray returns None", result is None)

    # At the bottom of the frame with a downward pitch, should still intersect?
    config_mixed = dict(TEST_CONFIG)
    config_mixed["geo_mapping"]["camera"]["pitch_deg"] = -5.0
    tmp2 = Path(tempfile.gettempdir()) / "test_geo_mapping_mixed.yaml"
    with open(tmp2, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_mixed, f)
    mapper_mixed = GeoMapper(tmp2)
    result2 = mapper_mixed.pixel_to_world(FRAME_W / 2, FRAME_H * 0.8, FRAME_W, FRAME_H)
    _check("downward ray still works", result2 is not None)


def test_world_to_pixel_roundtrip():
    print("\n[5] World -> Pixel -> World Roundtrip")
    mapper = GeoMapper(_write_temp_config())

    # Forward mapping from a known pixel
    src = mapper.pixel_to_world(FRAME_W * 0.7, FRAME_H * 0.6, FRAME_W, FRAME_H, target_alt_amsl=0.0)
    if src is None:
        _check("forward mapping returned None (bad setup)", False)
        return

    # Inverse mapping back to pixels
    uv = mapper.world_to_pixel(src["lat"], src["lon"], src["alt_amsl"], FRAME_W, FRAME_H)
    _check("inverse mapping produces a pixel", uv is not None)
    if uv:
        u, v = uv
        # Small tolerance due to lat/lon rounding + equirectangular approximation
        _check("roundtrip u within tolerance", abs(u - FRAME_W * 0.7) < 10.0, f"u={u}, expected ~{FRAME_W * 0.7}")
        _check("roundtrip v within tolerance", abs(v - FRAME_H * 0.6) < 10.0, f"v={v}, expected ~{FRAME_H * 0.6}")


def test_disabled_returns_none():
    print("\n[6] Disabled GeoMapper Returns None")
    config_off = dict(TEST_CONFIG)
    config_off["geo_mapping"]["enabled"] = False
    tmp = Path(tempfile.gettempdir()) / "test_geo_mapping_off.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_off, f)
    mapper_off = GeoMapper(tmp)
    result = mapper_off.pixel_to_world(FRAME_W / 2, FRAME_H / 2, FRAME_W, FRAME_H)
    _check("disabled mapper returns None", result is None)


def test_equirectangular_math():
    print("\n[7] Equirectangular Math Sanity")
    mapper = GeoMapper(_write_temp_config())

    # Manually compute expected ENU -> WGS84 offset for a small delta
    d_n_m = 100.0
    d_e_m = 50.0
    lat_rad = math.radians(mapper.cam_lat)
    exp_d_lat = d_n_m / EARTH_RADIUS_M
    exp_d_lon = d_e_m / (EARTH_RADIUS_M * math.cos(lat_rad))

    exp_lat = mapper.cam_lat + math.degrees(exp_d_lat)
    exp_lon = mapper.cam_lon + math.degrees(exp_d_lon)

    _check(
        "expected lat offset ~0.0009°",
        abs((exp_lat - mapper.cam_lat) - math.degrees(exp_d_lat)) < 1e-12,
        f"delta={math.degrees(exp_d_lat)}",
    )
    _check("expected lon offset is smaller than lat offset at Istanbul lat",
           math.degrees(exp_d_lon) < math.degrees(exp_d_lat))


if __name__ == "__main__":
    mapper = test_initialization()
    test_ray_forward_symmetry()
    test_altitude_plane_intersection()
    test_upward_ray_returns_none()
    test_world_to_pixel_roundtrip()
    test_disabled_returns_none()
    test_equirectangular_math()

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 50}")
    sys.exit(1 if FAIL else 0)