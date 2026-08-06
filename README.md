# Drone Detection & Tracking System

A modular, production-ready, cross-platform Computer Vision application built with **Python**, **YOLOv8**, **PyTorch**, and **OpenCV**. The system provides real-time drone detection, multi-object tracking, automatic debounced event snapshots, real-time satellite skyview panels, and daily structured audit logs.

---

## 🌟 Features

- **Object Detection**: Core YOLOv8 inference with optional **SAHI (Slicing Aided Hyper Inference)** for small object/long-range drone detection.
- **Multi-Object Tracking**: Hungarian matching algorithm based on Intersection-over-Union (IoU) to maintain persistent track IDs and movement trail trajectories.
- **Real-Time Satellite HUD**: Real-time TLE orbital mechanics tracking for overhead satellites via Skyfield.
- **Debounced Snapshots**: Saves debounced target JPEG snapshots to daily subfolders (`captures/YYYY-MM-DD/`) when confidence requirements are met.
- **Event Audit Logs**: Appends structured JSON Lines (JSONL) records (`logs/drone_events_YYYY-MM-DD.jsonl`) detailing timestamps, track IDs, box coordinates, and snapshot paths.
- **Model Compilation**: Compiles `.pt` PyTorch models to **ONNX** or **TensorRT (.engine)** formats for maximum GPU inference frame rates.

---

## 📂 Folder Structure

```directory
drone-detect/
│
├── config/
│   └── config.yaml          # Hyperparameters, paths, and visual layouts
│
├── src/
│   ├── core/
│   │   ├── detector.py      # Core DroneDetector class (YOLO + SAHI)
│   │   ├── tracker.py       # DroneTracker class (Hungarian match + trails)
│   │   └── satellite.py     # SatelliteTracker (Orbital mechanics TLE tracker)
│   │
│   ├── utils/
│   │   ├── geo_mapper.py    # GeoMapper class (pixel -> WGS84 lat/lon)
│   │   ├── logger.py        # AuditLogger class (daily structured logs)
│   │   └── snapshot.py      # SnapshotManager class (debounced images)
│   │
│   └── pipeline.py          # Unified DronePipeline orchestration
│
├── weights/                 # Compiled and raw model weights (e.g. best.pt, best.engine)
├── captures/                # Snapshot output directories (saves captures/YYYY-MM-DD/)
├── logs/                    # Event audit logs directories (saves drone_events_YYYY-MM-DD.jsonl)
│
├── requirements.txt         # Project package requirements
├── train.py                 # YOLOv8 model training script
├── export.py                # Performance ONNX/TensorRT compilation script
├── main.py                  # CLI application entrypoint
└── README.md                # Documentation
```

---

## 🛠️ Installation & Setup

1. **Clone or navigate** to the project workspace:
   ```bash
   cd drone-detect
   ```

2. **Initialize a virtual environment** and install the dependencies:
   Using `uv` (recommended for speed):
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```
   Or standard `venv` + `pip`:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\pip install -r requirements.txt
   # Linux/macOS:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 🚀 Execution & Usage Guide

### 1. Training the Model
Train a YOLO model on your custom dataset by configuring the dataset yaml path in `config/config.yaml` and running:
```bash
python train.py
```
This automatically saves the best trained weights to `weights/best.pt`.

### 2. Model Compilation (Performance Export)
Compile the trained model to ONNX or TensorRT (.engine) format to achieve maximum GPU inference speed (TensorRT is recommended for NVIDIA GPUs):
```bash
# Export to TensorRT (.engine) format
python export.py --weights weights/best.pt --format engine --imgsz 640

# Export to ONNX format
python export.py --weights weights/best.pt --format onnx --imgsz 640
```
*Note: Update `weights_path` in `config/config.yaml` to point to the newly generated `weights/best.engine` or `weights/best.onnx` file.*

### 3. Run Inference Pipeline
Run the main tracking application using the unified pipeline CLI:

- **Webcam feed (device index 0)**:
  ```bash
  python main.py --source 0
  ```

- **Video file input**:
  ```bash
  python main.py --source path/to/drone_footage.mp4
  ```

- **Headless mode** (runs detection, writes event logs, saves snapshots without showing the GUI window):
  ```bash
  python main.py --source path/to/drone_footage.mp4 --no-display
  ```

- **Save annotated output video** (writes output file to `captures/output_video.mp4`):
  ```bash
  python main.py --source path/to/drone_footage.mp4 --save-output
  ```

---

## ⚙️ Configuration File (`config/config.yaml`)

Manage hyperparameters and rendering layouts in `config/config.yaml`:
```yaml
# Dataset configuration
data_yaml_path: "dataset/data.yaml"

# Hardware configurations
device: 0                        # GPU index (0, 1) or "cpu"

# Inference configurations
weights_path: "weights/best.pt"  # Load .pt, .onnx, or .engine compiled weights
confidence_threshold: 0.50       # YOLO prediction confidence threshold
iou_threshold: 0.45              # NMS Intersection-over-Union threshold

# SAHI settings
sahi:
  enabled: true                  # Set to true to enable sliced inference
  slice_height: 512
  slice_width: 512

# Tracking settings
tracking:
  enabled: true
  tracker_type: "hungarian_iou"
  track_thresh: 0.5
  draw_trail: false
  trail_length: 30
```

---

## 🗺️ Real-World GPS Coordinate Mapping (GeoMapper)

The system can convert detected target **pixel coordinates** into **real-world WGS84 latitude/longitude** using a static (fixed) camera setup.

### How It Works

Monocular cameras cannot measure absolute depth from a single image. Instead, **GeoMapper** uses the **ground/target-plane intersection** method:

1. Pixel `(u, v)` is converted into a normalized camera ray using the configured horizontal/vertical FOV.
2. The ray is transformed into a local **ENU** (East-North-Up) frame using the camera's `heading_deg`, `pitch_deg`, and `roll_deg`.
3. The ray is intersected with the configured **target altitude plane** (`target_altitude_amsl_m`) to compute North/East offsets in meters.
4. The offsets are converted to WGS84 lat/lon using an equirectangular approximation around the camera GPS position.

### Configuration

```yaml
geo_mapping:
  enabled: true
  camera:
    latitude: 41.0082         # Camera GPS latitude (degrees)
    longitude: 28.9784        # Camera GPS longitude (degrees)
    altitude_m: 5.0           # Camera height above mean sea level (m)
    heading_deg: 0.0          # Yaw: azimuth clockwise from North (0 = due North)
    pitch_deg: -10.0          # Negative = looking downward, positive = looking upward
    roll_deg: 0.0             # Camera roll
    fov_h_deg: 70.0           # Horizontal field of view (degrees)
    fov_v_deg: 0.0            # Vertical FOV; 0 = auto-derived from aspect ratio
    reference_size: [1280, 720]  # Resolution the FOV values were measured at
  target_altitude_amsl_m: 0.0 # Assumed target (drone) altitude above mean sea level
  show_on_screen: true        # Render Lat/Lon below each bounding box
```

### Interactive Calibration Tool

A helper script walks you through each parameter step-by-step and prints a ready-to-paste YAML block:

```bash
python tools/calibrate.py
```

It prompts for:
- Camera GPS position (lat / lon / altitude)
- Heading (yaw) — measured with a compass app
- Pitch (camera tilt; negative = looking down)
- FOV — either entered directly or **measured automatically** with the built-in distance/width dialog
- Reference resolution
- Target (drone) altitude (region elevation + typical drone flight height)

Run the integrated unit tests to verify the mathematics after calibration:

```bash
python test_geo_mapping.py
```

### Calibration Checklist

To get accurate results you must measure/configure these values for your physical camera:

| Parameter | How to measure |
|-----------|----------------|
| `latitude` / `longitude` | GPS position of the camera location (e.g. phone compass/GPS app). |
| `altitude_m` | Camera height above sea level (altimeter app or survey data). |
| `heading_deg` | Compass bearing of the camera boresight (0 = North, 90 = East). |
| `pitch_deg` | Tilt angle; negative when the camera looks down at the scene. |
| `fov_h_deg` | Horizontal field of view of the lens (lens datasheet or FOV calculator). |
| `target_altitude_amsl_m` | The cruising altitude of the drones you are tracking (e.g. 120 m for many small UAS). |

### Output

- **On-screen**: `Lat:xx.xxxxx Lon:xx.xxxxx` rendered in orange below each bounding box (when `show_on_screen: true`).
- **JSONL audit logs** (`logs/drone_events_YYYY-MM-DD.jsonl`): each record now includes a `geo` object:
  ```json
  "geo": {
    "lat": 41.0123456,
    "lon": 28.9876543,
    "alt_amsl": 0.0,
    "dist_m": 512.34,
    "ground_dist_m": 510.12,
    "bearing_deg": 15.67
  }
  ```
- **Daily Markdown reports** (`logs/drone_report_YYYY-MM-DD.md`): each track now shows **Initial GPS Location**, **Final GPS Location**, and (when multiple GPS samples exist) a collapsible **GPS Trajectory table** with timestamp, lat, lon, altitude, distance, and bearing.

### Limitations

- The result assumes the target is flying exactly at the configured `target_altitude_amsl_m`. If the drone is higher/lower, the lat/lon will be offset.
- Lens distortion is not corrected in the current version.
- This module is designed for **static cameras**. PTZ/gimbal operation requires updating heading/pitch per frame.

