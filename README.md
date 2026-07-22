# Drone Detection, Tracking & Geofencing System

A modular, production-ready, cross-platform Computer Vision application built with **Python**, **YOLOv8**, **PyTorch**, and **OpenCV**. The system provides real-time drone detection, multi-object tracking, geofence zone monitoring, automatic debounced event snapshots, and daily structured audit logs.

---

## 🌟 Features

- **Object Detection**: Core YOLOv8 inference with optional **SAHI (Slicing Aided Hyper Inference)** for small object/long-range drone detection.
- **Multi-Object Tracking**: Hungarian matching algorithm based on Intersection-over-Union (IoU) to maintain persistent track IDs and movement trail trajectories.
- **Geofencing & Alarm Zones**: Defines restricted polygon areas, checks for drone violations, and updates display bounding boxes to Red upon zone entry.
- **Translucent UI & Warning Banners**: Annotates video feeds with zone outlines, motion trails, a dashboard status overlay, and a high-visibility flashing warning banner.
- **Debounced Snapshots**: Saves debounced target JPEG snapshots to daily subfolders (`captures/YYYY-MM-DD/`) when confidence requirements are met.
- **Event Audit Logs**: Appends structured JSON Lines (JSONL) records (`logs/drone_events_YYYY-MM-DD.jsonl`) detailing timestamps, track IDs, box coordinates, and snapshot paths.
- **Model Compilation**: Compiles `.pt` PyTorch models to **ONNX** or **TensorRT (.engine)** formats for maximum GPU inference frame rates.

---

## 📂 Folder Structure

```directory
drone-detect/
│
├── config/
│   └── config.yaml          # Hyperparameters, paths, geofences, and visual layouts
│
├── src/
│   ├── core/
│   │   ├── detector.py      # Core DroneDetector class (YOLO + SAHI)
│   │   ├── tracker.py       # DroneTracker class (Hungarian match + trails)
│   │   └── zone_logic.py    # GeofenceManager (Polygon violations + overlays)
│   │
│   ├── utils/
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

Manage hyperparameters, geofence polygons, and rendering layouts in `config/config.yaml`:
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
  tracker_type: "bytetrack"
  track_thresh: 0.5
  draw_trail: true
  trail_length: 30

# Restricted Geofencing Zones
geofencing:
  enabled: true
  zones:
    - name: "Restricted_Zone_1"
      polygon: [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]] # Normalized coordinates
      color: [0, 0, 255]          # Zone outline BGR color
```
