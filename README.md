# Motion Capture System - Module Documentation

A dual-sensor motion capture system that combines Bluetooth Low Energy (BLE) IMU data and real-time pose estimation from video to collect synchronized motion data.

## Overview

This system consists of three main modules working together to capture motion data from both inertial sensors and computer vision:

- **IMUClient**: Collects accelerometer and gyroscope data via Bluetooth
- **CameraClient**: Captures pose keypoints (shoulder, elbow, wrist) from video
- **MainPy**: Orchestrates both clients and provides a unified control interface

---

## Module Descriptions

### 1. `imuclient.py` - IMU Data Collection

**Purpose**: Connects to an Arduino Nano 33 BLE device and collects IMU (Inertial Measurement Unit) sensor data via Bluetooth Low Energy.

**Key Features**:
- Scans and connects to BLE device by name (`Nano33BLE`) with fallback to MAC address
- Receives and logs accelerometer (ax, ay, az) and gyroscope (gx, gy, gz) readings
- Records data with precise timestamps (both PC and Arduino time)
- Data collection can be started/stopped via commands

**Class: `IMUClient`**

| Method | Description |
|--------|-------------|
| `__init__()` | Initializes CSV file for IMU data storage with header row |
| `handler(sender, data)` | BLE notification callback that processes incoming IMU data |
| `start()` | Async method to scan, connect to BLE device, and enable notifications |
| `send_start()` | Async method to command Arduino to start collecting data |
| `send_stop()` | Async method to command Arduino to stop collecting data |
| `stop()` | Async method to disconnect and close the data file |

**Output CSV Columns**:
- `pc_time`: Timestamp on PC when data was received
- `arduino_time`: Timestamp from Arduino sensor
- `ax, ay, az`: Accelerometer readings (m/s²)
- `gx, gy, gz`: Gyroscope readings (rad/s)

**Hardware Requirements**:
- Arduino Nano 33 BLE with IMU capabilities
- BLE GATT services configured with:
  - IMU data characteristic UUID: `19B10001-E8F2-537E-4F6C-D104768A1214`
  - Control command characteristic UUID: `19B10002-E8F2-537E-4F6C-D104768A1214`

---

### 2. `cameraclient.py` - Pose Detection

**Purpose**: Captures video from webcam, detects human pose keypoints using MediaPipe, and records the right arm joints (shoulder, elbow, wrist) positions.

**Key Features**:
- Real-time video capture from system webcam
- MediaPipe Pose detection for full-body skeleton tracking
- Extracts right shoulder, elbow, and wrist coordinates
- Records 3D position (x, y, z) and detection confidence (visibility)
- Threaded operation for non-blocking execution
- Interactive video display with early exit option (press 'q')

**Class: `CameraClient`**

| Method | Description |
|--------|-------------|
| `__init__()` | Initializes webcam capture, pose detector, and CSV output file |
| `start()` | Enables data collection (does not start processing loop) |
| `stop()` | Disables data collection but continues video display |
| `shutdown()` | Stops processing loop, releases resources, closes files |
| `run()` | Main loop that captures frames, detects poses, and logs data |

**Output CSV Columns**:
- `frame`: Frame number in capture sequence
- `timestamp`: Precise time when frame was captured (perf_counter)
- `shoulder_x, shoulder_y, shoulder_z`: Right shoulder position (normalized 0-1)
- `shoulder_visibility`: Confidence of shoulder detection (0-1)
- `elbow_x, elbow_y, elbow_z`: Right elbow position (normalized 0-1)
- `elbow_visibility`: Confidence of elbow detection (0-1)
- `wrist_x, wrist_y, wrist_z`: Right wrist position (normalized 0-1)
- `wrist_visibility`: Confidence of wrist detection (0-1)

**Dependencies**:
- OpenCV (`cv2`) - Video capture and display
- MediaPipe (`mediapipe`) - Pose detection (v0.11.0+ for NumPy 2.x compatibility)
- NumPy 2.2.6+

---

### 3. `mainpy.py` - System Orchestrator

**Purpose**: Main entry point that manages both IMU and Camera clients, providing a unified CLI interface for data collection control.

**Key Features**:
- Handles Windows COM initialization for Bleak/WinRT compatibility
- Asynchronous management of BLE communications
- Threaded camera capture running concurrently
- Interactive command-line interface for operation control
- Synchronized start/stop of both sensors

**Control Commands**:

| Command | Effect |
|---------|--------|
| `start` | Start data collection on both IMU and camera |
| `stop` | Pause data collection (video display continues) |
| `exit` | Stop all operations, close files, shutdown system |

**Execution Flow**:
1. Connects to BLE IMU device
2. Spawns camera capture thread
3. Waits for user commands
4. Synchronizes IMU and camera collection states
5. On exit: cleanly shuts down both clients and joins threads

**Platform Support**:
- Special handling for Windows (disables COM auto-initialization for Bleak)
- Should work on Linux/macOS with native Bleak support

---

## Integration Overview

```
┌─────────────────────────────────────────────────────────┐
│                    MainPy (Main Thread)                 │
│         - CLI interface (input/output blocking)         │
│         - Async event loop for BLE operations           │
│         - Command dispatching to clients                │
└─────────────────┬─────────────────┬─────────────────────┘
                  │                 │
        ┌─────────▼──┐       ┌──────▼────────┐
        │  IMUClient │       │ CameraClient  │
        │(Async BLE) │       │ (Thread-based)│
        └─────────┬──┘       └──────┬────────┘
                  │                 │
        ┌─────────▼──┐       ┌──────▼────────┐
        │imu_data.csv│       │camera_data.csv│
        └────────────┘       └───────────────┘
```

---

## Data Output Files

**imu_data.csv**
- IMU sensor readings (accelerometer + gyroscope)
- Sampling rate depends on Arduino firmware
- 8 columns: pc_time, arduino_time, ax, ay, az, gx, gy, gz

**camera_data.csv**
- Pose keypoint detection results
- Frame-based collection (depends on webcam framerate, typically 30 FPS)
- 14 columns: frame, timestamp, shoulder_*, elbow_*, wrist_* (x, y, z, visibility for each joint)

---

## Installation & Setup

### Requirements
- Python 3.10+
- Windows, macOS, or Linux

### Dependencies
```
bleak>=3.0.1          # BLE communication
mediapipe>=0.11.0     # Pose detection (NumPy 2.x compatible)
numpy>=2.2.6          # Numerical computation
opencv-python>=4.13.0 # Video capture
```

### Installation
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Running
```bash
# Using uv
uv run python mainpy.py

# Or directly
python mainpy.py
```

---

## Usage Example

```
$ python mainpy.py
Scanning for BLE devices...
Found: FB:5E:F5:A6:04:CB - Nano33BLE
Connecting to Nano33BLE (FB:5E:F5:A6:04:CB)...
IMU connected
READY (type start / stop / exit)

start
SYSTEM START
Camera START
[Video window opens, data collection begins]

stop
SYSTEM STOP
Camera STOP
[Video continues, but no data is logged]

exit
SYSTEM EXIT
Camera SHUTDOWN
[Video window closes, files are saved and closed]
```

---

## Troubleshooting

### BLE Connection Issues
- Ensure Arduino device is powered and advertising
- Check device name (`DEVICE_NAME`) and MAC address (`ADDRESS`) in `imuclient.py`
- On Windows, try pairing the device via Bluetooth settings first

### MediaPipe Error: "module 'mediapipe' has no attribute 'solutions'"
- Caused by incompatibility between MediaPipe 0.10.x and NumPy 2.x
- Solution: Upgrade to `mediapipe>=0.11.0`
```bash
pip install --upgrade "mediapipe>=0.11.0"
```

### Camera Not Detected
- Check that `cv2.VideoCapture(0)` is the correct device index
- Verify webcam is not in use by another application
- Try `cv2.VideoCapture(-1)` to auto-detect

### Poor Pose Detection
- Ensure adequate lighting
- Keep full right arm visible to camera
- Check MediaPipe visibility values in output CSV

---

## Notes

- Both sensors run asynchronously; their data is timestamped separately
- Data synchronization must be done in post-processing using the timestamp columns
- Camera display window can be closed with 'q' key to immediately shutdown
- For long recordings, monitor disk space as CSV files grow with collection time