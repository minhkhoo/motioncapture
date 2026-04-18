# Motion Capture System

Dual-sensor motion capture: BLE IMU (accelerometer + gyroscope) + webcam pose estimation (MediaPipe).

## Setup

```bash
uv sync
uv run python mainpy.py
```

Requires Python 3.10, Arduino Nano 33 BLE, and a webcam.

## Usage

```
READY (type start / stop / exit)

start    → creates recording/YYYYMMDD_HHMMSS/ and begins capture
stop     → stops capture, closes CSVs
exit     → disconnects BLE, releases camera, quits
```

Multiple start/stop cycles create separate session folders. Press `q` in the video window to quit.

## Output

Each session folder contains:

- **imu_data.csv** — `pc_time, arduino_time, ax, ay, az, gx, gy, gz`
- **camera_data.csv** — `frame, timestamp, shoulder_xyz, elbow_xyz, wrist_xyz` (right arm, normalized 0–1)

## Architecture

- `mainpy.py` — CLI orchestrator, async event loop
- `imuclient.py` — BLE on a dedicated MTA thread (Windows COM isolation from MediaPipe)
- `cameraclient.py` — OpenCV + MediaPipe on a separate thread

## Troubleshooting

- **BLE not found** — ensure Arduino is powered and not connected elsewhere
- **Windows BLE issues** — handled automatically (MTA thread, `use_cached_services=False`)
- **Camera not detected** — check webcam index in `CameraClient.__init__`