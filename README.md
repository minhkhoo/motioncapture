# Hybrid Motion Capture System: BLE IMU + Computer Vision

A high-performance, asynchronous dual-sensor motion capture system. This project combines **internal kinematics** (Accelerometer + Gyroscope via BLE IMU) with **external spatial tracking** (Smartphone Camera + MediaPipe) to accurately record and analyze human movement.

## Product Overview

Traditional motion capture systems are either expensive (Vicon) or suffer from occlusion (Vision-only) / drift (IMU-only). This system bridges the gap by offering a hybrid, low-cost solution:
1. **Vision-based Tracking:** Uses a smartphone camera (via Iriun Webcam) and Google's MediaPipe to capture 3D spatial coordinates (Pose Estimation) at 60 FPS.
2. **Inertial Tracking:** Uses an Arduino Nano 33 BLE to capture high-frequency 100Hz rotational and acceleration data to compensate for camera blind spots and occlusion.
3. **Data Fusion Pipeline:** Both data streams are timestamped precisely against the PC's internal clock for offline synchronization, interpolation, and visualization.

## Key Features

- **Asynchronous Architecture:** Camera processing and BLE communication run on completely separate threads, ensuring heavy computer vision tasks (MediaPipe) do not block the BLE event loop.
- **Data Batching Protocol:** Bypasses Windows' native 20Hz BLE limitation by batching 5 IMU readings per payload, achieving **100Hz** sampling rate.
- **Zero-Drop RAM Buffering:** IMU data is cached directly into RAM ($O(1)$ operations) during capture and flushed to disk only when the session stops, eliminating Disk I/O bottlenecks.
- **Automated Post-Processing:** Built-in tools to mathematically synchronize, interpolate, and plot the fused data streams for trajectory analysis.

---

## System Architecture & Code Structure

The project is modularized into dedicated background clients orchestrated by a main async loop:

### 1. `mainpy.py` (The Orchestrator)
- Acts as the central Command Line Interface (CLI) (`start`, `stop`, `exit`).
- Manages the `asyncio` event loop and coordinates the lifecycle of both the Camera and IMU clients.
- Dynamically creates timestamped session folders (`recording/YYYYMMDD_HHMMSS/`) for each capture cycle.

### 2. `imuclient.py` (BLE & Data Handling)
- **MTA Thread Isolation:** Initializes `CoInitializeEx(MTA)` to prevent Windows COM pollution from TensorFlow/MediaPipe, which commonly causes BLE crashes on Windows.
- **BLE Connection Management:** Handles scanning, connecting, and subscribing to the Arduino's notification characteristic.
- **RAM Buffer:** Efficiently unpacks incoming byte arrays (`struct.unpack`) and stores them in memory to prevent packet loss.

### 3. `cameraclient.py` (Vision Pipeline)
- Connects to the external video feed (e.g., Iriun Webcam via USB).
- Runs **MediaPipe Pose** to extract full-body landmarks (focusing on shoulder, elbow, wrist).
- Normalizes coordinates and renders the tracking skeleton overlay in real-time using OpenCV.

### 4. `shareddata.py`
- A lightweight, thread-safe module containing shared memory structures (`shared_data`).
- Allows cross-thread data access if real-time synchronization or live tracking is needed.

### 5. `dataprocessing.py` (Data Fusion & Visualization)
- **Time Synchronization:** Aligns the 60 FPS camera data with the 100Hz IMU data using the shared PC performance counter (`pc_time`).
- **Data Interpolation:** Uses mathematical interpolation to fill in temporal gaps, ensuring smooth and continuous data arrays.
- **Graphing & Plotting:** Automatically generates clear, comparative graphs for movement analysis, system debugging, and visualization of the raw vs. processed trajectory.

---

## Installation & Usage

### 1. Environment Setup
```bash
uv sync