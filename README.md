# Hybrid Motion Capture System — Volleyball Spike Analysis

A dual-sensor motion capture system combining an inertial measurement unit (IMU) worn on the arm with computer vision (camera + MediaPipe) to record, analyse, and visualise volleyball spike mechanics — and a downstream **feature-extraction and machine-learning pipeline** that turns recorded sessions into biomechanical features and a professional-vs-amateur classifier.

The project has **two halves**:

1. **Data collection & review** (`mainpy.py`, `cameraclient.py`, `imuclient.py`, `dataprocessing.py`) — records synchronised IMU + camera sessions and lets you review each one in a desktop viewer.
2. **Analysis & modelling** (`extraction.py`, `src/prepare_dataset.py`, `src/promote_session.py`, `src/train_model.py`) — extracts per-spike biomechanical features and trains/evaluates a classifier.

> ⚠️ **On the current results:** the dataset has only 4 players, which is far too few to draw any conclusion. The modelling pipeline is built to be **re-run when more data arrives**, not to publish accuracy. The evaluation reports its own limits bluntly (see `models/model_metrics.json` → `trustworthy: false`). A detailed file-by-file map of the analysis half lives in **`FILE_MAP.md`**.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Environment Setup](#3-environment-setup)
4. [Usage Guide](#4-usage-guide)
5. [Session Folder Structure](#5-session-folder-structure)
6. [Module Reference](#6-module-reference)
7. [Viewer & Analysis Charts](#7-viewer--analysis-charts)
8. [Offline Video Analysis](#8-offline-video-analysis)
9. [Analysis & Modelling Pipeline](#9-analysis--modelling-pipeline)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        mainpy.py                                │
│                    (Central Orchestrator)                       │
│         CLI: start / stop / exit    │    asyncio event loop     │
└────────────┬───────────────────────────────┬────────────────────┘
             │                               │
             ▼                               ▼
┌────────────────────┐           ┌────────────────────────┐
│   imuclient.py     │           │    cameraclient.py     │
│  Arduino Nano BLE  │           │  Iriun Webcam / Cam 0  │
│                    │           │                        │
│  Dedicated BLE     │           │  Dedicated camera      │
│  thread            │           │  thread                │
│  100 Hz (batch×5)  │           │  60 FPS recording      │
│  RAM buffer → CSV  │           │  raw_video.mp4         │
│  imu_data.csv      │           │  → MediaPipe offline   │
└────────────────────┘           │  → camera_data.csv     │
             │                   │  → annotated_video.mp4 │
             │                   └────────────┬───────────┘
             │                                │
             └────────────┬───────────────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │    dataprocessing.py   │
             │                        │
             │  EMA filter            │
             │  IMU fusion (interp)   │
             │  Elbow angle calc      │
             │  → motion_report.html  │
             │  → Open_Viewer.bat     │
             │  → PyQt6 viewer        │
             └────────────────────────┘

  ═══════════════ ANALYSIS & MODELLING HALF ═══════════════

  recording/<session>/          (a good session, reviewed by eye)
          │  src/promote_session.py   (copy + auto-update metadata)
          ▼
  ../Zipped data/{Amatures|Proffesionals}/<session>/   +   prepared_dataset/metadata.csv
          │  extraction.py --root "../Zipped data"
          ▼
  features.csv   (one row per detected spike, 54 columns)
          │  src/train_model.py  features.csv --metadata prepared_dataset/metadata.csv
          ▼
  results.csv · feature_importance.csv · models/  (joblib + feature_list + metrics)
```

**Data flow (collection half):**
- IMU (100 Hz) and Camera (60 FPS) run in parallel on independent threads
- Both use `time.perf_counter()` for PC-side timestamps → enables accurate post-hoc synchronisation
- After `stop`: IMU flushes RAM → `imu_data.csv`; camera runs MediaPipe offline → `camera_data.csv` + `annotated_video.mp4`
- `dataprocessing.py` reads both CSVs, fuses the data streams, and outputs an HTML report + viewer

**Data flow (analysis half):**
- `mainpy.py` writes a `session_info.json` (professional / amateur / test) into each session
- `promote_session.py` moves a good session into the research dataset (`Zipped data`), refusing test sessions, and updates `metadata.csv` (merge, never overwrite)
- `extraction.py` recursively scans `Zipped data`, detects the ball-contact frame per spike, segments the spike into 3 phases, and writes `features.csv`
- `train_model.py` joins features with `metadata.csv`, splits **by player** (LeaveOneGroupOut), and evaluates a Dummy / LogisticRegression / RandomForest ladder — plus confound checks

---

## 2. Hardware Requirements

| Device | Role | Notes |
|--------|------|-------|
| Arduino Nano 33 BLE | Captures IMU data (ax,ay,az,gx,gy,gz) @ 100 Hz | Worn on the forearm near the wrist of the spiking arm |
| Smartphone with Iriun Webcam | 60 FPS camera input | Connect via USB or WiFi; appears as camera Index 1 |
| Windows PC | Runs all processing | Python 3.10, sufficient RAM for IMU buffering |

**Camera fallback:** If Iriun is unavailable, the system automatically falls back to the default webcam (Index 0).

---

## 3. Environment Setup

```bash
# Clone the repository
git clone https://github.com/minhkhoo/motioncapture
cd motioncapture

# Install all dependencies
uv sync

# Install PyQt6 WebEngine (required for the viewer)
uv add pyqt6-webengine
```

**Key dependencies:**
- `bleak==0.22.3` — BLE communication with Arduino
- `mediapipe==0.10.9` — Pose estimation (skeleton tracking)
- `opencv-python` — Video read/write, frame processing
- `pandas`, `numpy`, `scipy` — Data processing and filtering
- `plotly` — Interactive 3D and 2D charts
- `PyQt6` + `PyQt6-WebEngine` — Desktop viewer

---

## 4. Usage Guide

### 4.1 Recording a new session

```bash
uv run python mainpy.py
```

The startup menu appears:
```
1. Camera only
2. IMU sensor only
3. Both devices simultaneously
```

After selecting a mode:
```
=== SYSTEM READY (type: start / exit) ===
> start
Session kind?  [1] professional   [2] amateur   [3] test
Enter your choice (1/2/3): 1
Enter athlete name or ID: Hoang
>>> RECORDING DATA (type 'stop' to finish)
> stop
```

The **session kind** question is important for the analysis half:
- **[1] professional / [2] amateur** → prompts for the athlete name; folder is `<name>_<timestamp>/`
- **[3] test** → no name; folder is `TEST_<timestamp>/`, and `promote_session.py` will refuse to add it to the research dataset

Either way, a `session_info.json` is written into the session folder (the source of truth used later by `promote_session.py` and `prepare_dataset.py`).

The system then automatically:
1. Stops IMU recording and flushes RAM → `imu_data.csv`
2. Stops camera recording and runs MediaPipe offline (may take ~30 seconds)
3. Exports `camera_data.csv` + `annotated_video.mp4`
4. Exports `motion_report.html` + `Open_Viewer.bat`

### 4.2 Reviewing a recorded session

```bash
# Open the viewer for a specific session
uv run python dataprocessing.py recording/Hoang_20260612_110759

# Or double-click the shortcut inside the session folder
Open_Viewer.bat
```

### 4.3 Analysing an existing video (no Arduino required)

```bash
# Analyse any video file and create a new session
uv run python dataprocessing.py spike_clip.mp4

# Analyse the left arm instead
uv run python dataprocessing.py spike_clip.mp4 --side LEFT

# Use the most accurate model (slower)
uv run python dataprocessing.py spike_clip.mp4 --complexity 2

# Write output into an existing session folder
uv run python dataprocessing.py recording/Hoang_20260612_110759/raw_video.mp4 --out recording/Hoang_20260612_110759
```

---

## 5. Session Folder Structure

```
recording/
└── Hoang_20260612_110759/
    ├── raw_video.mp4          ← Raw camera footage (XVID codec)
    ├── annotated_video.mp4    ← Video with skeleton overlay (mp4v codec)
    ├── camera_data.csv        ← Per-frame 3D coordinates: shoulder/elbow/wrist
    ├── imu_data.csv           ← ax,ay,az,gx,gy,gz @ 100 Hz from Arduino
    ├── session_info.json      ← {session_kind, player_id, recorded_at} — source of truth for the analysis half
    ├── motion_report.html     ← Combined report (opened automatically by viewer)
    └── Open_Viewer.bat        ← Double-click shortcut to open viewer
```

**Files that can be deleted:**
- `temp_timestamps.csv` — temporary file, deleted automatically after post-processing
- `raw_video_h264.mp4` — created by an older code version, safe to remove
- `Xem_Do_Thi_3D.bat` — created by an older code version, replaced by `Open_Viewer.bat`

**`camera_data.csv` format:**
```
frame, timestamp, shoulder_x, shoulder_y, shoulder_z, shoulder_visibility,
       elbow_x, elbow_y, elbow_z, elbow_visibility,
       wrist_x, wrist_y, wrist_z, wrist_visibility
```
- Coordinates are normalised to [0.0 – 1.0] relative to frame width/height
- `timestamp` = PC `time.perf_counter()` value at the moment the frame was captured

**`imu_data.csv` format:**
```
pc_time, arduino_time, ax, ay, az, gx, gy, gz
```
- `ax,ay,az` = linear acceleration (G-force)
- `gx,gy,gz` = angular velocity (°/s or rad/s)
- `pc_time` = PC timestamp (synchronised with camera)
- `arduino_time` = Arduino `millis()` value (reference only)

---

## 6. Module Reference

### `mainpy.py` — Central Orchestrator
- Runs an `asyncio` event loop on the main thread listening for CLI commands
- Initialises the camera thread and BLE thread independently
- Manages the full recording lifecycle: `open_session → start → stop → close_session`
- After stop: automatically calls `dataprocessing.run()` to export the report

### `cameraclient.py` — Camera Pipeline
- Connects to the camera via OpenCV (Index 1 = Iriun, fallback to Index 0)
- A `run()` thread continuously captures frames; only writes to disk when `collect_event` is set
- Writes in parallel: frames → `raw_video.mp4`, timestamps → `temp_timestamps.csv`
- After stop, `_run_offline_post_processing()` runs MediaPipe on each frame:
  - Detects body skeleton and draws overlay (full body in grey, spiking arm highlighted in cyan)
  - Renders elbow angle as text on each frame
  - Exports `camera_data.csv` + `annotated_video.mp4`

### `imuclient.py` — BLE IMU Client
- Runs a BLE event loop on a dedicated thread (required on Windows due to COM threading constraints)
- Uses **data batching**: Arduino sends 5 readings per BLE packet (140 bytes), bypassing the Windows BLE stack's ~20 notification/second limit → achieves **100 Hz** effective sample rate
- Data is buffered in RAM (O(1) per reading) and flushed to disk in a single write at the end of the session — eliminates I/O bottleneck during recording
- Sends `START`/`STOP` commands via BLE Write characteristic

### `shareddata.py` — Shared Memory
- `camera_buffer`, `imu_buffer`: deques (maxlen=20000) for live cross-thread data sharing
- `session_dir`: path of the active session folder
- `graph_running`, `program_running`: coordination flags between threads

### `dataprocessing.py` — Processing & Visualisation

**Step A — EMA Filter (coordinate smoothing):**
```
x_smooth = alpha × x_smooth_prev + (1 − alpha) × x_raw
```
- Shoulder: alpha=0.85 (heavy smoothing — stable reference origin)
- Elbow: alpha=0.50
- Wrist: alpha=0.30 (faster response to preserve swing detail)

**Step B — IMU Data Fusion:**
- Computes `accel_magnitude = sqrt(ax² + ay² + az²)` and removes baseline offset
- Uses `numpy.interp` to align 100 Hz IMU timestamps with 60 FPS camera timestamps

**Step C — Elbow Angle Calculation:**
```
angle = 180° − arccos( (v1 · v2) / (|v1| × |v2|) )
```
Where v1 = elbow − shoulder, v2 = wrist − elbow

**Step D — HTML Report + PyQt6 Viewer:**
- HTML contains all session data as inline JSON (no server required)
- PyQt6 viewer: `QMediaPlayer` (video) + `QWebEngineView` (graphs) inside a `QSplitter`
- `graphs_only.html`: a stripped version of the HTML with the video panel removed, served to Chromium for graph-only rendering

### `extraction.py` — Feature Extraction *(analysis half, at repo root)*
- Reads `camera_data.csv` + `imu_data.csv` per session and writes one row per detected spike to `features.csv`
- **Contact detection (IPF)**: `IPF = local_dev(accel) × local_dev(gyro)` peaks at the ball-contact moment
- **Phase segmentation**: splits each spike into cocking / acceleration / follow-through
- Two IMU time axes with separate roles: `arduino_time` (monotonic, for all internal timing) vs a PC-mapped axis (only for comparing to the camera). The PC mapping uses an NTP-style lower-envelope fit.
- Flags data-quality problems per spike: `clip_fraction`, `peak_is_clipped` (the LSM9DS1 accel hard-clips at ±4g), `contact_near_gap`, `sync_quality_ms`
- Run: `python extraction.py --root "../Zipped data" --output features.csv --quiet`

### `src/prepare_dataset.py` — Metadata Builder *(analysis half)*
- Recursively scans `Zipped data`, copies each session's files into `prepared_dataset/` with an `attempt_id` prefix, and **merges** into `metadata.csv` (keeps hand-edited `good_recording` / `notes`, never overwrites; backs up `.bak`)
- Skips test sessions; lists any session whose `skill_level` could not be inferred

### `src/promote_session.py` — Session Promotion *(analysis half)*
- One command to move a recorded session from `recording/` into `Zipped data/{Proffesionals|Amatures}/` and update `metadata.csv`
- **Refuses test sessions** unless `--force`; supports `--dry-run`
- Run: `python src/promote_session.py recording/<session>`

### `src/train_model.py` — Training & Evaluation *(analysis half)*
- Joins `features.csv` with `metadata.csv` (the **single source of truth** — only listed sessions are used)
- Splits **by player** (LeaveOneGroupOut) so no player appears in both train and test
- Model ladder: `DummyClassifier` → `LogisticRegression` → `RandomForest`; reports **balanced accuracy** and macro-F1, pooled across folds
- **Confound checks**: re-predicts `session_type` with the same features, and correlates each feature with recording-quality signals — to detect whether the model is learning recording conditions instead of skill
- Outputs: `results.csv`, `feature_importance.csv`, `models/spike_skill_model.joblib`, `models/feature_list.json`, `models/model_metrics.json`

---

## 7. Viewer & Analysis Charts

Opened via `Open_Viewer.bat` or `uv run python dataprocessing.py <session_folder>`:

```
┌──────────────────┬──────────────────────────────────────────────┐
│   📹  VIDEO      │  ◈ 3D View  │ ◈ Front (Z–Y) │ ◈ Bird (X–Z) │
│                  │───────────────────────────────────────────── │
│  QMediaPlayer    │       📈 Elbow Angle & IMU Acceleration      │
│  (native codec)  │                                              │
│                  │  ● Green line  = elbow angle (°)             │
│  Frame 127       │  ● Orange line = IMU acceleration (G)        │
│  152.0°  9.929s  │  ● Cyan dashed = current frame cursor        │
├──────────────────┤                                              │
│ ▶ Play  ⏮ ‹ › ⏭ │                                             │
│ 📹 Annotated     │                                             │
│ 1.00×            │                                              │
└──────────────────┴──────────────────────────────────────────────┘
```

**Controls:**
- `Space` — Play / Pause
- `←` `→` — Step ±1 frame
- `Shift+←` `Shift+→` — Jump ±10 frames
- **📹 Annotated** button — Toggle between raw video and skeleton overlay video
- Speed selector — 0.25×, 0.5×, 1×, 2×

**Three coordinate views (all relative to the shoulder as origin):**
- **3D View** — Free-angle interactive view; rotate with mouse while paused
- **Front View (Z–Y)** — Camera-facing plane (depth vs height); matches what the camera sees
- **Birdeye View (X–Z)** — Top-down plane (lateral vs depth)

Faded trail = full session trajectory | Bright trail = most recent 45 frames

---

## 8. Offline Video Analysis

No Arduino required — works with any video file:

```bash
uv run python dataprocessing.py <input> [--side RIGHT|LEFT] [--complexity 0|1|2] [--out <folder>] [--no-viewer]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--side` | RIGHT | Which arm to analyse |
| `--complexity` | 1 | MediaPipe model: 0=fast, 1=balanced, 2=most accurate |
| `--out` | auto | Custom output folder |
| `--no-viewer` | False | Export files only, do not open viewer |

If detection rate is low (<50%), try `--complexity 2` or check lighting and camera angle.

---

## 9. Analysis & Modelling Pipeline

Run every command from the `motioncapture/` folder. See **`FILE_MAP.md`** for a file-by-file map, column dictionaries, and current known issues.

**Full workflow after recording a good session:**

```bash
# 1. Promote a reviewed session into the research dataset (one command)
python src/promote_session.py recording/<session>
#    → copies into ../Zipped data/, refuses test sessions, updates metadata.csv (merge)
#    Then open prepared_dataset/metadata.csv and, by eye: set good_recording=no for bad
#    sessions, fill session_type (primary/secondary) after reviewing the video, add notes.

# 2. Extract features from the whole dataset (~2s for ~24 sessions)
python extraction.py --root "../Zipped data" --output features.csv --quiet

# 3. Train + evaluate (5–15s)
python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv

# 4. Trace confounds across feature sets (optional, 20–40s)
python src/train_model.py features.csv --metadata prepared_dataset/metadata.csv --compare-feature-sets
```

**Key rules:**
- `metadata.csv` is the **single source of truth** — a session in `features.csv` that is not listed there is ignored. This is deliberate: `recording/` contains many throwaway test sessions.
- Always use `extraction.py --root` on `Zipped data`, **not** `--all` on `recording/` (that would miss sessions that only exist in `Zipped data`).
- `train_model.py` never mixes a player across train/test; with only 4 players it prints a **loud warning** and treats results as not reliable.

**Automation status:** the mechanical steps (copy, rename, metadata update) are automated by `promote_session.py`; the **judgment** steps (reviewing video, deciding `good_recording`, rating `session_type`, writing `notes`) are intentionally left manual.

---

## 10. Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| `raw_video.mp4` is 0–1 KB | Camera disconnected mid-recording (Iriun dropped due to device overheating or unstable WiFi/USB connection) | Use USB cable instead of WiFi; ensure device is cool before recording |
| `ModuleNotFoundError: PyQt6.QtWebEngineWidgets` | Missing package | Run `uv add pyqt6-webengine` |
| Video spins indefinitely in browser | CORS blocks `file:///` local resource loading | Use `Open_Viewer.bat` instead of opening the HTML directly in a browser |
| Charts are empty, `seek is not defined` error | JS crashed before Plotly initialised (null DOM element) | Fixed in current version via null-checks on all DOM elements |
| BLE cannot find Arduino | Wrong device name or not paired | Verify `DEVICE_NAME = "Nano33BLE"` in `imuclient.py` |
| Low pose detection rate | Poor lighting, occluded body, or low model complexity | Add `--complexity 2`; ensure the full upper body is visible in frame |
| Video and graphs play out of sync | FPS mismatch between video file and CSV-derived value | Fixed in current version: FPS is now read directly from the video file via `cv2.CAP_PROP_FPS` |