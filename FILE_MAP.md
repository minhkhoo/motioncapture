# File map

## Recording (run on the laptop)

- `mainpy.py`: the program you run to record. It asks which devices to use, then waits for `start` and `stop`.
- `imuclient.py`: connects to the Arduino over Bluetooth, sends START and STOP, and saves the IMU samples to `imu_data.csv`.
- `cameraclient.py`: records the camera, then runs MediaPipe on the saved video to produce `camera_data.csv` and `annotated_video.mp4`. It can also reprocess an old session: `uv run python cameraclient.py --reprocess recording\<session>`.
- `shareddata.py`: a small object the IMU and camera threads share while recording.

## Looking at a session

- `dataprocessing.py`: the viewer and report generator. It shows the video with the skeleton views and a chart of the elbow angle and IMU acceleration, so you can read the ball-contact time.

## Processing

- `build_tracking.py`: creates and updates `data_tracking.xlsx` (one row per session, with automatic quality columns and manual columns), and writes `manual_contacts.csv`.
- `build_features_tracking.py`: reads `data_tracking.xlsx` and `manual_contacts.csv` and writes `features_tracking.csv`.
- `extraction.py`: the code that computes the features. It is used by `build_features_tracking.py` and can also be run on its own on a session folder.

## Arduino

- `firmware/arduino_device.ino`: the sketch that runs on the Nano 33 BLE.
- `firmware/README.md`: how to flash it and what it sends.

## Environment and documentation

- `pyproject.toml` and `uv.lock`: the Python version and the exact library versions. `uv sync` installs them.
- `docs/data_tracking_columns_EN.md`: what every column of `data_tracking.xlsx` means.
- `README.md`: the overview and the step-by-step instructions.

## Files created while you work (not in the repository)

- `recording/`: one folder per recorded session.
- `data_tracking.xlsx`, `manual_contacts.csv`, `features_tracking.csv`: the tracking sheet, the contact times, and the feature table.
