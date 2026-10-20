# Hybrid Motion Capture for Volleyball Spikes

This project records a volleyball spike with two sensors at the same time and turns each recording into a small set of biomechanical features.

- A **wrist-worn IMU** (Arduino Nano 33 BLE) measures acceleration and rotation speed at about 100 samples per second and sends them to a laptop over Bluetooth.
- A **camera** (a phone used as a webcam) films the athlete. MediaPipe Pose then finds the shoulder, elbow and wrist in every frame.

The laptop software starts and stops both sensors together, saves everything into one folder per attempt, and later computes features such as the elbow angle at ball contact and how far the wrist travels during the swing.

This repository contains the recording software, the Arduino firmware, and the processing scripts. It does not contain any recorded data. You record your own sessions and the scripts work on those.

## What you need

Hardware: an Arduino Nano 33 BLE with its battery and enclosure (worn on the wrist of the hitting arm), a smartphone running the Iriun Webcam app on a tripod, and a Windows laptop with Bluetooth. The recording code only runs on Windows.

Software: Python 3.10 and the `uv` package manager.

## Setup

Install the dependencies:

    git clone https://github.com/minhkhoo/motioncapture
    cd motioncapture
    uv sync

Then flash the Arduino with the sketch in `firmware/`. The steps, the library you must install, and the Bluetooth packet format are described in `firmware/README.md`.

If your phone camera does not appear as camera number 1 on your laptop, change `CAMERA_INDEX` at the top of `cameraclient.py`. If the laptop cannot find the Arduino by its name (`Nano33BLE`), put its Bluetooth address in `ADDRESS` at the top of `imuclient.py`.

## Step 1: Record a session

Switch the Arduino on, start Iriun on the phone and the laptop, then run:

    uv run python mainpy.py

The program first asks which devices to use: camera only, IMU only, or both. Choose both for normal use. It connects to the Arduino and keeps the Bluetooth link open between recordings.

Then type `start`. The program asks what kind of session it is: professional, amateur, or test. For the first two it also asks for the athlete's name or ID. Recording begins right away. Perform the spike, then type `stop`. You can repeat start and stop as many times as you like, and `exit` closes the program.

Each recording creates a folder in `recording/`, named after the athlete and the start time, for example `recording/Alex_20260812_120719/`. Test sessions are named `TEST_` plus the time. The folder contains:

- `session_info.json`: whether the session was professional, amateur or test, and the athlete ID.
- `raw_video.mp4` and `frame_timestamps.csv`: the video as filmed and the exact laptop time of each frame.
- `imu_data.csv`: the IMU samples. Each row has the laptop time, the Arduino time, three acceleration values (in g) and three rotation values (in degrees per second).
- `camera_data.csv`: the shoulder, elbow and wrist positions found by MediaPipe for every frame. This file is produced right after you type `stop`, and it can take a while for long videos.
- `annotated_video.mp4`: the video with the skeleton drawn on it.
- `motion_report.html`: an interactive report, made automatically when camera data exists.

The camera is started before the IMU on purpose, because sending the start command over Bluetooth can take a moment. Both streams use the laptop clock, which is what allows the two to be lined up afterwards. If either sensor stops sending data for more than two seconds, the program prints a warning but keeps recording. If Bluetooth drops, the program tries to reconnect in the background without interrupting the camera.

## Step 2: Look at a session and mark the contact time

To open the viewer for one session:

    uv run python dataprocessing.py recording\Alex_20260812_120719

The viewer plays the annotated video together with three views of the skeleton (3D, front, and top-down) and one chart that shows the elbow angle and the IMU acceleration (in g) over time. It shows the time since the start of the video as `t = X.XXXs`. Step through the video, find the frame where the hand hits the ball, and write down that time. You will need it in the next step.

`dataprocessing.py` can also analyse an ordinary video file with no IMU: `uv run python dataprocessing.py spike.mp4`.

## Step 3: Build the tracking sheet

Run:

    uv run python build_tracking.py

This scans `recording/` and creates `data_tracking.xlsx`, with one row per session. It skips test sessions, sessions whose folder name contains `error`, and sessions without `session_info.json`.

Each row has two kinds of columns. The script fills the quality columns automatically every time it runs: how many samples the IMU delivered, how much data was lost, whether the sensor was moving at all, whether the signal hit the sensor's limit, and how many frames the camera delivered and in how many of them a person was found. The `imu_verdict` column summarises whether the IMU data is usable. Its value is `yes`, or `no` followed by the reason in brackets: missing file, a sampling rate below 30 Hz, too much packet loss, or a sensor that did not move (dead baseline).

You fill the manual columns yourself, and the script never overwrites them. The important one is `contact_ts`: type the contact time you read in the viewer. You can also use `good_recording`, `quality_star` and `notes`. If a recording is unusable, write `error` in `contact_ts` and it will be skipped later.

After you have entered contact times, run `build_tracking.py` again. It also writes `manual_contacts.csv`, the list of contact times that the feature extraction reads.

`docs/data_tracking_columns_EN.md` explains every column of the sheet.

## Step 4: Extract the features

Run:

    uv run python build_features_tracking.py

For every session that has a valid `contact_ts`, this runs `extraction.py` and writes one row to `features_tracking.csv`. A session with a blank or `error` contact time is skipped.

`extraction.py` works on the camera positions of the right shoulder, elbow and wrist. It uses the hand-marked contact time, then finds the moment of maximum elbow extension in the 1.5 seconds before contact. The time between those two moments is the acceleration phase of the swing. The eight features are:

- the elbow angle at contact
- the largest elbow angle before the acceleration phase (maximum cocking angle)
- the range of the elbow angle in the spike window
- the average elbow angular speed during the acceleration phase
- the distance the wrist travels during the acceleration phase
- the straight-line distance of the wrist from maximum extension to contact
- the peak wrist speed
- the duration of the acceleration phase

Wrist distances and speeds are divided by the athlete's upper-arm length (shoulder to elbow, taken from the same image coordinates), so they have no unit. This makes them less dependent on how far the athlete stands from the camera, but they are not metres, and the image coordinates themselves are not corrected for the image shape. `features_tracking.csv` also records the player, skill level, session date, the camera frame rate and the percentage of frames where a person was detected.

## Things to know before you trust the data

Check `imu_verdict` before using any IMU number. A recording can have no IMU file at all (the Bluetooth link never started), too many lost packets, or a sensor that stayed still. In our own recordings more than half of the sessions had unusable IMU data. The camera features do not depend on the IMU.

The IMU sampling is not perfectly regular. The effective rate and the amount of lost data are measured for every session, so look at those columns when you compare sessions.

The accelerometer was recorded at different sensitivity settings in different sessions (see `firmware/README.md`). Older recordings made at lower ranges clip on hard impacts.

## Repository contents

See `FILE_MAP.md` for a one-line description of every file.

## License

See the `LICENSE` file.
