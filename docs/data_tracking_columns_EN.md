# `data_tracking.xlsx` — Column Reference (English)

## What this project is

We record volleyball players performing a **spike** (the overhead attacking hit) using
**two sensors at the same time**:

1. **A camera** — video is run through *MediaPipe*, a Google library that finds the
   human skeleton in each video frame. It gives us the position of the shoulder,
   elbow, and wrist in every frame.
2. **An IMU** (Inertial Measurement Unit) — a small Arduino sensor worn on the wrist.
   It measures **acceleration** (in G, where 1 G = the force of gravity) and **rotation
   speed** (in dps = degrees per second). It sends data to the PC over Bluetooth (BLE).

One **session** = one recorded attempt (roughly one spike). Each session is one folder
containing `camera_data.csv`, `imu_data.csv` and the video.

The end goal is a model that tells a **professional** player's technique apart from an
**amateur**'s. `data_tracking.xlsx` is the master table: one row per session, holding
both what we measured automatically and what the researcher marked by hand.

## How to read this table

Columns fall into three groups:

| Group | Who fills it | Can I edit it? |
|---|---|---|
| **Identity** | script, automatically | No — regenerated each run |
| **Manual** | the researcher, by hand | **Yes — this is your input** |
| **Quality** | script, automatically | No — regenerated each run |

Regenerate with:

```bash
uv run --with openpyxl python build_tracking.py
```

The script never overwrites the Manual columns.

---

## A. Identity columns (automatic)

### `session_name`
The session folder name, e.g. `Hop_20260812_105835`. Format:
`{player}_{YYYYMMDD}_{HHMMSS}`. This is the unique key that links this table to the
raw data folders and to every other file in the pipeline.

### `source`
Where the session folder lives:
- `zipped` — inside `Zipped data/`, the curated research dataset already used for modelling.
- `recording` — inside `recording/`, the raw capture folder (newer sessions).

### `player_id`
The athlete's name, taken from the front of `session_name`. Used to **group** the data
when training: the model is validated by holding out one whole player at a time, so it
is never tested on someone it has already seen.

### `skill_level`
The label the model is trying to predict — `professional` or `amateur`. It comes from
`session_info.json` (the operator chose it when recording), or from the old folder
layout for the `zipped` sessions.

### `session_type`
Which camera setup was used:
- `primary` / `secondary` — two different camera angles/positions used in the older sessions.
- `new` — a session recorded after that distinction stopped being recorded; not yet classified.

This matters because the camera angle changes the measured values, so it is also used
as a *confound check*: if a model can guess the camera angle as easily as it guesses
skill, then it is learning the recording setup, not the technique.

---

## B. Manual columns (you fill these in)

### `contact_ts` ⭐ most important
**The moment the hand strikes the ball**, in seconds counted from the start of that
session's video. This is exactly the `t = …s` figure shown in the viewer produced by
`dataprocessing.py`.

Everything the model measures is anchored to this instant, so it is the single most
valuable piece of hand-marked information. Accepted formats:

| You type | Meaning |
|---|---|
| `4.56` | contact at 4.56 s |
| `~4.24` | approximately 4.24 s (uncertain) |
| `5.644-5.675` | somewhere in this range → the midpoint is used, flagged as approximate |
| `error` | this session cannot be marked → it is **excluded** from training |

### `landed_ts`
The moment the player's **feet touch the ground** again after the jump, same time base.
Reserved for a future "follow-through" measurement; **not used by the model yet**.

### `good_recording`
`yes` / `no` — whether the session should be used for research. `train_model.py` keeps
only rows marked `yes`.

### `quality_star`
A free-form personal marker (e.g. `STAR`) for sessions worth highlighting. Not used by
any script.

### `notes`
Free text about anything unusual — e.g. *"first attempt failed, second attempt starts
at 11.301"*, *"capture wrong hand until 3.298"*, *"video ends early"*. These matter:
the model looks at a **1.5-second window before contact**, so if something went wrong
inside that window the numbers for that session are unreliable.

---

## C. Quality columns (automatic)

### C1 — IMU verdict

### `imu_verdict`
The overall judgement on whether this session's IMU data is usable:
- `yes` — usable.
- `no (missing file)` — no `imu_data.csv`; the sensor never connected.
- `no (rate NN Hz<30)` — the effective sampling rate is below 30 Hz.
- `no (dead baseline)` — the file exists and the sensor kept transmitting, but it
  **stopped responding to movement**: the acceleration peak is below 2 G or the rotation
  peak is below 100 dps (a sensor lying still reads about 1 G and near 0 dps). That is what a sensor lying still reads, so the data is
  worthless even though it looks complete.
- `no (loss NN%>50)` — more than half the data packets were lost over Bluetooth, so
  any timing-based measurement is too broken to trust.

The checks are applied in this order: missing file, rate, loss, dead baseline. Treat every `no (...)` as **"this session has no IMU"**.

### C2 — IMU transmission health

### `imu_freq_hz`
Actual sampling rate received, in samples per second. The sensor is designed for
**100 Hz**; anything lower means packets were lost in transmission. Above ~85 Hz is good.

### `imu_stalls`
How many times the data stream **paused for more than 0.1 s**. Ideally 0.

### `imu_max_stall_ms`
The longest single pause, in milliseconds. A 1500 ms gap means one and a half seconds
of the spike is simply missing.

### `imu_loss_pct`
Percentage of expected samples that never arrived. Under 10 % is good; over 50 % makes
the session unusable (see `imu_verdict`).

### `lost_stall` / `lost_small`
Splits the lost samples by cause: `lost_stall` = lost inside the long pauses above,
`lost_small` = lost in many tiny dropouts. Useful for diagnosing *why* the link was bad
(a few long freezes vs. constant weak signal).

### C3 — IMU measurement range

### `accel_range_G`
The accelerometer's configured maximum, `4`, `8` or `24` G. Different firmware versions
were used over time. This matters because any real force **above** this ceiling is
clipped — recorded as the ceiling value instead of the true value.

### `accel_clip_pct`
Percentage of samples that hit that ceiling. When this is high, hard spikes and
very hard spikes both record the same maximum number, which **erases** exactly the
difference the model is trying to find.

### `accel_peak_G`
The strongest acceleration measured in the session. A real spike typically shows tens
of G. A peak near 1 G means the sensor was not moving — see `no (dead baseline)` above.

### `gyro_clip_pct` / `gyro_peak_dps`
Same idea for rotation. The gyroscope's ceiling is 2000 dps. Spikes often reach
1500–2500 dps, so some clipping here is normal and cannot be avoided with this hardware.

### C4 — Camera

### `cam_fps`
Frames actually captured per second. The target is 60.

### `cam_median_gap_ms`
The typical time between two frames. At 60 fps this should be about 16 ms.

### `cam_max_gap_ms`
The longest gap between two consecutive frames — a spike in this value means the camera
briefly froze.

### `video_frames`
Number of frames in `raw_video.mp4`. A value near 0 means the video file is empty and
the session cannot be reviewed or re-analysed.

### `detect_pct` and `detect_frames`
The percentage (and the raw count, `found/total`) of frames in which MediaPipe
successfully located the skeleton.

> **Important:** a low value here is usually **not** a fault. It normally means the
> athlete simply walked out of the camera frame between attempts. As long as the
> skeleton is correctly attached to the right person *during the spike itself*, a low
> percentage is harmless. Other causes — poor lighting, the skeleton latching onto the
> wrong person — would be a real problem.

### `pass1`, `pass2`, `pass0`
How each frame was resolved by the two-stage detector:
- `pass1` — found on the fast first pass (**highest quality**).
- `pass2` — found only on the slower recovery pass, which retries harder frames with a
  heavier model and brightness enhancement (**lower quality, treat with care**).
- `pass0` — not found at all; that row exists in the CSV but its coordinates are blank.

### C5 — Camera ↔ IMU synchronisation

### `sync_cam_first`
`yes` if the camera started recording **before** the IMU. This is the desired
order: the camera is started first on purpose, so that waiting for the Bluetooth link
can never cause the beginning of the movement to be missed.

### `sync_delay_s`
The gap in seconds between the two start times — in effect, how slow Bluetooth was that
time.

### `overlap_pct`
The percentage of the camera recording during which the IMU was *also* recording. Only
this overlapping period can be analysed with both sensors together. Above 90 % is good.

### C6 — Summary

### `actual_issues`
(real problems) A short summary of everything genuinely wrong with the session —
missing/dead IMU, high packet loss, long stalls, clipping, poor overlap, empty video.
`OK` means nothing was found.

**Deliberately excluded from this column:** a low `detect_pct`, for the reason given
above — the athlete stepping out of frame is normal behaviour, not a recording fault.

---

## Quick guide: is this session usable?

| Question | Look at |
|---|---|
| Can it be used for training at all? | `contact_ts` is filled in (not `error`) |
| Can the IMU features be trusted? | `imu_verdict` = `yes` |
| Can the camera features be trusted? | `notes` has no warning about the spike window |
| Is the joint IMU+camera analysis valid? | `overlap_pct` ≥ 90 and `sync_cam_first` = `yes` |
| Is `accel_peak_G` a true value? | `accel_clip_pct` is low |
