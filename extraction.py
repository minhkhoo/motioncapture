"""
feature_extraction.py — Biomechanical Feature Extraction Pipeline
==================================================================
Step 1: Locate the ball-contact frame. Default = IPF auto-detection (legacy). Migration
        in progress toward hand-marked contacts read from contacts.csv (Coleman et al.
        1993); once that lands, IPF moves behind --use-ipf.
Step 2: Find the two events that bound the ACCELERATION phase — max elbow extension
        (within [contact − 1.5s, contact]) and contact. (Simplified from the old
        4-boundary / 3-phase version, whose cocking-start and follow-through-end
        boundaries were arbitrary thresholds.)
Step 3: Extract the curated features — 13 model-inputs + QC/identifier columns.

Input:  One or more session folders containing camera_data.csv + imu_data.csv
Output: features.csv — one row per contact, columns = features

Usage:
  # One session:
  python extraction.py recording/Hoang_20260612_110759

  # A whole recording folder (all sessions):
  python extraction.py recording/ --all

  # Specify the output:
  python extraction.py recording/ --all --output results/features.csv
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

# Camera coordinate smoothing: Savitzky-Golay filter (~0.1s window, polyorder 2).
# Replaced EMA (which had initialization bias from the first frame, phase lag
# proportional to α, and fps-dependent behavior) — see _compute_kinematics.
CAM_SAVGOL_WINDOW_S = 0.10   # seconds — same as the IMU Savgol window

# IPF detection (legacy contact detector — kept for --use-ipf comparison only)
IPF_WINDOW     = 5    # samples per side for the local deviation (~50ms at 100Hz)
MIN_PEAK_DIST  = 50   # min distance between 2 peaks (500ms) — avoid detecting one spike twice

# Analysis windows around the contact frame, in SECONDS. PRE also bounds the camera
# search for the max-extension event; both bound the IMU feature window around contact.
# (Converted to sample/frame counts per session — the IMU BLE link drops packets so the
# rate varies 22-68Hz instead of a fixed 100Hz.)
PRE_CONTACT_WINDOW_S  = 1.5  # seconds before contact
POST_CONTACT_WINDOW_S = 1.0  # seconds after contact (IMU peak/impulse window only)

# ── Sensor limits (LSM9DS1 via the Arduino_LSM9DS1 library) ─────────
# Legacy firmware: CTRL_REG6_XL = 0x70 → ±4g per axis, saturates at 4.0 (magnitude
# ceiling sqrt(3)*4 = 6.93G). New firmware reconfigures to ±8g → saturates at 8.0.
# The right clip threshold per session is chosen from accel_range (see load_session).
ACC_CLIP_THRESHOLD_G_4G  = 3.99   # ±4g firmware (legacy sessions)
ACC_CLIP_THRESHOLD_G_8G  = 7.99   # ±8g firmware (newer sessions)
ACC_CLIP_THRESHOLD_G_24G = 23.99  # ±24g firmware (newest sessions, setAccelFS(1))
ACC_CLIP_THRESHOLD_G     = ACC_CLIP_THRESHOLD_G_4G   # back-compat default (±4g)

# Gyro never has its range changed by firmware (Part C1 audit: no .ino ever calls
# setGyroFS) -- always the LSM9DS1 library default +-2000 dps. Per-axis, matching
# the accel clip check below (checking the combined magnitude >=2000 overcounts:
# three axes at ~1200 dps each already give a magnitude over 2000 with no single
# axis anywhere near its real ceiling).
GYRO_CLIP_THRESHOLD_DPS = 1999.0

# Gap detection: arduino_time spacing above this threshold = a dropped BLE packet
IMU_GAP_THRESHOLD_S    = 0.050   # gap > 50ms
CONTACT_GAP_RADIUS_S   = 0.100   # contact within 100ms of a gap → flag it

# micros() on the Arduino is uint32 → wraps after 2^32 µs ≈ 71.6 minutes
UINT32_MICROS_PERIOD_S = 2**32 / 1e6

# Detect folders that "look like a session": name contains a {yyyymmdd}_{hhmmss} stamp
TIMESTAMP_IN_NAME_RE = re.compile(r"_\d{8}_\d{6}")


def _unwrap_micros(t_sec: np.ndarray) -> np.ndarray:
    """Undo uint32 wraparound of micros(): add a correction at each sudden backward jump.

    A wrap makes the diff ≈ -4294.97s. The -1000s threshold is wide enough not to
    confuse it with normal jitter (which is only milliseconds).
    """
    d = np.diff(t_sec)
    wraps = np.where(d < -1000.0)[0]
    if len(wraps) == 0:
        return t_sec
    out = t_sec.copy()
    for w in wraps:
        out[w + 1:] += UINT32_MICROS_PERIOD_S
    return out


def _map_to_pc_clock(t_ard: np.ndarray, pc: np.ndarray, n_bins: int = 25,
                     low_frac: float = 0.05) -> tuple[np.ndarray, float, float]:
    """Map arduino_time → the PC time scale using a LOWER-ENVELOPE fit.

    delta = pc − arduino consists of the true offset PLUS the BLE latency. The
    latency is always POSITIVE and highly variable (delayed packets, batching), so a
    plain least-squares fit is dragged by the latency tail. The sample with the
    SMALLEST delta is the least-delayed one → closest to the true offset. Take the
    lower envelope per bin then fit a line (same lower-envelope principle as NTP).

    Fit affine (a*t + b) rather than a constant, to also compensate for the clock
    rate mismatch between the Arduino and the PC.

    Returns (pc_time_scale, sync_quality_ms, sync_lowlat_fraction).
    """
    delta = pc - t_ard
    edges = np.linspace(t_ard[0], t_ard[-1], n_bins + 1)

    pts_t, pts_d = [], []
    for i in range(n_bins):
        m = (t_ard >= edges[i]) & (t_ard <= edges[i + 1] if i == n_bins - 1
                                   else t_ard < edges[i + 1])
        if not np.any(m):
            continue
        dd, tt = delta[m], t_ard[m]
        k = max(1, int(np.ceil(low_frac * len(dd))))
        lo = np.argsort(dd)[:k]          # the k lowest-latency samples in the bin
        pts_t.append(float(np.mean(tt[lo])))
        pts_d.append(float(np.mean(dd[lo])))

    if len(pts_t) >= 2:
        a, b = np.polyfit(np.array(pts_t), np.array(pts_d), 1)
        # sync_quality: how well the line fits the lower-envelope points themselves
        resid_pts = np.array(pts_d) - (a * np.array(pts_t) + b)
        sync_quality_ms = float(np.std(resid_pts) * 1000.0)
    else:
        a, b = 0.0, float(np.median(delta))
        sync_quality_ms = float("nan")

    offset = a * t_ard + b
    imu_ts_pc = t_ard + offset

    # Residual over ALL samples = the latency remaining after removing the offset (≥0)
    resid_all = delta - offset
    sync_lowlat_fraction = float(np.mean(resid_all <= 0.010))

    return imu_ts_pc, sync_quality_ms, sync_lowlat_fraction


# np.trapz is deprecated as of numpy 2.0 → use trapezoid when available
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _jerk_stats(sig: np.ndarray, t: np.ndarray, prefix: str) -> dict:
    """Jerk statistics (time derivative of acceleration) for one signal.

    sig  : the FILTERED source signal (savgol) — required, because differentiation
           amplifies noise heavily; differentiating the raw signal yields almost pure noise.
    t    : the corresponding REAL time axis (seconds, must be strictly increasing).
    prefix: "imu_gyro" or "imu_acc".

    dimensionless_jerk = (T³ / A²) × ∫ jerk² dt
      T = window duration, A = source-signal amplitude (max − min).
    This normalization removes the effect of duration and amplitude: comparing |jerk|
    directly would always give a hard hitter a high jerk, which just re-measures SPEED
    rather than SMOOTHNESS. Larger = jerkier, smaller = smoother.

    Window < 3 samples or a non-strictly-increasing t → return NaN (no crash).
    """
    keys = [f"{prefix}_jerk_mean_abs", f"{prefix}_jerk_max_abs",
            f"{prefix}_jerk_rms", f"{prefix}_jerk_dimensionless"]
    out = {k: np.nan for k in keys}

    if sig is None or t is None or len(sig) < 3 or len(t) != len(sig):
        return out
    if not np.all(np.diff(t) > 0):      # gradient divides by dt → dt=0 gives inf
        return out

    jerk = np.gradient(sig, t)
    aj = np.abs(jerk)
    out[f"{prefix}_jerk_mean_abs"] = float(np.mean(aj))
    out[f"{prefix}_jerk_max_abs"]  = float(np.max(aj))
    out[f"{prefix}_jerk_rms"]      = float(np.sqrt(np.mean(jerk ** 2)))

    T = float(t[-1] - t[0])
    A = float(np.max(sig) - np.min(sig))
    if T > 0 and A > 0:
        out[f"{prefix}_jerk_dimensionless"] = float(
            (T ** 3 / A ** 2) * _trapz(jerk ** 2, t))

    return out


# ──────────────────────────────────────────────────────────────────────
# STEP A: READ AND PREPARE THE DATA
# ──────────────────────────────────────────────────────────────────────

def _probe_video_fps(session_folder: Path) -> float | None:
    """Declared FPS of the session's video, for the timestamp→video-frame estimate.

    Only used for the human-readable contact report (NOT for feature values).
    Returns None if there is no video or it can't be read.
    """
    import cv2  # local import — feature extraction itself doesn't need OpenCV
    for name in ("annotated_video.mp4", "raw_video.mp4"):
        vp = session_folder / name
        if vp.exists():
            cap = cv2.VideoCapture(str(vp))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and 0 < fps <= 240:
                return float(fps)
    return None


def _detect_accel_range(session_folder: Path, axes: np.ndarray) -> int:
    """Accelerometer full-scale range in ±g (4, 8 or 24) for a session.

    Priority:
      1. 'accel_range' in session_info.json, if present (4, 8 or 24) — declare it there
         to be certain.
      2. Auto-detect from the largest observed sample: >8.05g proves it is not the ±8g
         build (which saturates at 8.0) → ±24g; >4.05g proves it is not the ±4g build
         (saturates at 4.0) → ±8g; otherwise default to 4 (legacy). A higher-range
         session that never exceeded a lower ceiling is reported as the lower range,
         which is harmless — with no sample near either ceiling, no clip is flagged.
    """
    info_path = session_folder / "session_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            r = info.get("accel_range")
            if r in (4, 8, 24, "4", "8", "24"):
                return int(r)
        except Exception:
            pass
    mx = float(np.nanmax(np.abs(axes)))
    return 24 if mx > 8.05 else (8 if mx > 4.05 else 4)


def load_session(session_folder: Path, force_camera_only: bool = False) -> dict | None:
    """
    Read camera_data.csv and imu_data.csv, align them on a common timestamp.
    Return a dict with all processed data, or None if a file is missing.

    force_camera_only=True: ignore imu_data.csv even if it exists (used when
    camera_data.csv was regenerated independently so its timestamps no longer share
    a clock with imu_data.csv — mixing the two sources would place the contact wrong).
    """
    cam_path = session_folder / "camera_data.csv"
    imu_path = session_folder / "imu_data.csv"

    if not cam_path.exists():
        print(f"  [SKIP] No camera_data.csv: {session_folder.name}")
        return None

    has_imu = imu_path.exists() and not force_camera_only
    if force_camera_only and imu_path.exists():
        print(f"  [INFO] --no-imu: ignoring imu_data.csv, using camera features only")
    if not has_imu:
        print(f"  [WARN] No imu_data.csv — using camera features only")

    # --- Camera data ---
    try:
        df_cam = pd.read_csv(cam_path)
    except pd.errors.EmptyDataError:
        print(f"  [SKIP] Empty camera_data.csv: {session_folder.name}")
        return None

    # Drop NaN rows (undetected poses) before computing — they are kept in the CSV file
    # but cannot be used for kinematics. Savgol filter requires contiguous valid data.
    n_before = len(df_cam)
    df_cam = df_cam.dropna(subset=["timestamp", "shoulder_x"]).reset_index(drop=True)
    n_dropped = n_before - len(df_cam)
    if n_dropped > 0:
        print(f"  [INFO] Dropped {n_dropped}/{n_before} NaN rows (pose not detected / missing timestamp) "
              f"for the calculation, kept in the file: {session_folder.name}")

    if len(df_cam) < 30:
        print(f"  [SKIP] Camera data too short ({len(df_cam)} valid rows): {session_folder.name}")
        return None

    cam_ts = df_cam["timestamp"].values
    cam_fps = 1.0 / np.median(np.diff(cam_ts)) if len(cam_ts) > 1 else 60.0
    cam_fps = float(np.clip(cam_fps, 20, 120))

    # Savgol filter on the camera coordinates (symmetric, no phase lag)
    elbow_angles, wrist_positions, arm_length = _compute_kinematics(df_cam, cam_fps)

    # MediaPipe visibility (0–1) — a DATA QUALITY indicator, not a technique feature
    # (see the warning in the extract_features docstring). Falls back to NaN if an
    # older camera_data.csv lacks these columns.
    def _vis_col(name: str) -> np.ndarray:
        if name in df_cam.columns:
            return df_cam[name].values.astype(float)
        return np.full(len(df_cam), np.nan)

    cam_vis_shoulder = _vis_col("shoulder_visibility")
    cam_vis_elbow    = _vis_col("elbow_visibility")
    cam_vis_wrist    = _vis_col("wrist_visibility")

    # --- IMU data ---
    imu_ts = imu_acc = imu_gyro = None
    if has_imu:
        df_imu = pd.read_csv(imu_path)
        if len(df_imu) < 10:
            has_imu = False
        else:
            # ── TWO TIME AXES, WITH CLEARLY SEPARATED ROLES ──────────────
            #  • imu_t  = arduino_time (µs→s, unwrapped): ABSOLUTELY MONOTONIC,
            #    written by micros() AT THE MOMENT the sensor is read → used for ALL
            #    internal duration measurements (spacing, sample rate, impulse, time-to-peak, windows).
            #  • imu_ts = imu_t mapped onto the PC scale → ONLY used when comparing to cam_ts.
            #
            # Do NOT use pc_time as the internal axis: imuclient.py back-dates the whole
            # batch of 5 samples from the moment the BLE packet arrives (pc_arrival − (4−i)*10ms),
            # assuming a uniform 10ms spacing. BLE jitter makes a later packet overlap an
            # earlier one → pc_time runs BACKWARDS (measured on 20/20 sessions, worst −38.6ms).
            # np.interp and every time subtraction then silently return garbage on a non-monotonic axis.
            has_ard = "arduino_time" in df_imu.columns
            has_pc  = "pc_time" in df_imu.columns

            sync_quality_ms = float("nan")
            sync_lowlat_fraction = float("nan")

            if has_ard:
                imu_timebase = "arduino_time"
                imu_t = _unwrap_micros(df_imu["arduino_time"].values.astype(float) / 1e6)
                if has_pc:
                    imu_ts, sync_quality_ms, sync_lowlat_fraction = _map_to_pc_clock(
                        imu_t, df_imu["pc_time"].values.astype(float))
                    if sync_lowlat_fraction < 0.10:
                        print(f"  [WARN] Camera–IMU sync unreliable: only "
                              f"{sync_lowlat_fraction*100:.1f}% of samples have latency ≤10ms "
                              f"({session_folder.name})")
                else:
                    imu_ts = imu_t   # no PC reference → cannot compare to the camera
            else:
                # Fallback: old file without arduino_time
                imu_timebase = "pc_time" if has_pc else df_imu.columns[0]
                imu_ts = df_imu[imu_timebase].values.astype(float)
                imu_t  = imu_ts
                print(f"  [WARN] No arduino_time column — falling back to '{imu_timebase}' "
                      f"(axis may be non-monotonic): {session_folder.name}")

            imu_acc   = np.sqrt(df_imu["ax"]**2 + df_imu["ay"]**2 + df_imu["az"]**2).values
            imu_gyro  = np.sqrt(df_imu["gx"]**2 + df_imu["gy"]**2 + df_imu["gz"]**2).values

            # ── ACCEL RANGE + CLIPPING ───────────────────────────────────
            # The LSM9DS1 accel hard-clips PER AXIS (check each axis, not the magnitude:
            # the magnitude can be < ceiling while one axis is already clipped). Firmware
            # range = ±4g (clip 3.99), ±8g (clip 7.99) or ±24g (clip 23.9). accel_range
            # decides which threshold to use AND tells train_model.py whose imu_peak_acc_g
            # is trustworthy (a lower-range session compresses hard spikes to the ceiling).
            axes = df_imu[["ax", "ay", "az"]].values
            accel_range = _detect_accel_range(session_folder, axes)
            if accel_range == 24:
                clip_thresh = ACC_CLIP_THRESHOLD_G_24G
            elif accel_range == 8:
                clip_thresh = ACC_CLIP_THRESHOLD_G_8G
            else:
                clip_thresh = ACC_CLIP_THRESHOLD_G_4G
            imu_clip_mask = (np.abs(axes) >= clip_thresh).any(axis=1)
            clip_fraction = float(imu_clip_mask.mean())

            # ── GYRO CLIPPING (per-axis, Part C2) ─────────────────────────
            # Checking the combined magnitude >=2000 overcounts: three axes at
            # ~1200 dps each already produce a magnitude over 2000 with no
            # single axis anywhere near its real ±2000 dps ceiling. Per-axis
            # avoids that false positive (same reasoning as the accel check).
            gyro_axes = df_imu[["gx", "gy", "gz"]].values
            gyro_clip_mask = (np.abs(gyro_axes) >= GYRO_CLIP_THRESHOLD_DPS).any(axis=1)
            gyro_clip_fraction = float(gyro_clip_mask.mean())

            # Average sample rate, computed on the monotonic axis (arduino_time)
            imu_duration = imu_t[-1] - imu_t[0]
            imu_sample_rate = len(imu_t) / imu_duration if imu_duration > 0 else 100.0
            if imu_sample_rate < 40:
                print(f"  [WARN] Low IMU sample rate ({imu_sample_rate:.1f} Hz) "
                      f"— poor data quality due to packet loss: {session_folder.name}")

            # Baseline subtraction — remove the gravity offset from the accel
            baseline_n = min(15, len(imu_acc) // 4)
            imu_acc_clean = np.abs(imu_acc - np.mean(imu_acc[:baseline_n]))

            # Savitzky-Golay smoothing — reduces noise while keeping peaks sharper than EMA.
            # Window scales with the actual sample rate (~0.1s), min 5 (polyorder+2), always odd.
            window_length = max(int(round(imu_sample_rate * 0.1)), 5)
            if window_length % 2 == 0:
                window_length += 1

            if len(imu_acc_clean) > window_length:
                imu_acc_smooth  = savgol_filter(imu_acc_clean, window_length=window_length, polyorder=3)
                imu_gyro_smooth = savgol_filter(imu_gyro,      window_length=window_length, polyorder=3)
            else:
                imu_acc_smooth  = imu_acc_clean
                imu_gyro_smooth = imu_gyro

    return {
        "session_name": session_folder.name,
        "cam_ts":       cam_ts,
        "cam_fps":      cam_fps,
        "video_fps":    _probe_video_fps(session_folder),
        "elbow_angles": elbow_angles,
        "wrist_positions": wrist_positions,
        "arm_length":   arm_length,
        "cam_vis_shoulder": cam_vis_shoulder,
        "cam_vis_elbow":    cam_vis_elbow,
        "cam_vis_wrist":    cam_vis_wrist,
        "has_imu":      has_imu,
        "imu_ts":       imu_ts,                      # PC scale — for comparing to cam_ts
        "imu_t":        imu_t if has_imu else None,  # arduino scale — internal timing
        "imu_timebase":         imu_timebase         if has_imu else None,
        "sync_quality_ms":      sync_quality_ms      if has_imu else None,
        "sync_lowlat_fraction": sync_lowlat_fraction if has_imu else None,
        "imu_clip_mask":  imu_clip_mask if has_imu else None,
        "clip_fraction":  clip_fraction if has_imu else None,
        "gyro_clip_mask":     gyro_clip_mask     if has_imu else None,
        "gyro_clip_fraction": gyro_clip_fraction if has_imu else None,
        "accel_range":    accel_range   if has_imu else np.nan,
        "n_contacts_rejected": 0,
        "imu_sample_rate": imu_sample_rate if has_imu else None,
        "imu_acc":      imu_acc_smooth  if has_imu else None,
        "imu_gyro":     imu_gyro_smooth if has_imu else None,
        "imu_acc_raw":  imu_acc_clean   if has_imu else None,
        "imu_gyro_raw": imu_gyro        if has_imu else None,
    }


def _compute_kinematics(df_cam: pd.DataFrame, cam_fps: float = 60.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Savgol-smoothed elbow-angle and wrist-position computation.

    Replaced EMA (which seeded from the first frame and accumulated state):
    - No initialization bias: Savgol is a symmetric, non-causal FIR filter.
    - No phase lag: the smoothed value is centered, not delayed.
    - FPS-independent: window set in seconds, converted to frames internally.

    Returns 3 things:
    - angles:          (N,)   elbow angle per frame
    - wrist_positions: (N,3)  wrist coordinates relative to the shoulder
    - arm_length:      scalar median shoulder→elbow distance (robust normalizer)
    """
    sh = np.column_stack([ df_cam["shoulder_x"].values,
                          -df_cam["shoulder_y"].values,
                          -df_cam["shoulder_z"].values])
    el = np.column_stack([ df_cam["elbow_x"].values,
                          -df_cam["elbow_y"].values,
                          -df_cam["elbow_z"].values])
    wr = np.column_stack([ df_cam["wrist_x"].values,
                          -df_cam["wrist_y"].values,
                          -df_cam["wrist_z"].values])

    n = len(sh)

    # Savgol window: ~0.1s, min 5, always odd, capped at array length
    win = max(int(round(cam_fps * CAM_SAVGOL_WINDOW_S)), 5)
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else max(n - 1, 5)

    if n >= win >= 5:
        for coords in (sh, el, wr):
            for j in range(3):
                coords[:, j] = savgol_filter(coords[:, j], window_length=win, polyorder=2)

    # Relative vectors
    el_rel = el - sh          # shoulder → elbow
    wr_rel = wr - sh          # shoulder → wrist

    # Elbow angle: 180 − arccos(v1·v2 / |v1||v2|)
    v1 = el_rel               # upper arm
    v2 = wr_rel - el_rel      # forearm
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos_a = dot / (n1 * n2 + 1e-6)
    angles = 180.0 - np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    arm_length = float(np.median(n1)) if n > 0 else np.nan
    return angles, wr_rel, arm_length


# ──────────────────────────────────────────────────────────────────────
# STEP B: DETECT THE CONTACT FRAME (IPF)
# ──────────────────────────────────────────────────────────────────────

def detect_contact_frames(data: dict) -> list[dict]:
    """
    Find all contact moments in the session via IPF.

    IPF (IMU Peak Function) = local_deviation(accel) × local_deviation(gyro)
    Both signals peak at the ball-contact moment:
    - accel: the impact with the ball produces an acceleration spike
    - gyro:  the wrist snap produces a peak angular velocity

    Returns a list of contact events, each with its position and a confidence value.
    """
    if not data["has_imu"]:
        # Fallback: use the camera — find frames where the elbow angle drops fastest
        return _detect_contacts_camera_only(data)

    acc  = data["imu_acc"]
    gyro = data["imu_gyro"]
    n    = len(acc)

    # Compute the local deviation for each sample
    def local_dev(signal, w):
        dev = np.zeros(len(signal))
        for i in range(w, len(signal) - w):
            local_mean = np.mean(signal[i-w:i+w+1])
            dev[i] = signal[i] - local_mean
        return dev

    acc_dev  = local_dev(acc,  IPF_WINDOW)
    gyro_dev = local_dev(gyro, IPF_WINDOW)

    # IPF = product of the two local deviations
    # Keep only positive values (when both are above the local mean)
    ipf = np.maximum(acc_dev, 0) * np.maximum(gyro_dev, 0)

    if np.max(ipf) == 0:
        print(f"  [WARN] IPF = 0 for the whole session — no contact detected")
        return []

    # Normalize for easier thresholding
    ipf_norm = ipf / np.max(ipf)

    # Find peaks: must exceed 30% of the global max, at least MIN_PEAK_DIST samples apart
    threshold = 0.3
    peaks, props = find_peaks(
        ipf_norm,
        height=threshold,
        distance=MIN_PEAK_DIST,
        prominence=0.1,
    )

    if len(peaks) == 0:
        # Retry with a lower threshold
        peaks, props = find_peaks(ipf_norm, height=0.15, distance=MIN_PEAK_DIST)

    if len(peaks) == 0:
        print(f"  [WARN] Could not find a contact frame for {data['session_name']}")
        return []

    # Convert IMU sample index → timestamp → camera frame index
    imu_ts  = data["imu_ts"]
    cam_ts  = data["cam_ts"]
    cam_fps = data["cam_fps"]
    contacts = []
    n_rejected = 0

    # Tolerance: the contact must fall within 1 camera frame. Farther away means the
    # camera was NOT recording at that moment (the two sources only partly overlap —
    # measured median overlap is only ~66%, lowest 11%).
    max_dt = 1.0 / cam_fps if cam_fps > 0 else 0.05

    for peak_idx in peaks:
        contact_ts = imu_ts[peak_idx]

        # Find the camera frame closest to the contact timestamp
        cam_frame_idx = int(np.argmin(np.abs(cam_ts - contact_ts)))

        # argmin ALWAYS returns an index even when the contact lies entirely outside
        # the camera window — it then clamps to the first/last frame and is silently
        # wrong. Check the real distance before accepting.
        dt = abs(float(cam_ts[cam_frame_idx]) - float(contact_ts))
        if dt > max_dt:
            n_rejected += 1
            print(f"  [SKIP] Contact at ts={float(contact_ts):.3f} rejected: nearest camera "
                  f"frame is {dt*1000:.0f}ms away (camera and IMU don't overlap here)")
            continue

        contacts.append({
            "imu_idx":       int(peak_idx),
            "cam_frame_idx": cam_frame_idx,
            "contact_ts":    float(contact_ts),
            "ipf_value":     float(ipf_norm[peak_idx]),
            "acc_at_contact": float(acc[peak_idx]),
            "gyro_at_contact": float(gyro[peak_idx]),
        })

    data["n_contacts_rejected"] = n_rejected
    return contacts


def _detect_contacts_camera_only(data: dict) -> list[dict]:
    """Fallback when there is no IMU: find contact from the elbow-angle drop rate."""
    angles = data["elbow_angles"]
    # Largest negative derivative = fastest elbow flexion = near contact
    d_angle = -np.gradient(angles)
    if np.max(d_angle) == 0:
        return []

    peaks, _ = find_peaks(d_angle / np.max(d_angle), height=0.3, distance=30)
    contacts = []
    for p in peaks:
        contacts.append({
            "imu_idx":       None,
            "cam_frame_idx": int(p),
            "contact_ts":    float(data["cam_ts"][p]),
            "ipf_value":     float(d_angle[p] / np.max(d_angle)),
            "acc_at_contact": None,
            "gyro_at_contact": None,
        })
    return contacts


# ──────────────────────────────────────────────────────────────────────
# STEP C: PHASE SEGMENTATION
# ──────────────────────────────────────────────────────────────────────

def segment_phases(data: dict, contact: dict) -> dict | None:
    """
    Locate the TWO events that bound the ACCELERATION phase of the spike.

    Simplified from the old 4-boundary version, which relied on two arbitrary
    thresholds now removed: cocking-start (gyro > mean + 1.5·SD) and
    follow-through-end (a fixed post-contact window). Only signal-derived events remain:

      • max_extension : elbow angle is largest (arm cocked furthest back) within
                        [contact − PRE_CONTACT_WINDOW_S, contact]  ← the one correct
                        boundary kept from the old version.
      • contact       : the detected / hand-marked ball-contact frame.

    ACCELERATION phase = [max_extension → contact]. Returns their camera-frame indices,
    timestamps, and the phase duration; None if there aren't enough pre-contact frames.
    """
    cam_ts      = data["cam_ts"]
    angles      = data["elbow_angles"]
    cam_fps     = data["cam_fps"]
    contact_cam = contact["cam_frame_idx"]

    # Pre-contact search window in camera frames (session's own cam_fps)
    pre_window_cam = int(PRE_CONTACT_WINDOW_S * cam_fps)
    window_start   = max(0, contact_cam - pre_window_cam)
    if window_start >= contact_cam:
        return None

    pre_angles = angles[window_start:contact_cam]
    if len(pre_angles) == 0:
        return None

    # Event 1: max elbow extension (arm cocked furthest back) in the pre-contact window
    max_extension_cam = window_start + int(np.argmax(pre_angles))

    return {
        "max_extension_cam": int(max_extension_cam),
        "contact_cam":       int(contact_cam),
        "max_extension_ts":  float(cam_ts[max_extension_cam]),
        "contact_ts":        float(contact["contact_ts"]),
        "acceleration_duration_s": float(cam_ts[contact_cam] - cam_ts[max_extension_cam]),
    }


# ──────────────────────────────────────────────────────────────────────
# STEP D: FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def extract_features(data: dict, contact: dict, phases: dict) -> dict:
    """
    Extract the curated biomechanical features from one spike.

    Output = 13 MODEL-INPUT features + QC/identifier columns (see the combine block at
    the end for the exact split). Everything not in the 13 is a QC/identifier column:
    keep it for filtering and cross-referencing, but NEVER feed it to the classifier.

    ⚠️ RELIABILITY WHEN THE ACCEL IS CLIPPED
    =========================================
    The LSM9DS1 accelerometer hard-clips PER AXIS: ±4g firmware saturates at 4.0
    (magnitude ceiling sqrt(3)*4 = 6.93G), ±8g firmware at 8.0. On the legacy ±4g
    dataset a median 13% of samples were clipped and 95% of sessions had the peak
    sample clipped. Use the accel_range column (4 or 8) to tell them apart.

    imu_peak_acc_g is NOT trustworthy when clip_fraction is high / peak_is_clipped = 1
    on a ±4g session (hard and very-hard spikes both read ~6.93G, compressing away the
    skill difference). ipf_confidence is likewise affected (IPF = local_dev(acc) ×
    local_dev(gyro)) — but ipf_confidence is QC-only, not a model input.

    CLIP-SAFE — gyro does not hit its ±2000dps ceiling (measured max 0.23% of samples):
        imu_peak_gyro_dps, imu_gyro_at_contact, imu_gyro_impulse_duration_s,
        imu_gyro_jerk_dimensionless, all cam_*, phase_acceleration_duration_s.

    Companion flag columns for filtering: clip_fraction, accel_range, peak_is_clipped,
    contact_near_gap, sync_quality_ms, sync_lowlat_fraction, n_contacts_rejected.

    📐 JERK — MOTION SMOOTHNESS (gyro only)
    ==================================
    imu_gyro_jerk_dimensionless = (T³/A²)·∫jerk²dt over the acceleration phase —
    normalized to remove duration and amplitude so it compares across spikes of
    different strength (see _jerk_stats); comparing raw |jerk| would just re-measure
    speed, not smoothness. Gyro-based (trustworthy); the accel-based jerk was dropped
    because a clipped, flat-topped accel segment produces a large FAKE derivative.

    ⚠️ cam_vis_* — DATA QUALITY INDICATORS, NOT FEATURES
    =============================================================
    The cam_vis_* features are DATA QUALITY INDICATORS, NOT technique features.
    NEVER use them as training features for the skill classifier.

    Reason: visibility reflects recording conditions (lighting, camera angle,
    occlusion), not player skill. In the current dataset the detection rate varies
    53–100% and is systematically skewed between the two groups — professional
    sessions have worse detection quality. Fed in as features, the model would learn
    'recording conditions' instead of 'spike technique', yielding a falsely high score.

    Correct use: as a session-filtering criterion, or as a control variable in analysis.

    The cam_vis_* prefix is uniform so they can be dropped in bulk when selecting features:
        feature_cols = [c for c in df.columns if not c.startswith("cam_vis_")]
    """
    angles  = data["elbow_angles"]
    cam_ts  = data["cam_ts"]
    has_imu = data["has_imu"]

    # Phase indices (2-event model): max extension → contact
    mx_e  = phases["max_extension_cam"]
    acc_e = phases["contact_cam"]          # = contact frame

    # ── ELBOW ANGLE FEATURES ──────────────────────────────────────────
    # At the contact moment
    angle_at_contact = float(angles[acc_e]) if acc_e < len(angles) else np.nan

    # Max extension angle = elbow angle at the max_extension event (which IS the max
    # over the [contact − 1.5s, contact] search window). Column name kept for compatibility.
    angle_max_cocking = float(angles[mx_e]) if mx_e < len(angles) else np.nan

    # ROM within the acceleration phase [max_extension → contact] ONLY.
    # BUG FIX: this used to be np.min/np.max over the WHOLE session, so every contact in
    # a session reported an identical ROM (e.g. all three Vang contacts = 94.901).
    seg_ang = angles[mx_e:acc_e + 1] if acc_e >= mx_e else angles[acc_e:mx_e + 1]
    angle_rom = float(np.max(seg_ang) - np.min(seg_ang)) if len(seg_ang) > 0 else np.nan

    # Angular closing rate over the acceleration phase (deg/s; negative = snapping down)
    def angular_rate(idx_start, idx_end):
        if idx_end <= idx_start or idx_end >= len(angles) or idx_start >= len(angles):
            return np.nan
        dt = cam_ts[idx_end] - cam_ts[idx_start]
        if dt <= 0:
            return np.nan
        return float((angles[idx_end] - angles[idx_start]) / dt)

    angle_rate_acceleration = angular_rate(mx_e, acc_e)

    # ── WRIST PATH FEATURES (normalized by arm_length) ─────────────────
    # Divide by arm_length to remove the effect of player-camera distance:
    # the unit becomes "number of arm lengths" instead of raw pixels.
    wrist_pos  = data.get("wrist_positions")
    arm_length = data.get("arm_length", np.nan)

    cam_wrist_path_length_norm  = np.nan
    cam_wrist_displacement_norm = np.nan
    cam_wrist_peak_speed_ratio_per_sec = np.nan

    # Only compute when arm_length is valid (>0 and not NaN) — avoid div-by-0 / bad sessions
    if wrist_pos is not None and np.isfinite(arm_length) and arm_length > 0:
        lo = max(0, mx_e)
        hi = min(len(wrist_pos) - 1, acc_e)
        if hi > lo:
            seg     = wrist_pos[lo:hi+1]                  # (M,3) wrist positions during the spike
            seg_ts  = cam_ts[lo:hi+1]
            steps   = np.diff(seg, axis=0)                # displacement vectors between frames
            step_d  = np.linalg.norm(steps, axis=1)       # length of each step

            # Path length: total distance the wrist travels over the acceleration phase
            cam_wrist_path_length_norm = float(np.sum(step_d) / arm_length)

            # Displacement: straight-line distance from max extension → contact
            contact_local = min(max(acc_e - lo, 0), len(seg) - 1)
            disp = np.linalg.norm(seg[contact_local] - seg[0])
            cam_wrist_displacement_norm = float(disp / arm_length)

            # Peak speed: max(|Δpos| / Δt) — the fastest wrist speed
            dt = np.diff(seg_ts)
            valid = dt > 0
            if np.any(valid):
                speed = step_d[valid] / dt[valid]         # pixels/second
                cam_wrist_peak_speed_ratio_per_sec = float(np.max(speed) / arm_length)

    # ── IMU FEATURES ─────────────────────────────────────────────────
    imu_feats = {}
    if has_imu:
        acc   = data["imu_acc"]
        gyro  = data["imu_gyro"]
        # (don't take imu_ts here: all internal timing uses imu_t — the arduino axis)

        imu_sample_rate = data["imu_sample_rate"]
        imu_t = data["imu_t"]          # arduino axis, monotonic — all internal timing
        contact_imu = contact["imu_idx"]
        t_contact = imu_t[contact_imu]

        # Cut the window by REAL TIME on the arduino axis (see the note in
        # segment_phases): searchsorted instead of seconds × average_sample_rate.
        pre_imu  = int(np.searchsorted(imu_t, t_contact - PRE_CONTACT_WINDOW_S, side="left"))
        post_imu = int(np.searchsorted(imu_t, t_contact + POST_CONTACT_WINDOW_S, side="right")) - 1
        pre_imu  = max(0, min(pre_imu, contact_imu))
        post_imu = min(len(acc) - 1, max(post_imu, contact_imu))

        acc_window  = acc[pre_imu:post_imu+1]
        gyro_window = gyro[pre_imu:post_imu+1]
        t_window    = imu_t[pre_imu:post_imu+1]

        # Peak values (window around contact)
        imu_feats["imu_peak_acc_g"]     = float(np.max(acc_window))
        imu_feats["imu_peak_gyro_dps"]  = float(np.max(gyro_window))

        # Gyro at contact
        imu_feats["imu_gyro_at_contact"] = float(gyro[contact_imu])

        # Impulse duration (time gyro > 50% of peak, proxy for "snap" duration).
        # Sum the REAL duration each sample represents (np.gradient on the arduino
        # axis) instead of counting samples / average_sample_rate — the average sample
        # rate is skewed by gaps, so counting samples would give the wrong duration.
        gyro_thresh = 0.5 * np.max(gyro_window)
        above = gyro_window > gyro_thresh
        if len(t_window) > 1:
            dt_w = np.gradient(t_window)
            imu_feats["imu_gyro_impulse_duration_s"] = float(np.sum(dt_w[above]))
        else:
            imu_feats["imu_gyro_impulse_duration_s"] = 0.0

        # IPF value at contact — QC/diagnostic ONLY (kept until the contacts.csv path
        # replaces IPF as the default contact source; NOT a model-input feature).
        imu_feats["ipf_confidence"] = float(contact["ipf_value"])

        # ── DATA QUALITY FLAGS ───────────────────────────────────────
        # peak_is_clipped: whether the sample AT contact has any axis hitting the
        # accel_range ceiling (±4g or ±8g). If =1, imu_peak_acc_g for this spike is a
        # CHOPPED value, not the true value.
        clip_mask = data["imu_clip_mask"]
        imu_feats["peak_is_clipped"] = int(bool(clip_mask[contact_imu]))

        # contact_near_gap: the contact sits next to a data gap (dropped BLE packet) →
        # the surrounding window is missing samples, so every IMU feature of this spike is less reliable.
        near_gap = 0
        if len(imu_t) > 1:
            d = np.diff(imu_t)
            gap_i = np.where(d > IMU_GAP_THRESHOLD_S)[0]
            for gi in gap_i:
                if (abs(t_contact - imu_t[gi])     <= CONTACT_GAP_RADIUS_S or
                    abs(t_contact - imu_t[gi + 1]) <= CONTACT_GAP_RADIUS_S):
                    near_gap = 1
                    break
        imu_feats["contact_near_gap"] = near_gap

    # ── JERK / SMOOTHNESS (gyro only — accel clips, gyro doesn't) ─────
    # Dimensionless gyro jerk over the ACCELERATION phase [max_extension → contact].
    # Camera indices → IMU indices via imu_ts (PC scale, same frame as cam_ts), but
    # DIFFERENTIATE on imu_t (arduino scale, monotonic, correct physical dt).
    jerk_feats = {"imu_gyro_jerk_dimensionless": np.nan}
    if has_imu:
        imu_ts_pc = data["imu_ts"]
        imu_t_ard = data["imu_t"]
        gyro_s    = data["imu_gyro"]

        t0, t1 = float(cam_ts[mx_e]), float(cam_ts[acc_e])
        if t1 < t0:
            t0, t1 = t1, t0
        i0 = int(np.searchsorted(imu_ts_pc, t0, side="left"))
        i1 = int(np.searchsorted(imu_ts_pc, t1, side="right"))
        sl = slice(max(0, i0), min(len(imu_t_ard), i1))
        js = _jerk_stats(gyro_s[sl], imu_t_ard[sl], "imu_gyro")
        jerk_feats["imu_gyro_jerk_dimensionless"] = js["imu_gyro_jerk_dimensionless"]

    # ── MEDIAPIPE VISIBILITY (QUALITY INDICATOR, NOT A FEATURE) ──
    vis_sh = data.get("cam_vis_shoulder")
    vis_el = data.get("cam_vis_elbow")
    vis_wr = data.get("cam_vis_wrist")

    def _nanmean(a):
        return float(np.nanmean(a)) if a is not None and len(a) and not np.all(np.isnan(a)) else np.nan

    def _pct_below(a, thr=0.5):
        if a is None or len(a) == 0:
            return np.nan
        m = ~np.isnan(a)
        return float(np.mean(a[m] < thr) * 100.0) if m.any() else np.nan

    w0, w1 = min(mx_e, acc_e), max(mx_e, acc_e) + 1     # spike window
    vis_wr_win = vis_wr[w0:w1] if vis_wr is not None else None

    means = [_nanmean(vis_sh), _nanmean(vis_el), _nanmean(vis_wr)]
    vis_feats = {
        "cam_vis_shoulder_mean":             means[0],
        "cam_vis_elbow_mean":                means[1],
        "cam_vis_wrist_mean":                means[2],
        "cam_vis_wrist_pct_below_05":        _pct_below(vis_wr),
        "cam_vis_min_of_three":              float(np.nanmin(means)) if not np.all(np.isnan(means)) else np.nan,
        # Within the spike window: detection quality AT the moment of the spike matters
        # more than the whole-session average (can be good while still, lost during the swing).
        "cam_vis_wrist_mean_window":         _nanmean(vis_wr_win),
        "cam_vis_wrist_pct_below_05_window": _pct_below(vis_wr_win),
    }

    # ── PHASE TIMING FEATURE ─────────────────────────────────────────
    phase_feats = {
        "phase_acceleration_duration_s": phases["acceleration_duration_s"],
    }

    # ── COMBINE — curated: 13 MODEL-INPUT features + QC/identifier columns ──
    # Anything not in the 13 model-inputs is a QC/identifier column (keep for filtering
    # and cross-referencing, never feed to the skill classifier). See docstring.
    features = {
        # ── Identifiers / meta ──
        "session_name":         data["session_name"],
        "has_imu":              int(has_imu),
        "contact_frame":        int(acc_e),
        "contact_ts":           float(contact["contact_ts"]),
        "contact_ts_offset_s":  float(cam_ts[acc_e] - cam_ts[0]),

        # ── Data / sync quality (QC — for filtering, NOT model input) ──
        "imu_sample_rate":      float(data["imu_sample_rate"]) if data["imu_sample_rate"] is not None else np.nan,
        "imu_timebase":         data["imu_timebase"] if has_imu else "",
        "sync_quality_ms":      float(data["sync_quality_ms"])      if has_imu else np.nan,
        "sync_lowlat_fraction": float(data["sync_lowlat_fraction"]) if has_imu else np.nan,
        "clip_fraction":        float(data["clip_fraction"])        if has_imu else np.nan,
        "gyro_clip_fraction":   float(data["gyro_clip_fraction"])   if has_imu else np.nan,
        "accel_range":          data.get("accel_range", np.nan),   # 4, 8 or 24 (±g) — which imu_peak_acc_g is trustworthy
        "n_contacts_rejected":  int(data.get("n_contacts_rejected", 0)),

        # ── MODEL-INPUT FEATURES (13) ──
        # Elbow angle (camera) — 4
        "cam_angle_at_contact_deg":        angle_at_contact,
        "cam_angle_max_cocking_deg":       angle_max_cocking,
        "cam_angle_rom_deg":               angle_rom,
        "cam_angle_rate_acceleration_dps": angle_rate_acceleration,
        # Wrist path, normalized by arm length — 3
        "cam_wrist_path_length_norm":   cam_wrist_path_length_norm,
        "cam_wrist_displacement_norm":  cam_wrist_displacement_norm,
        "cam_wrist_peak_speed_ratio_per_sec": cam_wrist_peak_speed_ratio_per_sec,
        # Phase timing — 1  (phase_acceleration_duration_s)
        **phase_feats,
        # IMU — 4 model-inputs (imu_peak_acc_g, imu_peak_gyro_dps, imu_gyro_at_contact,
        # imu_gyro_impulse_duration_s) PLUS QC flags (ipf_confidence, peak_is_clipped,
        # contact_near_gap) carried in the same dict.
        **imu_feats,
        # Jerk / smoothness — 1  (imu_gyro_jerk_dimensionless)
        **jerk_feats,
        # ── QC: MediaPipe visibility indicators (NOT model input) ──
        **vis_feats,
    }

    return features


# ──────────────────────────────────────────────────────────────────────
# STEP E: VISUALIZE (debug — print to terminal)
# ──────────────────────────────────────────────────────────────────────

def print_session_summary(data: dict, contacts: list, all_phases: list, all_features: list):
    """Print a summary of the analysis results to the terminal."""
    print(f"\n{'='*60}")
    print(f"  SESSION: {data['session_name']}")
    print(f"{'='*60}")
    print(f"  Camera: {len(data['elbow_angles'])} frames @ {data['cam_fps']:.1f} fps")
    if data["has_imu"]:
        print(f"  IMU:    {len(data['imu_acc'])} samples @ {data['imu_sample_rate']:.1f} Hz (actual)")
    else:
        print(f"  IMU:    none")

    # Common anchor for cross-referencing with the viewer: the real timestamp.
    # 'CSV row' is the positional row index (NOT the video frame — the camera_data.csv
    # 'frame' column is renumbered/inconsistent across sessions). The estimated video
    # frame lets you scrub the raw mp4: video_frame ≈ round(t_offset × video_fps).
    cam_ts    = data["cam_ts"]
    video_fps = data.get("video_fps")

    print(f"\n  Detected {len(contacts)} contact event(s):")
    for i, (contact, phases, feats) in enumerate(zip(contacts, all_phases, all_features)):
        cfi   = contact["cam_frame_idx"]
        t_off = float(cam_ts[cfi] - cam_ts[0])
        vframe = f"~{round(t_off * video_fps)}" if video_fps else "n/a (no video)"
        print(f"\n  ── Contact {i+1} ──")
        print(f"     CSV row {cfi}  |  t={t_off:.3f}s  |  video frame {vframe}"
              + (f"  (@ {video_fps:.0f} fps)" if video_fps else ""))
        print(f"     IPF confidence (QC): {contact['ipf_value']:.3f}")
        if data["has_imu"]:
            print(f"     Gyro at contact: {feats.get('imu_gyro_at_contact', float('nan')):.1f} °/s")
            print(f"     Peak gyro: {feats.get('imu_peak_gyro_dps', float('nan')):.1f} °/s")
            print(f"     Peak acc: {feats.get('imu_peak_acc_g', float('nan')):.2f} G "
                  f"(accel_range ±{feats.get('accel_range', '?')}g)")

        print(f"\n     Acceleration phase: "
              f"{phases['acceleration_duration_s']*1000:.0f} ms "
              f"(max extension → contact)")

        print(f"\n     Elbow angle:")
        print(f"     Max extension: {feats['cam_angle_max_cocking_deg']:.1f}°")
        print(f"     At contact:    {feats['cam_angle_at_contact_deg']:.1f}°")
        print(f"     ROM (phase):   {feats['cam_angle_rom_deg']:.1f}°")


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def _find_manual_contacts_file(session_folder: Path) -> Path | None:
    """Locate manual_contacts.csv. Search order: the session folder, then up to 3 parent
    levels, then the directory containing extraction.py. Returns the first that exists."""
    candidates = [session_folder]
    p = session_folder
    for _ in range(3):                       # up to 3 levels up
        p = p.parent
        candidates.append(p)
    candidates.append(Path(__file__).resolve().parent)
    for c in candidates:
        f = c / "manual_contacts.csv"
        if f.exists():
            return f
    return None


def _load_manual_contacts(session_folder: Path, data: dict) -> list[dict] | None:
    """Hand-marked contacts (Coleman method) from manual_contacts.csv, replacing IPF.

    Returns a list of contact dicts (same shape as detect_contact_frames), or None if
    there is no file or no matching row — the caller then falls back to IPF.

    IMPORTANT — contact_ts in the CSV is an OFFSET in seconds from the START of the
    session's video: exactly the 't = X.XXXs' shown in the dataprocessing HTML viewer,
    which is cam_ts - cam_ts[0]. The absolute value on the cam_ts clock is therefore
    cam_ts[0] + offset. (extraction and dataprocessing both drop the same NaN rows, so
    their cam_ts[0] match.) landing_ts is reserved for a future follow-through feature.
    """
    path = _find_manual_contacts_file(session_folder)
    if path is None:
        return None
    try:
        dfm = pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}")
        return None
    if not {"session_name", "contact_ts"}.issubset(dfm.columns):
        print(f"  [WARN] {path.name} missing 'session_name'/'contact_ts' columns")
        return None

    rows = dfm[dfm["session_name"].astype(str) == str(data["session_name"])]
    rows = rows.dropna(subset=["contact_ts"])
    if len(rows) == 0:
        return None

    cam_ts = data["cam_ts"]
    t0 = float(cam_ts[0])
    imu_ts = data["imu_ts"] if data["has_imu"] else None

    contacts = []
    for _, r in rows.iterrows():
        offset = float(r["contact_ts"])
        abs_ts = t0 + offset                                   # offset → absolute (cam_ts clock)
        cam_frame_idx = int(np.argmin(np.abs(cam_ts - abs_ts)))
        imu_idx = int(np.argmin(np.abs(imu_ts - abs_ts))) if imu_ts is not None else None
        src = str(r["source"]) if "source" in dfm.columns and pd.notna(r.get("source")) else "manual"
        contacts.append({
            "imu_idx":         imu_idx,
            "cam_frame_idx":   cam_frame_idx,
            "contact_ts":      abs_ts,
            "ipf_value":       np.nan,          # not using IPF for manual contacts
            "acc_at_contact":  None,
            "gyro_at_contact": None,
            "source":          src,
        })
        print(f"  [INFO] Using manual contact: t={offset:.3f}s (source: {src})")
    return contacts


def process_session(session_folder: Path, verbose: bool = True, force_camera_only: bool = False) -> list[dict]:
    """
    Process one session folder, return a list of features (one entry per detected spike).
    """
    session_folder = Path(session_folder)
    if verbose:
        print(f"\nProcessing: {session_folder.name}")

    data = load_session(session_folder, force_camera_only=force_camera_only)
    if data is None:
        return []

    # PART 1: prefer the hand-marked contact (manual_contacts.csv); if absent -> IPF.
    contacts = _load_manual_contacts(session_folder, data)
    if contacts is None:
        print(f"  [WARN] No manual contact for {data['session_name']}, falling back to IPF")
        contacts = detect_contact_frames(data)
    if len(contacts) == 0:
        print(f"  [SKIP] No contact event detected")
        return []

    valid_contacts = []
    all_phases   = []
    all_features = []

    for i, contact in enumerate(contacts):
        phases = segment_phases(data, contact)
        if phases is None:
            print(f"  [SKIP] Contact {i+1}/{len(contacts)} (frame {contact['cam_frame_idx']}) "
                  f"skipped: not enough frames before contact to segment the phases "
                  f"(the contact may be too close to the start of the video, or "
                  f"camera_data.csv/imu_data.csv are clock-misaligned so the contact maps "
                  f"to the wrong frame — try re-running with --no-imu).")
            continue

        feats = extract_features(data, contact, phases)
        valid_contacts.append(contact)
        all_phases.append(phases)
        all_features.append(feats)

    if verbose:
        print_session_summary(data, valid_contacts, all_phases, all_features)

    return all_features


def process_all_sessions(recording_dir: Path, verbose: bool = False, force_camera_only: bool = False) -> pd.DataFrame:
    """
    Process all session folders in recording_dir.
    Return a DataFrame with all features.
    """
    recording_dir = Path(recording_dir)
    session_folders = sorted([
        d for d in recording_dir.iterdir()
        if d.is_dir() and (d / "camera_data.csv").exists()
    ])

    if len(session_folders) == 0:
        print(f"No session found in: {recording_dir}")
        return pd.DataFrame()

    print(f"Found {len(session_folders)} session(s) in {recording_dir}")

    all_rows = []
    for folder in session_folders:
        rows = process_session(folder, verbose=verbose, force_camera_only=force_camera_only)
        all_rows.extend(rows)

    if len(all_rows) == 0:
        print("No features extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} contact events from {len(session_folders)} sessions")
    return df


def find_sessions_recursive(root: Path) -> tuple[list[Path], list[Path]]:
    """RECURSIVELY scan the whole tree under root to find session folders.

    A session folder = a folder CONTAINING a camera_data.csv file (at any depth).
    Used for nested datasets, e.g.:
        Zipped data/Primary sessions/Amatures/Thinh_20260706_211732_primary/

    - Skip hidden folders (names starting with a dot).
    - Do not descend into a folder already identified as a session.

    Returns (session_list, suspected_broken_list) — suspected broken = a folder whose
    name looks like a session ({name}_{yyyymmdd}_{hhmmss}) but lacks camera_data.csv.
    """
    sessions, suspicious = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden folders — edit dirnames in place so os.walk won't descend into them
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        here = Path(dirpath)

        if "camera_data.csv" in filenames:
            sessions.append(here)
            dirnames[:] = []          # a session does not nest another session
            continue

        # Looks like a session (has a timestamp in the name) but is missing the file
        if here != root and TIMESTAMP_IN_NAME_RE.search(here.name):
            suspicious.append(here)

    return sorted(sessions), sorted(suspicious)


def process_root_recursive(root: Path, verbose: bool = False,
                           force_camera_only: bool = False) -> pd.DataFrame:
    """Recursively scan root, print a report grouped by parent folder, then extract features."""
    root = Path(root)
    if not root.exists():
        print(f"[ERROR] Folder not found: {root}")
        return pd.DataFrame()

    sessions, suspicious = find_sessions_recursive(root)

    if not sessions:
        print(f"No session found (folder containing camera_data.csv) in: {root}")
        return pd.DataFrame()

    # ── Discovery report, grouped by relative parent path ─────────────
    groups: dict[str, list[str]] = {}
    for s in sessions:
        try:
            rel_parent = s.relative_to(root).parent
        except ValueError:
            rel_parent = Path(".")
        key = str(rel_parent).replace("\\", "/") if str(rel_parent) != "." else "(root)"
        groups.setdefault(key, []).append(s.name)

    print("=" * 70)
    print(f"  RECURSIVE SCAN: {root}")
    print("=" * 70)
    width = max((len(k) for k in groups), default=10)
    width = max(width, len("TOTAL"))
    for key in sorted(groups):
        names = sorted(groups[key])
        print(f"  {key.replace('/', ' / '):<{width + 6}}: {len(names)} session")
        for n in names:
            print(f"       - {n}")
    print("  " + "─" * (width + 22))
    print(f"  {'TOTAL':<{width + 6}}: {len(sessions)} session")

    # session_name = FOLDER NAME → duplicate names would break the metadata.csv join
    from collections import Counter
    dups = [n for n, c in Counter(s.name for s in sessions).items() if c > 1]
    if dups:
        print(f"\n  [WARN] {len(dups)} DUPLICATE session name(s) across different branches —")
        print("         session_name would be ambiguous when joining metadata.csv:")
        for n in dups:
            for s in sessions:
                if s.name == n:
                    print(f"           {n}  ←  {s.relative_to(root)}")

    if suspicious:
        print(f"\n  [WARN] {len(suspicious)} folder(s) look like a session but are MISSING "
              "camera_data.csv (possibly broken sessions):")
        for s in suspicious:
            print(f"       - {s.relative_to(root)}")

    # ── Extract ───────────────────────────────────────────────────────
    print()
    all_rows = []
    for folder in sessions:
        all_rows.extend(process_session(folder, verbose=verbose,
                                        force_camera_only=force_camera_only))

    if not all_rows:
        print("No features extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} contact events from "
          f"{df['session_name'].nunique()}/{len(sessions)} session "
          f"(sessions with no contact do not appear in the output)")
    return df


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Biomechanical feature extraction from volleyball spike sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One session (verbose, no file saved):
  python feature_extraction.py recording/Hoang_20260612_110759

  # All sessions, save to CSV:
  python feature_extraction.py recording/ --all --output features.csv

  # Quiet output only:
  python feature_extraction.py recording/ --all --output features.csv --quiet

  # RECURSIVELY scan a nested folder tree (the real dataset):
  python extraction.py --root "../Zipped data" --output features.csv
        """
    )
    parser.add_argument("input", nargs="?", default=None,
        help="Session folder or recording folder (if using --all). "
             "Leave empty when using --root.")
    parser.add_argument("--all", action="store_true",
        help="Process all sub-folders in input (FLAT 1-level scan)")
    parser.add_argument("--root", default=None,
        help="RECURSIVELY scan the whole folder tree under this path. "
             "A session = a folder containing camera_data.csv, at any depth. "
             "Used for nested datasets like 'Zipped data/Primary sessions/Amatures/...'")
    parser.add_argument("--output", default=None,
        help="Output CSV path (default: print to terminal)")
    parser.add_argument("--quiet", action="store_true",
        help="Print only the summary, not the per-session detail")
    parser.add_argument("--no-imu", action="store_true",
        help="Ignore imu_data.csv, detect contact from the camera only "
             "(use when camera_data.csv was regenerated independently and is clock-misaligned with imu_data.csv)")

    args = parser.parse_args()

    # Exactly ONE of the two: the positional input (old mode) or --root (recursive mode)
    if args.root and args.input:
        parser.error("Do not use both 'input' and --root — pick one.")
    if not args.root and not args.input:
        parser.error("Need one of: a session/folder path, or --root <path>.")

    if args.root:
        df = process_root_recursive(Path(args.root), verbose=not args.quiet,
                                    force_camera_only=args.no_imu)
    elif args.all:
        input_path = Path(args.input)
        df = process_all_sessions(input_path, verbose=not args.quiet, force_camera_only=args.no_imu)
    else:
        input_path = Path(args.input)
        rows = process_session(input_path, verbose=True, force_camera_only=args.no_imu)
        df = pd.DataFrame(rows)

    if df.empty:
        sys.exit(1)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.4f")
        print(f"\n✅ Features saved to: {out_path}")
        print(f"   {len(df)} rows × {len(df.columns)} columns")
        print(f"\nFeatures:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            print(f"   {col:<45} ({non_null}/{len(df)} non-null)")
    else:
        print("\n--- FEATURES OUTPUT ---")
        print(df.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()