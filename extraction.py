"""
feature_extraction.py — Biomechanical Feature Extraction Pipeline
==================================================================
Step 1: Automatically detect the contact frame (ball-contact moment) via IPF
Step 2: Segment the spike into 3 phases (Cocking / Acceleration / Follow-through)
Step 3: Extract features from each phase

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

# EMA smoothing (same as dataprocessing.py for consistency)
SMOOTH_SH = 0.85
SMOOTH_EL = 0.50
SMOOTH_WR = 0.30

# IPF detection
IPF_WINDOW     = 5    # samples per side for the local deviation (~50ms at 100Hz)
MIN_PEAK_DIST  = 50   # min distance between 2 peaks (500ms) — avoid detecting one spike twice
GYRO_THRESHOLD_STD = 1.5  # number of std above the mean to detect the start of the cocking phase

# Phase boundaries (measured from the contact frame, in SECONDS — converted to a
# sample count using each session's actual sample rate, because the IMU BLE link
# drops packets so the rate varies 22-68Hz instead of a fixed 100Hz)
PRE_CONTACT_WINDOW_S  = 1.5  # seconds before contact = cocking + acceleration
POST_CONTACT_WINDOW_S = 1.0  # seconds after contact = follow-through

# ── Sensor limits (LSM9DS1 via the Arduino_LSM9DS1 library) ─────────
# CTRL_REG6_XL = 0x70 → ±4g per axis, the range CANNOT be changed via the library.
# Saturated samples are "clamped" at 4.0 → magnitude ceiling = sqrt(3)*4 = 6.93G.
ACC_CLIP_THRESHOLD_G = 3.99

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


def _jerk_rms_only(sig: np.ndarray, t: np.ndarray) -> float:
    """RMS jerk for a short segment (used per phase). NaN if it can't be computed."""
    if sig is None or t is None or len(sig) < 3 or len(t) != len(sig):
        return np.nan
    if not np.all(np.diff(t) > 0):
        return np.nan
    return float(np.sqrt(np.mean(np.gradient(sig, t) ** 2)))


# ──────────────────────────────────────────────────────────────────────
# STEP A: READ AND PREPARE THE DATA
# ──────────────────────────────────────────────────────────────────────

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
    df_cam = pd.read_csv(cam_path)
    if len(df_cam) < 30:
        print(f"  [SKIP] Camera data too short ({len(df_cam)} rows): {session_folder.name}")
        return None

    cam_ts = df_cam["timestamp"].values
    cam_fps = 1.0 / np.median(np.diff(cam_ts)) if len(cam_ts) > 1 else 60.0
    cam_fps = float(np.clip(cam_fps, 20, 120))

    # EMA filter on the camera coordinates (same as dataprocessing.py)
    elbow_angles, wrist_positions, arm_length = _compute_kinematics(df_cam)

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

            # ── CLIPPING (FIX 3) ─────────────────────────────────────────
            # The LSM9DS1 accel hard-clips at ±4g PER AXIS → check each axis, not the
            # magnitude (the magnitude can be <6.93 while an axis is already clipped).
            axes = df_imu[["ax", "ay", "az"]].values
            imu_clip_mask = (np.abs(axes) >= ACC_CLIP_THRESHOLD_G).any(axis=1)
            clip_fraction = float(imu_clip_mask.mean())

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
        "n_contacts_rejected": 0,
        "imu_sample_rate": imu_sample_rate if has_imu else None,
        "imu_acc":      imu_acc_smooth  if has_imu else None,
        "imu_gyro":     imu_gyro_smooth if has_imu else None,
        "imu_acc_raw":  imu_acc_clean   if has_imu else None,
        "imu_gyro_raw": imu_gyro        if has_imu else None,
    }


def _compute_kinematics(df_cam: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """EMA filter + elbow-angle computation for the whole session.

    Returns 3 things:
    - angles:          (N,)   elbow angle per frame (unchanged from before)
    - wrist_positions: (N,3)  wrist coordinates relative to the shoulder (wr_rel) after EMA
    - arm_length:      scalar median shoulder→elbow distance over the session,
                       used as the "measuring unit" to normalize the wrist-position features.
                       Median (not mean) to be robust to noisy frames.
    """
    sh_x = sh_y = sh_z = None
    el_x = el_y = el_z = None
    wr_x = wr_y = wr_z = None
    angles = []
    wrist_positions = []
    se_dists = []  # shoulder→elbow distance per frame

    for _, row in df_cam.iterrows():
        raw = [
            [ row["shoulder_x"], -row["shoulder_y"], -row["shoulder_z"]],
            [ row["elbow_x"],    -row["elbow_y"],    -row["elbow_z"]],
            [ row["wrist_x"],    -row["wrist_y"],    -row["wrist_z"]],
        ]
        if sh_x is None:
            sh_x, sh_y, sh_z = raw[0]
            el_x, el_y, el_z = raw[1]
            wr_x, wr_y, wr_z = raw[2]

        sh_x = SMOOTH_SH*sh_x + (1-SMOOTH_SH)*raw[0][0]
        sh_y = SMOOTH_SH*sh_y + (1-SMOOTH_SH)*raw[0][1]
        sh_z = SMOOTH_SH*sh_z + (1-SMOOTH_SH)*raw[0][2]
        el_x = SMOOTH_EL*el_x + (1-SMOOTH_EL)*raw[1][0]
        el_y = SMOOTH_EL*el_y + (1-SMOOTH_EL)*raw[1][1]
        el_z = SMOOTH_EL*el_z + (1-SMOOTH_EL)*raw[1][2]
        wr_x = SMOOTH_WR*wr_x + (1-SMOOTH_WR)*raw[2][0]
        wr_y = SMOOTH_WR*wr_y + (1-SMOOTH_WR)*raw[2][1]
        wr_z = SMOOTH_WR*wr_z + (1-SMOOTH_WR)*raw[2][2]

        el_rel = np.array([el_x - sh_x, el_y - sh_y, el_z - sh_z])
        wr_rel = np.array([wr_x - sh_x, wr_y - sh_y, wr_z - sh_z])

        v1 = el_rel
        v2 = wr_rel - el_rel
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = 180 - np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        angles.append(float(angle))

        wrist_positions.append(wr_rel)
        se_dists.append(float(np.linalg.norm(el_rel)))  # |shoulder→elbow|

    arm_length = float(np.median(se_dists)) if len(se_dists) > 0 else np.nan
    return np.array(angles), np.array(wrist_positions), arm_length


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
    Split the spike into 3 phases based on the contact frame.

    Phase 1 — ARM COCKING:
        Start: gyro exceeds the threshold (= mean + 1.5 std)
        End:   elbow angle reaches its maximum in the pre-contact window

    Phase 2 — ARM ACCELERATION:
        Start: end of Phase 1
        End:   contact frame (peak IPF)

    Phase 3 — FOLLOW-THROUGH:
        Start: contact frame
        End:   gyro returns to baseline (or a fixed window)

    Returns a dict of the indices and timestamps of each phase boundary.
    """
    cam_ts    = data["cam_ts"]
    angles    = data["elbow_angles"]
    n_cam     = len(angles)
    contact_cam = contact["cam_frame_idx"]

    # Analysis window converted directly from seconds to camera frames (using the
    # session's own cam_fps, unrelated to the IMU sample rate)
    cam_fps   = data["cam_fps"]
    pre_window_cam = int(PRE_CONTACT_WINDOW_S * cam_fps)
    post_window_cam = int(POST_CONTACT_WINDOW_S * cam_fps)

    # Indices in camera space
    window_start = max(0, contact_cam - pre_window_cam)
    window_end   = min(n_cam - 1, contact_cam + post_window_cam)

    if window_start >= contact_cam:
        return None

    # ── Phase 1/2 boundary: elbow angle reaches its max in the pre-contact window ──
    # This is the "arm cocked" point — the elbow furthest back before the swing
    pre_angles = angles[window_start:contact_cam]
    if len(pre_angles) == 0:
        return None

    # Max angle in the pre-contact window
    max_angle_local_idx = int(np.argmax(pre_angles))
    cocking_end_cam = window_start + max_angle_local_idx

    # ── Phase 1 start: when movement begins (gyro exceeds the threshold) ──
    if data["has_imu"]:
        gyro = data["imu_gyro"]
        imu_ts = data["imu_ts"]
        baseline_gyro = np.mean(gyro[:15])
        std_gyro      = np.std(gyro[:15])
        threshold_gyro = baseline_gyro + GYRO_THRESHOLD_STD * std_gyro

        # Find the IMU point before contact where gyro exceeds the threshold.
        # Cut the window by REAL TIME on the arduino axis (searchsorted) instead of
        # seconds × average_sample_rate: the average sample rate is dragged down by
        # dropped-packet gaps, while locally the sensor still runs ~100Hz.
        imu_t = data["imu_t"]
        contact_imu = contact["imu_idx"]
        pre_imu_start = int(np.searchsorted(
            imu_t, imu_t[contact_imu] - PRE_CONTACT_WINDOW_S, side="left"))
        pre_imu_start = max(0, min(pre_imu_start, contact_imu))
        gyro_pre = gyro[pre_imu_start:contact_imu]

        # Find the first time gyro exceeds the threshold
        above_threshold = np.where(gyro_pre > threshold_gyro)[0]
        if len(above_threshold) > 0:
            cocking_start_imu = pre_imu_start + above_threshold[0]
            cocking_start_ts  = imu_ts[cocking_start_imu]
            cocking_start_cam = int(np.argmin(np.abs(cam_ts - cocking_start_ts)))
        else:
            cocking_start_cam = window_start
    else:
        cocking_start_cam = window_start

    # ── Phase 3 end: signal returns to baseline after contact ──
    followthrough_end_cam = window_end  # default: fixed window

    return {
        # Camera frame indices
        "cocking_start_cam":    int(max(0, cocking_start_cam)),
        "cocking_end_cam":      int(cocking_end_cam),
        "acceleration_end_cam": int(contact_cam),
        "followthrough_end_cam": int(followthrough_end_cam),

        # Timestamps
        "cocking_start_ts":    float(cam_ts[max(0, cocking_start_cam)]),
        "contact_ts":          float(contact["contact_ts"]),
        "followthrough_end_ts": float(cam_ts[followthrough_end_cam]),

        # Phase durations (seconds)
        "cocking_duration_s":       float(cam_ts[cocking_end_cam] - cam_ts[max(0, cocking_start_cam)]),
        "acceleration_duration_s":  float(cam_ts[contact_cam] - cam_ts[cocking_end_cam]),
        "followthrough_duration_s": float(cam_ts[followthrough_end_cam] - cam_ts[contact_cam]),
    }


# ──────────────────────────────────────────────────────────────────────
# STEP D: FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def extract_features(data: dict, contact: dict, phases: dict) -> dict:
    """
    Extract all biomechanical features from one spike.

    Features are grouped by source:
    - [IMU] : from the accelerometer and gyroscope
    - [CAM] : from the elbow angle (camera)
    - [PHASE]: timing of each phase

    ⚠️ RELIABILITY WHEN THE ACCEL IS CLIPPED
    =========================================
    The LSM9DS1 accelerometer hard-clips at ±4g per axis (the Arduino_LSM9DS1
    library does not allow changing the range), magnitude ceiling = sqrt(3)*4 =
    6.93G. Measured on the dataset: a median of 13% of samples are clipped, and 95%
    of sessions have the EXACT sample at the peak clipped.

    NOT trustworthy when clip_fraction is high / peak_is_clipped = 1
    (values are chopped; hard and very-hard spikes both read ~6.93G, so it COMPRESSES
     away the difference between skill groups):
        imu_peak_acc_g, imu_acc_at_contact, imu_ratio_gyro_acc,
        imu_time_to_peak_acc_s, ipf_confidence
        (ipf_confidence is affected because IPF = local_dev(acc) × local_dev(gyro))

    CLIP-SAFE — gyro does not hit its ±2000dps ceiling (measured max 0.23% of samples):
        imu_peak_gyro_dps, imu_gyro_at_contact, imu_gyro_impulse_duration_s,
        all cam_*  and  all phase_*

    Companion flag columns for filtering: clip_fraction, peak_is_clipped,
    contact_near_gap, sync_quality_ms, sync_lowlat_fraction, n_contacts_rejected.

    📐 JERK — MOTION SMOOTHNESS
    ==================================
    Jerk = time derivative of acceleration.

    High jerk = jerky motion, abrupt acceleration changes.
    Low jerk = smooth, controlled motion.
    Hypothesis: skilled players are smoother in the cocking phase (controlled setup)
    but may be jerkier in the acceleration phase (a decisive arm swing).
    imu_acc_jerk_* is NOT TRUSTWORTHY when clip_fraction is high — the chopped signal
    produces a fake derivative.

    Specifically: the accel clips at ±4g on 69% of contacts. Differentiating a
    flat-topped segment that then springs back produces a very large FAKE jerk right
    at the clip. So jerk is computed from BOTH sources for cross-checking:
        imu_gyro_jerk_*  (°/s²) — TRUSTWORTHY, gyro doesn't hit its ±2000dps ceiling
        imu_acc_jerk_*   (g/s)  — only use when clip_fraction is low

    *_jerk_dimensionless = (T³/A²)·∫jerk²dt — normalized to remove duration and
    amplitude, so it compares across spikes of different strength (see _jerk_stats).
    Comparing raw |jerk| would just re-measure speed, not smoothness.

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

    # Shorthand for the phase indices
    ck_s  = phases["cocking_start_cam"]
    ck_e  = phases["cocking_end_cam"]
    acc_e = phases["acceleration_end_cam"]  # = contact frame
    ft_e  = phases["followthrough_end_cam"]

    # ── ELBOW ANGLE FEATURES ──────────────────────────────────────────
    # Whole session
    angle_min  = float(np.min(angles))
    angle_max  = float(np.max(angles))
    angle_rom  = angle_max - angle_min  # Range of Motion

    # At the contact moment
    angle_at_contact = float(angles[acc_e]) if acc_e < len(angles) else np.nan

    # Max angle in the cocking phase
    cocking_angles = angles[ck_s:ck_e+1] if ck_e > ck_s else angles[ck_s:ck_s+1]
    angle_max_cocking = float(np.max(cocking_angles)) if len(cocking_angles) > 0 else np.nan

    # Min angle in the acceleration phase (pre-contact)
    accel_angles = angles[ck_e:acc_e+1] if acc_e > ck_e else angles[acc_e:acc_e+1]
    angle_min_accel = float(np.min(accel_angles)) if len(accel_angles) > 0 else np.nan

    # Angular rate of change per phase (deg/s)
    def angular_rate(idx_start, idx_end):
        if idx_end <= idx_start or idx_end >= len(angles) or idx_start >= len(angles):
            return np.nan
        dt = cam_ts[idx_end] - cam_ts[idx_start]
        if dt <= 0:
            return np.nan
        return float((angles[idx_end] - angles[idx_start]) / dt)

    angle_rate_cocking       = angular_rate(ck_s, ck_e)     # + = opening up
    angle_rate_acceleration  = angular_rate(ck_e, acc_e)    # - = snapping down
    angle_rate_followthrough = angular_rate(acc_e, ft_e)

    # ── WRIST PATH FEATURES (normalized by arm_length) ─────────────────
    # Divide by arm_length to remove the effect of player-camera distance:
    # the unit becomes "number of arm lengths" instead of raw pixels.
    wrist_pos  = data.get("wrist_positions")
    arm_length = data.get("arm_length", np.nan)

    cam_wrist_path_length_norm  = np.nan
    cam_wrist_displacement_norm = np.nan
    cam_wrist_peak_speed_norm   = np.nan

    # Only compute when arm_length is valid (>0 and not NaN) — avoid div-by-0 / bad sessions
    if wrist_pos is not None and np.isfinite(arm_length) and arm_length > 0:
        lo = max(0, ck_s)
        hi = min(len(wrist_pos) - 1, ft_e)
        if hi > lo:
            seg     = wrist_pos[lo:hi+1]                  # (M,3) wrist positions during the spike
            seg_ts  = cam_ts[lo:hi+1]
            steps   = np.diff(seg, axis=0)                # displacement vectors between frames
            step_d  = np.linalg.norm(steps, axis=1)       # length of each step

            # Path length: total distance the wrist travels over the whole spike
            cam_wrist_path_length_norm = float(np.sum(step_d) / arm_length)

            # Displacement: straight-line distance from cocking start → contact
            contact_local = min(max(acc_e - lo, 0), len(seg) - 1)
            disp = np.linalg.norm(seg[contact_local] - seg[0])
            cam_wrist_displacement_norm = float(disp / arm_length)

            # Peak speed: max(|Δpos| / Δt) — the fastest wrist speed
            dt = np.diff(seg_ts)
            valid = dt > 0
            if np.any(valid):
                speed = step_d[valid] / dt[valid]         # pixels/second
                cam_wrist_peak_speed_norm = float(np.max(speed) / arm_length)

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

        # Peak values
        imu_feats["imu_peak_acc_g"]     = float(np.max(acc_window))
        imu_feats["imu_peak_gyro_dps"]  = float(np.max(gyro_window))

        # Values at contact
        imu_feats["imu_acc_at_contact"]  = float(acc[contact_imu])
        imu_feats["imu_gyro_at_contact"] = float(gyro[contact_imu])

        # Time to peak acc (from cocking start to peak) — measured on the arduino axis
        peak_acc_local = int(np.argmax(acc_window))
        imu_feats["imu_time_to_peak_acc_s"] = float(
            t_window[peak_acc_local] - t_window[0]
        )

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

        # Ratio angular/linear at contact
        if acc[contact_imu] > 0:
            imu_feats["imu_ratio_gyro_acc"] = float(gyro[contact_imu] / acc[contact_imu])
        else:
            imu_feats["imu_ratio_gyro_acc"] = np.nan

        # IPF value at contact (confidence score)
        imu_feats["ipf_confidence"] = float(contact["ipf_value"])

        # ── DATA QUALITY FLAGS (FIX 3) ───────────────────────────────
        # peak_is_clipped: whether the sample AT contact has any axis hitting the ±4g ceiling.
        # If =1, imu_acc_at_contact / imu_peak_acc_g / imu_ratio_gyro_acc for this
        # spike are CHOPPED values, not the true values.
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

    # ── JERK / SMOOTHNESS ────────────────────────────────────────────
    # The spike window is given by CAMERA indices (ck_s→ft_e). Convert to IMU indices
    # via imu_ts (PC scale — same reference frame as cam_ts), but DIFFERENTIATE on
    # imu_t (arduino scale, monotonic, correct physical dt).
    jerk_feats = {}
    jerk_feats.update(_jerk_stats(None, None, "imu_gyro"))   # default NaN
    jerk_feats.update(_jerk_stats(None, None, "imu_acc"))
    jerk_feats["imu_gyro_jerk_rms_cocking"]       = np.nan
    jerk_feats["imu_gyro_jerk_rms_acceleration"]  = np.nan
    jerk_feats["imu_gyro_jerk_rms_followthrough"] = np.nan

    if has_imu:
        imu_ts_pc = data["imu_ts"]
        imu_t_ard = data["imu_t"]
        acc_s, gyro_s = data["imu_acc"], data["imu_gyro"]

        def _imu_slice(cam_i0: int, cam_i1: int) -> slice:
            """IMU indices covering the time span between 2 camera frames."""
            t0, t1 = float(cam_ts[cam_i0]), float(cam_ts[cam_i1])
            if t1 < t0:
                t0, t1 = t1, t0
            i0 = int(np.searchsorted(imu_ts_pc, t0, side="left"))
            i1 = int(np.searchsorted(imu_ts_pc, t1, side="right"))
            return slice(max(0, i0), min(len(imu_t_ard), i1))

        # Whole spike: cocking_start → followthrough_end
        sl = _imu_slice(ck_s, ft_e)
        tw = imu_t_ard[sl]
        jerk_feats.update(_jerk_stats(gyro_s[sl], tw, "imu_gyro"))
        jerk_feats.update(_jerk_stats(acc_s[sl],  tw, "imu_acc"))

        # Per phase (gyro only — the trustworthy source, not clipped at ±4g)
        for name, (a, b) in {
            "cocking":       (ck_s,  ck_e),
            "acceleration":  (ck_e,  acc_e),
            "followthrough": (acc_e, ft_e),
        }.items():
            s = _imu_slice(a, b)
            jerk_feats[f"imu_gyro_jerk_rms_{name}"] = _jerk_rms_only(
                gyro_s[s], imu_t_ard[s])

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

    w0, w1 = min(ck_s, ft_e), max(ck_s, ft_e) + 1     # spike window
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

    # ── PHASE TIMING FEATURES ────────────────────────────────────────
    total_dur = (
        phases["cocking_duration_s"]
        + phases["acceleration_duration_s"]
        + phases["followthrough_duration_s"]
    )
    phase_feats = {
        "phase_cocking_duration_s":       phases["cocking_duration_s"],
        "phase_acceleration_duration_s":  phases["acceleration_duration_s"],
        "phase_followthrough_duration_s": phases["followthrough_duration_s"],
        "phase_total_duration_s":         total_dur,
        # Time share of each phase (% of total time)
        "phase_cocking_pct":       phases["cocking_duration_s"] / total_dur * 100 if total_dur > 0 else np.nan,
        "phase_acceleration_pct":  phases["acceleration_duration_s"] / total_dur * 100 if total_dur > 0 else np.nan,
    }

    # ── COMBINE ALL FEATURES ─────────────────────────────────────────
    features = {
        "session_name":             data["session_name"],
        "has_imu":                  int(has_imu),
        "imu_sample_rate":          float(data["imu_sample_rate"]) if data["imu_sample_rate"] is not None else np.nan,

        # Data / sync quality — used to filter sessions before analysis
        "imu_timebase":         data["imu_timebase"] if has_imu else "",
        "sync_quality_ms":      float(data["sync_quality_ms"])      if has_imu else np.nan,
        "sync_lowlat_fraction": float(data["sync_lowlat_fraction"]) if has_imu else np.nan,
        "clip_fraction":        float(data["clip_fraction"])        if has_imu else np.nan,
        "n_contacts_rejected":  int(data.get("n_contacts_rejected", 0)),

        "contact_frame":            int(acc_e),
        "contact_ts_offset_s":      float(cam_ts[acc_e] - cam_ts[0]),

        # Elbow angle
        "cam_angle_at_contact_deg":     angle_at_contact,
        "cam_angle_max_cocking_deg":    angle_max_cocking,
        "cam_angle_min_acceleration_deg": angle_min_accel,
        "cam_angle_rom_deg":            angle_rom,
        "cam_angle_rate_cocking_dps":   angle_rate_cocking,
        "cam_angle_rate_acceleration_dps": angle_rate_acceleration,
        "cam_angle_rate_followthrough_dps": angle_rate_followthrough,

        # Wrist path (normalized by arm length)
        "cam_wrist_path_length_norm":   cam_wrist_path_length_norm,
        "cam_wrist_displacement_norm":  cam_wrist_displacement_norm,
        "cam_wrist_peak_speed_norm":    cam_wrist_peak_speed_norm,

        **phase_feats,
        **imu_feats,
        **jerk_feats,
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

    print(f"\n  Detected {len(contacts)} contact event(s):")
    for i, (contact, phases, feats) in enumerate(zip(contacts, all_phases, all_features)):
        print(f"\n  ── Contact {i+1} ──")
        print(f"     Camera frame: {contact['cam_frame_idx']}")
        print(f"     IPF confidence: {contact['ipf_value']:.3f}")
        if data["has_imu"]:
            print(f"     Acc at contact: {feats.get('imu_acc_at_contact', 'N/A'):.2f} G")
            print(f"     Gyro at contact: {feats.get('imu_gyro_at_contact', 'N/A'):.1f} °/s")
            print(f"     Peak acc: {feats.get('imu_peak_acc_g', 'N/A'):.2f} G")

        print(f"\n     Phase durations:")
        print(f"     Cocking:       {phases['cocking_duration_s']*1000:.0f} ms")
        print(f"     Acceleration:  {phases['acceleration_duration_s']*1000:.0f} ms")
        print(f"     Follow-through:{phases['followthrough_duration_s']*1000:.0f} ms")

        print(f"\n     Elbow angle:")
        print(f"     Max (cocking):      {feats['cam_angle_max_cocking_deg']:.1f}°")
        print(f"     At contact:         {feats['cam_angle_at_contact_deg']:.1f}°")
        print(f"     Min (acceleration): {feats['cam_angle_min_acceleration_deg']:.1f}°")
        print(f"     ROM:                {feats['cam_angle_rom_deg']:.1f}°")


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

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