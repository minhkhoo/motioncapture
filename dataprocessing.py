"""
dataprocessing.py — Motion Capture Report Generator (unified entry point)
=========================================================================
Can be used in 2 ways:

  1. From a session folder that already has CSVs:
       uv run python dataprocessing.py recording/Hoang_20260612_110759

  2. From any video file (offline analysis):
       uv run python dataprocessing.py spike.mp4
       uv run python dataprocessing.py spike.mp4 --side LEFT --complexity 2

Options (only used when the input is a video file):
  --side        RIGHT | LEFT  (default: RIGHT)
  --complexity  0 | 1 | 2    (default: 1)
  --out         <folder>      custom session output folder
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────
TRAIL_LENGTH = 45
SMOOTH_SH = 0.85
SMOOTH_EL = 0.50
SMOOTH_WR = 0.30


# ──────────────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────────────
def calculate_elbow_angle(shoulder, elbow, wrist):
    v1 = np.array(elbow) - np.array(shoulder)
    v2 = np.array(wrist)  - np.array(elbow)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return 180 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# ──────────────────────────────────────────────────────────────────────
# STEP A: EXTRACT POSE FROM VIDEO
# ──────────────────────────────────────────────────────────────────────
def extract_pose_from_video(
    video_path: Path,
    output_csv: Path,
    side: str = "RIGHT",
    model_complexity: int = 1,
) -> bool:
    """
    Run MediaPipe Pose on the video, exporting both:
      - camera_data.csv  (landmark coordinates)
      - annotated_video.mp4  (original video + skeleton overlay)
    """
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    side = side.upper()
    landmarks = {
        "RIGHT": (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                  mp_pose.PoseLandmark.RIGHT_ELBOW,
                  mp_pose.PoseLandmark.RIGHT_WRIST),
        "LEFT":  (mp_pose.PoseLandmark.LEFT_SHOULDER,
                  mp_pose.PoseLandmark.LEFT_ELBOW,
                  mp_pose.PoseLandmark.LEFT_WRIST),
    }
    if side not in landmarks:
        print(f"[ERROR] --side must be RIGHT or LEFT")
        return False
    SHOULDER, ELBOW, WRIST = landmarks[side]

    # Custom style for the analyzed arm skeleton — more prominent than the full body
    TRACKED_COLOR = (0, 229, 255)   # cyan BGR
    TRACKED_THICK = 4
    OTHER_COLOR   = (80, 80, 80)    # dim gray for the rest

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return False

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  → Video: {w}×{h} @ {src_fps:.1f} FPS · {total} frames")
    print(f"  → Analyzing {side} arm | complexity={model_complexity}")
    print(f"  → Exporting skeleton overlay → annotated_video.mp4")

    # VideoWriter cho annotated output (mp4v = H.264-compatible trong QWebEngineView)
    annotated_path = output_csv.parent / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(annotated_path), fourcc, src_fps, (w, h))

    start_ts = time.perf_counter()
    rows     = 0
    fidx     = 0
    last_pct = -1

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "timestamp",
            "shoulder_x", "shoulder_y", "shoulder_z", "shoulder_visibility",
            "elbow_x",    "elbow_y",    "elbow_z",    "elbow_visibility",
            "wrist_x",    "wrist_y",    "wrist_z",    "wrist_visibility",
        ])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts       = start_ts + (fidx / src_fps)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results  = pose.process(rgb)
            annotated = frame.copy()

            if results.pose_landmarks:
                lm  = results.pose_landmarks.landmark
                s, e, wr_ = lm[SHOULDER], lm[ELBOW], lm[WRIST]

                # Ghi CSV
                writer.writerow([
                    fidx, ts,
                    s.x, s.y, s.z, s.visibility,
                    e.x, e.y, e.z, e.visibility,
                    wr_.x, wr_.y, wr_.z, wr_.visibility,
                ])
                rows += 1

                # ── SKELETON OVERLAY ──────────────────────────────────
                # 1. Draw all pose connections in dim gray (full body context)
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=OTHER_COLOR, thickness=1, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=OTHER_COLOR, thickness=1),
                )

                # 2. Overlay the analyzed arm skeleton — brighter highlight
                sx, sy = int(s.x * w), int(s.y * h)
                ex, ey = int(e.x * w), int(e.y * h)
                wx, wy = int(wr_.x * w), int(wr_.y * h)

                # Connections
                cv2.line(annotated, (sx, sy), (ex, ey), TRACKED_COLOR, TRACKED_THICK)
                cv2.line(annotated, (ex, ey), (wx, wy), TRACKED_COLOR, TRACKED_THICK)

                # Joints
                for (px, py), radius in [((sx, sy), 8), ((ex, ey), 6), ((wx, wy), 8)]:
                    cv2.circle(annotated, (px, py), radius + 2, (0, 0, 0), -1)       # border
                    cv2.circle(annotated, (px, py), radius,     TRACKED_COLOR, -1)   # fill

                # 3. Elbow angle (text overlay)
                ang = calculate_elbow_angle(
                    [s.x, s.y, s.z], [e.x, e.y, e.z], [wr_.x, wr_.y, wr_.z]
                )
                label = f"{ang:.1f} deg"
                lx, ly = ex + 12, ey - 12
                # Shadow
                cv2.putText(annotated, label, (lx + 1, ly + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                # Text
                cv2.putText(annotated, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TRACKED_COLOR, 2)

                # 4. Frame + time watermark (top-left)
                info = f"F:{fidx}  T:{ts - start_ts:.2f}s"
                cv2.putText(annotated, info, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(annotated, info, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            vw.write(annotated)
            fidx += 1

            pct = int(fidx / total * 100) if total > 0 else 0
            if pct != last_pct and pct % 5 == 0:
                last_pct = pct
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  [{bar}] {pct:3d}%  {fidx}/{total} frames, {rows} poses",
                      end="", flush=True)

    print()
    cap.release()
    pose.close()
    vw.release()

    if rows == 0:
        print("  ⚠️  MediaPipe detected no pose. Try --complexity 2.")
        return False

    print(f"  ✅ {rows} data rows ({rows/fidx*100:.1f}% detection rate)")
    print(f"  ✅ annotated_video.mp4 exported")
    return True


# ──────────────────────────────────────────────────────────────────────
# STEP B: READ CSV + EMA FILTER + DATA FUSION
# ──────────────────────────────────────────────────────────────────────
def process_session(session_folder: Path) -> dict | None:
    csv_path   = session_folder / "camera_data.csv"
    imu_path   = session_folder / "imu_data.csv"
    video_path = session_folder / "raw_video.mp4"

    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}")
        return None

    print("  → Reading camera_data.csv ...")
    df = pd.read_csv(csv_path)
    cam_ts = df["timestamp"].values
    frames = df["frame"].values.astype(int)

    # IMU fusion
    accel = np.zeros(len(df))
    has_imu = False
    if imu_path.exists():
        print("  → Data fusion with IMU ...")
        df_imu = pd.read_csv(imu_path)
        has_imu = True
        mag = np.sqrt(df_imu["ax"]**2 + df_imu["ay"]**2 + df_imu["az"]**2)
        clean = np.abs(mag - np.mean(mag.iloc[:15]))
        if "timestamp" in df_imu.columns:
            accel = np.interp(cam_ts, df_imu["timestamp"].values, clean.values)
        else:
            idx = np.linspace(0, len(df) - 1, len(df_imu))
            accel = np.interp(np.arange(len(df)), idx, clean.values)

    # EMA filter
    print("  → EMA filter + angle computation ...")
    sh_x = sh_y = sh_z = None
    el_x = el_y = el_z = None
    wr_x = wr_y = wr_z = None
    trail = []
    fd = {}

    for idx, row in df.iterrows():
        fid = int(row["frame"])
        acc = float(accel[idx])
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

        el_rel = [el_x - sh_x, el_y - sh_y, el_z - sh_z]
        wr_rel = [wr_x - sh_x, wr_y - sh_y, wr_z - sh_z]

        trail.append((wr_rel, acc))
        if len(trail) > TRAIL_LENGTH:
            trail.pop(0)

        ang = calculate_elbow_angle([0, 0, 0], el_rel, wr_rel)
        fd[fid] = {
            "el":          [round(v, 5) for v in el_rel],
            "wr":          [round(v, 5) for v in wr_rel],
            "angle":       round(float(ang), 2),
            "trail":       [[round(v, 5) for v in p] for p, _ in trail],
            "trail_acc":   [round(float(a), 4) for _, a in trail],
            "time_offset": round(float(cam_ts[idx] - cam_ts[0]), 4),
            "acc":         round(float(acc), 4),
        }

    valid = [f for f in frames if f in fd]

    # src_fps MUST match the FPS actually declared in the video file, because the
    # browser plays the <video> at that FPS (not the real capture rate from
    # perf_counter) — a wrong value here makes the video and the graph run at different speeds.
    src_fps = None
    _video_for_fps = video_path if video_path.exists() else session_folder / "annotated_video.mp4"
    if _video_for_fps.exists():
        _cap = cv2.VideoCapture(str(_video_for_fps))
        _real_fps = _cap.get(cv2.CAP_PROP_FPS)
        _cap.release()
        if _real_fps and 0 < _real_fps <= 240:
            src_fps = float(_real_fps)
    if src_fps is None:
        src_fps = float(df["timestamp"].diff().median()**-1) if len(df) > 1 else 60.0
        if src_fps <= 0 or src_fps > 240:
            src_fps = 60.0

    return {
        "session_name": session_folder.name,
        "frames":       [int(f) for f in valid],
        "frame_dict":   {str(k): v for k, v in fd.items()},
        "time_series":  [fd[f]["time_offset"] for f in valid],
        "angle_series": [fd[f]["angle"]       for f in valid],
        "acc_series":   [fd[f]["acc"]          for f in valid],
        "has_imu":      has_imu,
        "has_video":    video_path.exists(),
        "video_path":   str(video_path.resolve()) if video_path.exists() else None,
        "src_fps":      src_fps,
    }


# ──────────────────────────────────────────────────────────────────────
# STEP C: GENERATE HTML (used by the PyQt6 WebEngineView)
# ──────────────────────────────────────────────────────────────────────
def generate_html(data: dict, output_path: Path):
    sn        = data["session_name"]
    has_imu   = data["has_imu"]
    has_video = data["has_video"]
    src_fps   = data.get("src_fps", 60.0)

    # Video src — PyQt6 serves file:// directly via QMediaPlayer, no workaround needed
    video_tag = (
        '<video id="mainVideo" preload="auto" playsinline>'
        f'<source src="raw_video.mp4" type="video/mp4">'
        '</video>'
    ) if has_video else (
        '<div class="no-video">📹 No video attached</div>'
    )

    imu_badge_cls  = "active"  if has_imu else "inactive"
    imu_badge_text = "● IMU FUSED" if has_imu else "○ CAM ONLY"
    chart_title    = "📈 Elbow Angle & IMU Acceleration" if has_imu else "📈 Elbow Angle"

    y2_layout = (
        'yaxis2:{gridcolor:"rgba(0,0,0,0)",title:{text:"Acceleration (G)",font:{size:9,color:"#ff6b35"}},'
        'overlaying:"y",side:"right",tickfont:{color:"#ff6b35",size:9}},'
    ) if has_imu else ""

    acc_trace = (
        'chartTraces.push({type:"scatter",mode:"lines",x:timeSeries,y:accSeries,'
        'line:{color:"#ff6b35",width:1.5},opacity:0.6,name:"Acceleration (G)",yaxis:"y2"});'
    ) if has_imu else ""

    payload = json.dumps({
        "frames":       data["frames"],
        "frame_dict":   data["frame_dict"],
        "time_series":  data["time_series"],
        "angle_series": data["angle_series"],
        "acc_series":   data["acc_series"],
        "has_imu":      has_imu,
        "src_fps":      src_fps,
    }, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Motion Report · {sn}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{{
  --bg:#0b0d11;--surface:#131720;--card:#1b202e;--border:#262f42;
  --accent:#00e5ff;--accent2:#ff6b35;--green:#39d98a;
  --text:#e8edf5;--sub:#6c7d93;
  --mono:"JetBrains Mono",monospace;--ui:system-ui,sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--ui);height:100vh;overflow:hidden;display:flex;flex-direction:column}}
header{{display:flex;align-items:center;justify-content:space-between;padding:11px 22px;
  background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0}}
.h-left{{display:flex;align-items:center;gap:10px}}
.logo{{width:8px;height:8px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 8px var(--accent);animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.4;transform:scale(.7)}}}}
.h-title{{font-size:14px;font-weight:700}}
.badge{{font-family:var(--mono);font-size:10px;color:var(--sub);background:var(--card);
  padding:3px 9px;border-radius:4px;border:1px solid var(--border)}}
.imu{{font-size:10px;font-weight:700;padding:3px 9px;border-radius:4px;border:1px solid}}
.imu.active{{color:var(--green);border-color:var(--green);background:rgba(57,217,138,.08)}}
.imu.inactive{{color:var(--sub);border-color:var(--border)}}
.main{{display:grid;grid-template-columns:400px 1fr;flex:1;min-height:0;overflow:hidden}}
.left{{display:flex;flex-direction:column;background:var(--surface);border-right:1px solid var(--border);overflow:hidden}}
.panel-lbl{{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
  color:var(--sub);padding:9px 14px;border-bottom:1px solid var(--border);flex-shrink:0}}
.vid-wrap{{flex:1;background:#000;display:flex;align-items:center;justify-content:center;
  min-height:0;overflow:hidden;position:relative}}
#mainVideo{{width:100%;height:100%;object-fit:contain;display:block}}
.no-video{{color:var(--sub);font-size:13px}}
#vidOverlay{{position:absolute;inset:0;background:rgba(11,13,17,.85);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:10px;font-size:12px;color:var(--sub);transition:opacity .3s}}
#vidOverlay.hidden{{opacity:0;pointer-events:none}}
.spinner{{width:28px;height:28px;border:2px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.ctrl-area{{flex-shrink:0;padding:12px 14px;background:var(--card);border-top:1px solid var(--border)}}
.finfo{{display:flex;justify-content:space-between;margin-bottom:8px;
  font-family:var(--mono);font-size:11px;gap:8px}}
.fval{{color:var(--accent);font-weight:700}}
.aval{{color:var(--green);font-weight:700}}
.tval{{color:var(--sub)}}
input[type=range]{{-webkit-appearance:none;width:100%;height:3px;background:var(--border);
  border-radius:2px;outline:none;cursor:pointer;margin-bottom:10px}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--accent);box-shadow:0 0 6px var(--accent);
  cursor:pointer;transition:transform .1s}}
input[type=range]::-webkit-slider-thumb:active{{transform:scale(1.4)}}
.ctrls{{display:flex;gap:6px;align-items:center;flex-wrap:nowrap}}
.btn{{background:var(--surface);border:1px solid var(--border);color:var(--text);
  border-radius:5px;cursor:pointer;font-size:12px;padding:5px 10px;
  white-space:nowrap;transition:border-color .12s,background .12s}}
.btn:hover{{border-color:var(--accent);background:var(--border)}}
.btn.primary{{background:var(--accent);border-color:var(--accent);color:#000;
  font-weight:700;padding:5px 14px}}
.btn.primary:hover{{filter:brightness(1.1)}}
select.spd{{background:var(--surface);border:1px solid var(--border);color:var(--text);
  border-radius:5px;font-size:11px;padding:4px 6px;cursor:pointer;
  font-family:var(--mono);margin-left:auto}}
.hints{{display:flex;gap:10px;margin-top:8px}}
.hint{{font-size:9px;color:var(--sub);display:flex;align-items:center;gap:3px}}
kbd{{background:var(--card);border:1px solid var(--border);border-radius:3px;
  padding:1px 4px;font-family:var(--mono);font-size:9px}}
.right{{display:flex;flex-direction:column;min-height:0;overflow:hidden}}
.graphs-row{{display:grid;grid-template-columns:1fr 1fr 1fr;
  flex:0 0 320px;border-bottom:1px solid var(--border);overflow:hidden}}
.gcell{{border-right:1px solid var(--border);display:flex;flex-direction:column;
  overflow:hidden;min-height:0}}
.gcell:last-child{{border-right:none}}
.gtitle{{font-size:9px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
  padding:8px 12px;border-bottom:1px solid var(--border);flex-shrink:0}}
.plotly-cell{{flex:1;min-height:0}}
.chart-area{{flex:1;display:flex;flex-direction:column;min-height:0;overflow:hidden}}
.ctitle{{font-size:9px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
  color:var(--sub);padding:8px 14px;border-bottom:1px solid var(--border);flex-shrink:0}}
#chartAngle{{flex:1;min-height:0}}
</style>
</head>
<body>
<header>
  <div class="h-left">
    <div class="logo"></div>
    <span class="h-title">Volleyball Spike · Motion Analysis</span>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="badge">{sn}</span>
    <span class="imu {imu_badge_cls}">{imu_badge_text}</span>
  </div>
</header>
<div class="main">
  <div class="left">
    <div class="panel-lbl">📹 Raw Video</div>
    <div class="vid-wrap">
      {video_tag}
      <div id="vidOverlay">
        <div class="spinner"></div>
        <span id="overlayMsg">Loading video...</span>
      </div>
    </div>
    <div class="ctrl-area">
      <div class="finfo">
        <span>Frame <span class="fval" id="dF">—</span></span>
        <span class="aval" id="dA">—°</span>
        <span class="tval" id="dT">0.000s</span>
      </div>
      <input type="range" id="slider" min="0" max="0" value="0" step="1">
      <div class="ctrls">
        <button class="btn primary" id="btnPlay" onclick="togglePlay()">▶ Play</button>
        <button class="btn" onclick="step(-10)">⏮</button>
        <button class="btn" onclick="step(-1)">‹</button>
        <button class="btn" onclick="step(1)">›</button>
        <button class="btn" onclick="step(10)">⏭</button>
        <select class="spd" id="spdSel" onchange="updateSpd()">
          <option value="4">0.25×</option>
          <option value="2">0.50×</option>
          <option value="1" selected>1.00×</option>
          <option value="0.5">2×</option>
        </select>
      </div>
      <div class="hints">
        <span class="hint"><kbd>Space</kbd> Play/Pause</span>
        <span class="hint"><kbd>←</kbd><kbd>→</kbd> ±1 frame</span>
        <span class="hint"><kbd>Shift</kbd>+<kbd>←/→</kbd> ±10</span>
      </div>
    </div>
  </div>
  <div class="right">
    <div class="graphs-row">
      <div class="gcell">
        <div class="gtitle" style="color:var(--accent)">◈ 3D View</div>
        <div class="plotly-cell" id="g3d"></div>
      </div>
      <div class="gcell">
        <div class="gtitle" style="color:var(--accent2)">◈ Front View (Z–Y)</div>
        <div class="plotly-cell" id="gFront"></div>
      </div>
      <div class="gcell">
        <div class="gtitle" style="color:var(--green)">◈ Birdeye View (X–Z)</div>
        <div class="plotly-cell" id="gBird"></div>
      </div>
    </div>
    <div class="chart-area">
      <div class="ctitle">{chart_title}</div>
      <div id="chartAngle"></div>
    </div>
  </div>
</div>
<script>
const P = {payload};
const frames = P.frames, FD = P.frame_dict;
const timeSeries = P.time_series, angleSeries = P.angle_series, accSeries = P.acc_series;
const hasIMU = P.has_imu;
const FPS = P.src_fps || 60.0;
const minF = frames[0], maxF = frames[frames.length - 1];

// Grab elements — null-safe because Qt strips the left panel in the viewer
const sl  = document.getElementById('slider');
const btn = document.getElementById('btnPlay');
const vid = document.getElementById('mainVideo');
// The slider only exists when opening the plain HTML, not in the PyQt viewer
if (sl) {{
  sl.min = minF; sl.max = maxF; sl.value = minF;
  sl.addEventListener('input', () => seek(+sl.value));
}}

const DARK = {{
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#6c7d93', family:'JetBrains Mono,monospace', size:9}},
  margin:{{l:42, r:14, t:12, b:32}}, showlegend:false,
}};
const AX = {{ gridcolor:'#262f42', zerolinecolor:'#3e4b66', tickfont:{{size:8,color:'#6c7d93'}} }};
const R  = [-0.65, 0.65];

const allX = frames.map(f => (FD[f]||{{}}).wr?.[0] ?? null);
const allY = frames.map(f => (FD[f]||{{}}).wr?.[1] ?? null);
const allZ = frames.map(f => (FD[f]||{{}}).wr?.[2] ?? null);

Plotly.newPlot('g3d', [
  {{ type:'scatter3d', mode:'lines', x:allX, y:allZ, z:allY,
     line:{{color:'#00e5ff',width:2}}, opacity:.12, hoverinfo:'skip' }},
  {{ type:'scatter3d', mode:'lines+markers', x:[0,0,0], y:[0,0,0], z:[0,0,0],
     line:{{color:'#e8edf5',width:5}},
     marker:{{size:[8,5,8], color:['#ff4d4d','#00e5ff','#ffcc00']}}, hoverinfo:'skip' }},
  {{ type:'scatter3d', mode:'lines', x:[], y:[], z:[],
     line:{{color:'#00e5ff',width:3.5}}, opacity:.9, hoverinfo:'skip' }},
], {{
  ...DARK, margin:{{l:0,r:0,t:0,b:0}},
  scene:{{
    xaxis:{{...AX, title:{{text:'X',font:{{size:8}}}}, range:R}},
    yaxis:{{...AX, title:{{text:'Depth Z',font:{{size:8}}}}, range:R}},
    zaxis:{{...AX, title:{{text:'Height Y',font:{{size:8}}}}, range:R}},
    camera:{{eye:{{x:1.3,y:-1.3,z:0.8}}}},
    aspectmode:'cube', bgcolor:'rgba(0,0,0,0)',
  }},
}}, {{responsive:true, displayModeBar:false}});

Plotly.newPlot('gFront', [
  {{ type:'scatter', mode:'lines', x:allZ, y:allY,
     line:{{color:'#ff6b35',width:1}}, opacity:.12, hoverinfo:'skip' }},
  {{ type:'scatter', mode:'lines+markers', x:[0,0,0], y:[0,0,0],
     line:{{color:'#ff6b35',width:3}},
     marker:{{size:[8,5,8], color:['#ff4d4d','#ff6b35','#ffcc00']}}, hoverinfo:'skip' }},
  {{ type:'scatter', mode:'lines', x:[], y:[],
     line:{{color:'#ff6b35',width:2.5}}, opacity:.9, hoverinfo:'skip' }},
], {{
  ...DARK,
  xaxis:{{...AX, title:{{text:'Depth Z',font:{{size:8}}}}, range:R}},
  yaxis:{{...AX, title:{{text:'Height Y',font:{{size:8}}}}, range:R, scaleanchor:'x'}},
}}, {{responsive:true, displayModeBar:false}});

Plotly.newPlot('gBird', [
  {{ type:'scatter', mode:'lines', x:allX, y:allZ,
     line:{{color:'#39d98a',width:1}}, opacity:.12, hoverinfo:'skip' }},
  {{ type:'scatter', mode:'lines+markers', x:[0,0,0], y:[0,0,0],
     line:{{color:'#39d98a',width:3}},
     marker:{{size:[8,5,8], color:['#ff4d4d','#39d98a','#ffcc00']}}, hoverinfo:'skip' }},
  {{ type:'scatter', mode:'lines', x:[], y:[],
     line:{{color:'#39d98a',width:2.5}}, opacity:.9, hoverinfo:'skip' }},
], {{
  ...DARK,
  xaxis:{{...AX, title:{{text:'Lateral X',font:{{size:8}}}}, range:R}},
  yaxis:{{...AX, title:{{text:'Depth Z',font:{{size:8}}}}, range:R, scaleanchor:'x'}},
}}, {{responsive:true, displayModeBar:false}});

const chartTraces = [
  {{ type:'scatter', mode:'lines', x:timeSeries, y:angleSeries,
     line:{{color:'#39d98a',width:2}}, name:'Elbow angle (°)' }},
];
{acc_trace}
chartTraces.push({{
  type:'scatter', mode:'lines',
  x:[timeSeries[0]||0, timeSeries[0]||0], y:[0, 180],
  line:{{color:'#00e5ff',width:1.5,dash:'dash'}},
  name:'Current frame', showlegend:false,
}});
const cursorIdx = chartTraces.length - 1;

Plotly.newPlot('chartAngle', chartTraces, {{
  ...DARK, margin:{{l:46,r:{52 if has_imu else 16},t:8,b:34}},
  xaxis:{{...AX, title:{{text:'Time (s)',font:{{size:9}}}}}},
  yaxis:{{...AX, title:{{text:'Angle (°)',font:{{size:9,color:'#39d98a'}}}},
    range:[0,180], tickfont:{{color:'#39d98a',size:9}}}},
  {y2_layout}
  showlegend:true,
  legend:{{x:.01,y:.99,bgcolor:'rgba(19,23,32,.88)',font:{{size:9,color:'#e8edf5'}},
    bordercolor:'#262f42',borderwidth:1}},
}}, {{responsive:true, displayModeBar:false}});

// VIDEO SETUP — null-safe (vid doesn't exist in the PyQt viewer)
let videoReady = false;
if (vid) {{
  const overlay    = document.getElementById('vidOverlay');
  const overlayMsg = document.getElementById('overlayMsg');
  function markReady() {{
    videoReady = true;
    if (overlay) overlay.classList.add('hidden');
  }}
  vid.addEventListener('loadedmetadata', () => {{
    if (overlayMsg) overlayMsg.textContent = 'Buffering...';
  }});
  vid.addEventListener('canplay', markReady);
  vid.addEventListener('canplaythrough', markReady);
  vid.addEventListener('error', () => {{
    if (overlay) overlay.classList.remove('hidden');
    if (overlayMsg) overlayMsg.textContent = '⚠️ Could not load the video (unsupported codec?)';
    videoReady = false;
  }});
  if (vid.readyState >= 3) markReady();
  vid.addEventListener('timeupdate', () => {{
    if (!ps.on || !videoReady) return;
    const cf = Math.max(minF, Math.min(maxF, Math.round(vid.currentTime * FPS) + minF));
    if (Math.abs(cf - curF) >= 1) seek(cf, true);
  }});
  vid.addEventListener('ended', () => pause());
}}

let curF = minF;
function seek(f, fromVid) {{
  f = Math.max(minF, Math.min(maxF, f)); curF = f;
  if (sl) sl.value = f;
  const fd = FD[f] || FD[String(f)];
  if (!fd) return;
  const dF = document.getElementById('dF'); if (dF) dF.textContent = f;
  const dA = document.getElementById('dA'); if (dA) dA.textContent = fd.angle.toFixed(1) + '°';
  const dT = document.getElementById('dT'); if (dT) dT.textContent = fd.time_offset.toFixed(3) + 's';
  if (vid && !fromVid && videoReady) {{
    const tgt = Math.max(0, (f - minF) / FPS);
    if (Math.abs(vid.currentTime - tgt) > 1 / FPS) vid.currentTime = tgt;
  }}
  const tx = fd.trail.map(p=>p[0]), ty = fd.trail.map(p=>p[1]), tz = fd.trail.map(p=>p[2]);
  Plotly.restyle('g3d', {{x:[[0,fd.el[0],fd.wr[0]]],y:[[0,fd.el[2],fd.wr[2]]],z:[[0,fd.el[1],fd.wr[1]]]}}, [1]);
  Plotly.restyle('g3d', {{x:[tx],y:[tz],z:[ty]}}, [2]);
  Plotly.restyle('gFront', {{x:[[0,fd.el[2],fd.wr[2]]],y:[[0,fd.el[1],fd.wr[1]]]}}, [1]);
  Plotly.restyle('gFront', {{x:[tz],y:[ty]}}, [2]);
  Plotly.restyle('gBird', {{x:[[0,fd.el[0],fd.wr[0]]],y:[[0,fd.el[2],fd.wr[2]]]}}, [1]);
  Plotly.restyle('gBird', {{x:[tx],y:[tz]}}, [2]);
  Plotly.restyle('chartAngle', {{x:[[fd.time_offset,fd.time_offset]]}}, [cursorIdx]);
}}

const ps = {{on:false, spd:1.0, rafId:null, lastTs:null}};
function updateSpd() {{
  const sel = document.getElementById('spdSel');
  if (sel) {{
    ps.spd = 1.0 / parseFloat(sel.value);
    if (vid) vid.playbackRate = ps.spd;
  }}
}}
function togglePlay() {{ ps.on ? pause() : play(); }}
function play() {{
  ps.on = true;
  if (btn) btn.textContent = '⏸ Pause';
  if (vid && videoReady) {{
    vid.playbackRate = ps.spd;
    vid.play().catch(err => {{ console.warn('play() blocked:', err); pause(); }});
  }} else if (!vid) {{ rafPlay(); }}
}}
function pause() {{
  ps.on = false;
  if (btn) btn.textContent = '▶ Play';
  if (vid) vid.pause();
  if (ps.rafId) {{ cancelAnimationFrame(ps.rafId); ps.rafId = null; }}
}}
function rafPlay() {{
  ps.rafId = requestAnimationFrame(ts => {{
    if (!ps.on) return;
    if (!ps.lastTs) ps.lastTs = ts;
    const nf = curF + Math.max(1, Math.round((ts - ps.lastTs) / 1000 * FPS * ps.spd));
    ps.lastTs = ts;
    if (nf > maxF) {{ pause(); seek(minF); return; }}
    seek(nf); rafPlay();
  }});
}}
function step(d) {{ pause(); seek(curF + d); }}
document.addEventListener('keydown', e => {{
  if (['INPUT','SELECT'].includes(e.target.tagName)) return;
  if (e.code === 'Space')           {{ e.preventDefault(); togglePlay(); }}
  else if (e.code === 'ArrowRight') {{ e.preventDefault(); step(e.shiftKey ? 10 : 1); }}
  else if (e.code === 'ArrowLeft')  {{ e.preventDefault(); step(e.shiftKey ? -10 : -1); }}
}});
seek(minF);
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\n✅ motion_report.html saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────
# STEP D: PyQt6 VIEWER
# Layout: horizontal QSplitter — left: video pane (QMediaPlayer+QVideoWidget)
#         right: QWebEngineView (graphs + controls)
# Video is decoded natively by Qt → independent of the Chromium codec
# ──────────────────────────────────────────────────────────────────────
def launch_pyqt_viewer(session_folder: Path, report_path: Path):
    try:
        from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                                     QVBoxLayout, QHBoxLayout, QSplitter,
                                     QPushButton, QSlider, QLabel, QComboBox,
                                     QProgressBar, QSizePolicy)
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PyQt6.QtMultimediaWidgets import QVideoWidget
        from PyQt6.QtCore import QUrl, Qt, QTimer, pyqtSignal, QObject
        from PyQt6.QtGui import QColor, QPalette
    except ModuleNotFoundError as exc:
        print(f"\n⚠️  Missing module: {exc}")
        print("   Run: uv add pyqt6-webengine")
        print(f"\n   Report: {report_path}")
        return
    import sys as _sys

    # ── Choose the video file: prefer annotated, fall back to raw ─────
    annotated = session_folder / "annotated_video.mp4"
    raw_src   = session_folder / "raw_video.mp4"
    video_file = annotated if annotated.exists() else raw_src
    has_annotated = annotated.exists()
    showing_annotated = has_annotated  # track the toggle state

    # ── FPS SYNC: distinguish 2 DIFFERENT quantities, don't mix them ──
    #  • declared_fps: the FPS in the video file header = the file's own timebase.
    #    QMediaPlayer.position() is MEDIA TIME (unchanged by playbackRate), so frame
    #    f always sits at position f/declared_fps → frame↔position conversion MUST use
    #    this value. Using real_fps here would give a position beyond the file length.
    #  • real_fps: the TRUE capture rate, read from the timestamps in camera_data.csv.
    # The video is written at a fixed 60fps but the real camera reaches only ~25-60fps →
    # the file plays faster than real life. Compensate with playbackRate = real/declared
    # (media time stays the same, it just slows by exactly the shortfall) → the video
    # matches the graph's speed.
    def _probe_declared_fps(path: Path) -> float:
        _c = cv2.VideoCapture(str(path))
        _d = _c.get(cv2.CAP_PROP_FPS)
        _c.release()
        return float(_d) if _d and 0 < _d <= 240 else 60.0

    def _probe_real_fps() -> float:
        _p = session_folder / "camera_data.csv"
        if not _p.exists():
            return 0.0
        try:
            _d = pd.read_csv(_p)
        except Exception:
            return 0.0
        if len(_d) < 2:
            return 0.0
        _span = float(_d["timestamp"].iloc[-1] - _d["timestamp"].iloc[0])
        # Use the FRAME-INDEX SPAN, not the row count: frames where MediaPipe fails to
        # detect a pose have no CSV row, so counting rows would give an FPS lower than
        # reality and playbackRate would still be off.
        _fspan = float(_d["frame"].iloc[-1] - _d["frame"].iloc[0])
        if _span <= 0 or _fspan <= 0:
            return 0.0
        _r = _fspan / _span
        return _r if 0 < _r <= 240 else 0.0

    _fps      = _probe_declared_fps(video_file)   # the video file's timebase
    _real_fps = _probe_real_fps() or _fps         # the real-world capture rate
    _rate     = _real_fps / _fps if _fps > 0 else 1.0
    print(f"  → Video declared {_fps:.1f} fps · real capture {_real_fps:.1f} fps "
          f"· playbackRate {_rate:.3f}")

    # ── Patch the HTML: drop the video tag & vidOverlay, keep only graphs+controls ──
    html_text = report_path.read_text(encoding="utf-8")
    # Remove the entire left panel from the HTML — Qt provides the video pane itself
    # Change the .main grid to a single column, showing only the right panel
    import re
    # Remove the whole left panel from the HTML with a precise string split
    # Find the start and end of <div class="left">...</div>
    LEFT_START  = '<div class="left">'
    RIGHT_START = '<div class="right">'
    if LEFT_START in html_text and RIGHT_START in html_text:
        idx_left  = html_text.index(LEFT_START)
        idx_right = html_text.index(RIGHT_START)
        # Remove everything from the start of left up to just before right
        html_text = html_text[:idx_left] + html_text[idx_right:]
    # Switch the grid to a single column since the left panel is gone
    html_text = html_text.replace(
        "grid-template-columns:400px 1fr",
        "grid-template-columns:1fr"
    )
    graphs_html_path = session_folder / "graphs_only.html"
    graphs_html_path.write_text(html_text, encoding="utf-8")

    # ── Qt App ─────────────────────────────────────────────────────────
    app = QApplication.instance() or QApplication(_sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(13, 15, 20))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(232, 237, 245))
    pal.setColor(QPalette.ColorRole.Base,            QColor(27, 32, 46))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(19, 23, 32))
    pal.setColor(QPalette.ColorRole.Button,          QColor(27, 32, 46))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(232, 237, 245))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(0, 229, 255))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(pal)

    window = QMainWindow()
    window.setWindowTitle(f"Motion Analysis · {session_folder.name}")
    window.resize(1600, 920)

    # ── SPLITTER: video left | graphs right ─────────────────────────────
    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setHandleWidth(2)
    splitter.setStyleSheet("QSplitter::handle { background: #262f42; }")

    # ════════════════════════════════════════════════════════════════════
    # LEFT: Video pane (Qt native)
    # ════════════════════════════════════════════════════════════════════
    left_widget = QWidget()
    left_widget.setStyleSheet("background:#0d0f14;")
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(0)

    # Panel label
    panel_lbl = QLabel("📹  Video")
    panel_lbl.setStyleSheet(
        "font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;"
        "color:#6c7d93;padding:8px 14px;border-bottom:1px solid #262f42;"
        "background:#131720;"
    )
    left_layout.addWidget(panel_lbl)

    # Video widget
    video_widget = QVideoWidget()
    video_widget.setStyleSheet("background:#000;")
    video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    left_layout.addWidget(video_widget, stretch=1)

    # Loading bar (visible while buffering)
    load_bar = QProgressBar()
    load_bar.setRange(0, 0)   # indeterminate
    load_bar.setTextVisible(False)
    load_bar.setFixedHeight(3)
    load_bar.setStyleSheet(
        "QProgressBar{background:#1b202e;border:none;}"
        "QProgressBar::chunk{background:#00e5ff;}"
    )
    left_layout.addWidget(load_bar)

    # ── Media player ───────────────────────────────────────────────────
    player   = QMediaPlayer()
    audio    = QAudioOutput()
    audio.setVolume(0.0)   # mute (no audio needed)
    player.setAudioOutput(audio)
    player.setVideoOutput(video_widget)

    def _load_video(path: Path):
        load_bar.setVisible(True)
        load_bar.setRange(0, 0)
        player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        player.pause()

    _load_video(video_file)

    def _on_media_status(status):
        ready_states = {
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        }
        if status in ready_states:
            load_bar.setVisible(False)
        elif status == QMediaPlayer.MediaStatus.LoadingMedia:
            load_bar.setRange(0, 0)
            load_bar.setVisible(True)

    player.mediaStatusChanged.connect(_on_media_status)

    # ── Frame info bar ──────────────────────────────────────────────────
    info_bar = QLabel("Frame —  |  —°  |  0.000s")
    info_bar.setStyleSheet(
        "font-family:'JetBrains Mono',monospace;font-size:11px;color:#6c7d93;"
        "padding:6px 14px;background:#1b202e;border-top:1px solid #262f42;"
    )
    left_layout.addWidget(info_bar)

    # ── Timeline slider ─────────────────────────────────────────────────
    tl_slider = QSlider(Qt.Orientation.Horizontal)
    tl_slider.setStyleSheet(
        "QSlider::groove:horizontal{height:3px;background:#262f42;border-radius:2px;}"
        "QSlider::handle:horizontal{width:14px;height:14px;margin:-6px 0;"
        "border-radius:7px;background:#00e5ff;}"
        "QSlider::sub-page:horizontal{background:#00e5ff;border-radius:2px;}"
    )
    left_layout.addWidget(tl_slider)

    # ── Control buttons ─────────────────────────────────────────────────
    ctrl_style = (
        "QPushButton{background:#1b202e;border:1px solid #262f42;color:#e8edf5;"
        "border-radius:5px;font-size:12px;padding:5px 10px;}"
        "QPushButton:hover{border-color:#00e5ff;background:#262f42;}"
        "QPushButton#btnPlay{background:#00e5ff;border-color:#00e5ff;color:#000;font-weight:700;padding:5px 16px;}"
        "QPushButton#btnPlay:hover{background:#33ecff;}"
        "QPushButton#btnAnnot[checked=true]{border-color:#39d98a;color:#39d98a;}"
    )

    ctrl_row = QWidget()
    ctrl_row.setStyleSheet(ctrl_style + "QWidget{background:#1b202e;}")
    ctrl_layout = QHBoxLayout(ctrl_row)
    ctrl_layout.setContentsMargins(10, 8, 10, 8)
    ctrl_layout.setSpacing(6)

    btn_play  = QPushButton("▶  Play");  btn_play.setObjectName("btnPlay")
    btn_m10   = QPushButton("⏮")
    btn_m1    = QPushButton("‹")
    btn_p1    = QPushButton("›")
    btn_p10   = QPushButton("⏭")

    spd_box = QComboBox()
    spd_box.setStyleSheet(
        "QComboBox{background:#131720;border:1px solid #262f42;color:#e8edf5;"
        "border-radius:5px;font-size:11px;padding:3px 6px;font-family:monospace;}"
        "QComboBox::drop-down{border:none;}"
    )
    for label in ["0.25×", "0.50×", "1.00×", "2×"]:
        spd_box.addItem(label)
    spd_box.setCurrentIndex(2)
    spd_multipliers = [0.25, 0.5, 1.0, 2.0]

    for w in [btn_play, btn_m10, btn_m1, btn_p1, btn_p10]:
        ctrl_layout.addWidget(w)
    ctrl_layout.addStretch()

    # Toggle annotated button (only shown if an annotated video exists)
    btn_annot = None
    if has_annotated:
        btn_annot = QPushButton("📹 Annotated")
        btn_annot.setObjectName("btnAnnot")
        btn_annot.setCheckable(True)
        btn_annot.setChecked(True)
        btn_annot.setStyleSheet(
            "QPushButton{background:#1b202e;border:1px solid #39d98a;color:#39d98a;"
            "border-radius:5px;font-size:12px;padding:5px 10px;}"
            "QPushButton:checked{background:rgba(57,217,138,.12);}"
            "QPushButton:!checked{border-color:#262f42;color:#6c7d93;}"
        )
        ctrl_layout.addWidget(btn_annot)

    ctrl_layout.addWidget(spd_box)
    left_layout.addWidget(ctrl_row)

    # Hint bar
    hint_bar = QLabel("Space: Play/Pause   ←→: ±1 frame   Shift+←→: ±10")
    hint_bar.setStyleSheet(
        "font-size:9px;color:#4a5568;padding:4px 14px 6px;background:#1b202e;"
        "border-top:1px solid #262f42;"
    )
    left_layout.addWidget(hint_bar)

    # ════════════════════════════════════════════════════════════════════
    # RIGHT: QWebEngineView — graphs only HTML
    # ════════════════════════════════════════════════════════════════════
    view = QWebEngineView()
    settings = view.settings()
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
    view.setUrl(QUrl.fromLocalFile(str(graphs_html_path.resolve())))

    splitter.addWidget(left_widget)
    splitter.addWidget(view)
    splitter.setSizes([400, 1200])

    window.setCentralWidget(splitter)

    # ════════════════════════════════════════════════════════════════════
    # LOGIC: frame / playback sync
    # ════════════════════════════════════════════════════════════════════
    import json as _json
    _html_src = report_path.read_text(encoding="utf-8")
    _m = re.search(r"const P = (\{.*?\});", _html_src, re.DOTALL)
    _payload = _json.loads(_m.group(1)) if _m else {}
    _frames   = _payload.get("frames", [])
    _fd       = _payload.get("frame_dict", {})
    # _fps / _real_fps / _rate were measured above DIRECTLY from the video being played.
    # No longer take src_fps from the payload: the payload reads raw_video.mp4's FPS, while
    # the viewer plays annotated_video.mp4 — the two files may declare different FPS.

    if _frames:
        tl_slider.setMinimum(_frames[0])
        tl_slider.setMaximum(_frames[-1])
        tl_slider.setValue(_frames[0])

    _playing      = False
    _cur_frame    = _frames[0] if _frames else 0
    _skip_video_cb = False   # guard against seek loop

    def _update_info(f):
        fd = _fd.get(str(f)) or _fd.get(f)
        if not fd:
            return
        ang = fd['angle']
        toff = fd['time_offset']
        info_bar.setText(
            f"Frame {f}  |  {ang:.1f}°  |  {toff:.3f}s"
        )
        # Sync cursor in graph via JS
        view.page().runJavaScript(f"seek({f});")

    def _seek_frame(f, from_video=False):
        nonlocal _cur_frame, _skip_video_cb
        f = max(tl_slider.minimum(), min(tl_slider.maximum(), f))
        _cur_frame = f
        tl_slider.setValue(f)
        _update_info(f)
        if not from_video:
            _skip_video_cb = True
            # the frame in camera_data.csv is the ABSOLUTE frame index in the video
            # file (cameraclient writes every frame to the video, but only writes a CSV
            # row when a pose is detected) → frame f sits exactly at media time f/_fps.
            # Do NOT subtract tl_slider.minimum(): the CSV usually starts at frame >0
            # (the start where no pose was detected yet); subtracting would map the first
            # frame to 0s and shift the video against the graph by exactly minimum/_fps seconds.
            target_ms = int(f / _fps * 1000)
            player.setPosition(target_ms)

    def _on_slider_moved(val):
        _seek_frame(val)

    tl_slider.sliderMoved.connect(_on_slider_moved)
    tl_slider.valueChanged.connect(lambda v: _seek_frame(v) if not _playing else None)

    # Sync frame from video position during playback
    def _on_position(ms):
        nonlocal _skip_video_cb, _playing
        if _skip_video_cb:
            _skip_video_cb = False
            return
        # Inverse of _seek_frame — also does NOT add tl_slider.minimum()
        f = int(ms / 1000 * _fps)
        # Stop as soon as the graph runs out of data: the video usually has leftover
        # frames at the end (the no-pose-detected part) — letting it keep going would be
        # "the video still playing after the chart has ended".
        if f >= tl_slider.maximum():
            _seek_frame(tl_slider.maximum(), from_video=True)
            if _playing:
                player.pause()
                btn_play.setText("▶  Play")
                _playing = False
            return
        f = max(tl_slider.minimum(), min(tl_slider.maximum(), f))
        if abs(f - _cur_frame) >= 1:
            _seek_frame(f, from_video=True)

    player.positionChanged.connect(_on_position)

    def _toggle_play():
        nonlocal _playing
        if _playing:
            player.pause()
            btn_play.setText("▶  Play")
            _playing = False
        else:
            # If sitting at the end of the data, jump back to the start before replaying
            if _cur_frame >= tl_slider.maximum():
                _seek_frame(tl_slider.minimum())
            spd = spd_multipliers[spd_box.currentIndex()]
            player.setPlaybackRate(spd * _rate)   # compensate declared FPS > real capture FPS
            player.play()
            btn_play.setText("⏸  Pause")
            _playing = True

    def _on_media_end(status):
        nonlocal _playing
        if status == QMediaPlayer.MediaStatus.EndOfMedia and _playing:
            player.pause()
            btn_play.setText("▶  Play")
            _playing = False
            _seek_frame(tl_slider.minimum())

    player.mediaStatusChanged.connect(_on_media_end)

    btn_play.clicked.connect(_toggle_play)
    btn_m10.clicked.connect(lambda: (_toggle_play() if _playing else None, _seek_frame(_cur_frame - 10)))
    btn_m1.clicked.connect(lambda: (_toggle_play() if _playing else None, _seek_frame(_cur_frame - 1)))
    btn_p1.clicked.connect(lambda: (_toggle_play() if _playing else None, _seek_frame(_cur_frame + 1)))
    btn_p10.clicked.connect(lambda: (_toggle_play() if _playing else None, _seek_frame(_cur_frame + 10)))

    def _update_speed(idx):
        spd = spd_multipliers[idx]
        if _playing:
            player.setPlaybackRate(spd * _rate)

    spd_box.currentIndexChanged.connect(_update_speed)

    # Toggle annotated / raw
    if btn_annot:
        def _toggle_annotated(checked):
            nonlocal _fps, _rate
            was_playing = _playing
            if was_playing:
                _toggle_play()
            # Save by FRAME, not by ms: raw_video.mp4 (written at a fixed 60fps) and
            # annotated_video.mp4 (written at the real FPS) declare different FPS, so the
            # same ms mark would point to 2 different frames across the 2 files.
            saved_frame = _cur_frame
            target = annotated if checked else raw_src
            _fps  = _probe_declared_fps(target)   # timebase changes with the file being played
            _rate = _real_fps / _fps if _fps > 0 else 1.0
            player.setSource(QUrl.fromLocalFile(str(target.resolve())))
            player.pause()
            # Restore position after load
            def _restore_pos(status):
                if status == QMediaPlayer.MediaStatus.LoadedMedia:
                    _seek_frame(saved_frame)
                    player.mediaStatusChanged.disconnect(_restore_pos)
                    if was_playing:
                        QTimer.singleShot(100, _toggle_play)
            player.mediaStatusChanged.connect(_restore_pos)
            btn_annot.setText("📹 Annotated" if checked else "🎬 Raw")
        btn_annot.toggled.connect(_toggle_annotated)

    # Keyboard shortcuts
    def _keypress(event):
        key = event.key()
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if key == Qt.Key.Key_Space:
            _toggle_play()
        elif key == Qt.Key.Key_Right:
            if _playing: _toggle_play()
            _seek_frame(_cur_frame + (10 if shift else 1))
        elif key == Qt.Key.Key_Left:
            if _playing: _toggle_play()
            _seek_frame(_cur_frame - (10 if shift else 1))
        event.accept()

    window.keyPressEvent = _keypress

    window.show()
    _seek_frame(_frames[0] if _frames else 0)

    print(f"🖥️  Viewer opened  (annotated video: {'yes' if has_annotated else 'not yet'})")
    print("   Close the window to exit.")
    app.exec()

    # Clean up temporary files
    if graphs_html_path.exists():
        graphs_html_path.unlink()


def _reencode_to_h264(src: Path, dst: Path):
    """Re-encode the video to H.264 (mp4v fourcc) so the browser in QWebEngineView can decode it."""
    print(f"  → Re-encoding video to H.264 for viewer compatibility...")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  ⚠️  Could not open the source video to re-encode.")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))

    fidx = 0
    last_pct = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        fidx += 1
        pct = int(fidx / total * 100) if total > 0 else 0
        if pct != last_pct and pct % 10 == 0:
            last_pct = pct
            print(f"\r  Re-encode: {pct}%", end="", flush=True)

    print()
    cap.release()
    out.release()
    print(f"  ✅ Re-encode done: {dst.name}")


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
def _write_viewer_bat(session_folder: Path):
    """Create Open_Viewer.bat in the session folder to open the viewer by double-clicking."""
    session_folder = Path(session_folder)
    bat_content = (
        "@echo off\n"
        "cd /d \"%~dp0..\\..\"\n"
        f'uv run python dataprocessing.py "recording\\{session_folder.name}"\n'
        "pause\n"
    )
    bat_path = session_folder / "Open_Viewer.bat"
    bat_path.write_text(bat_content, encoding="utf-8")
    print(f"  → Shortcut: {bat_path.name} (double-click to open the viewer)")


def run(session_folder, open_browser=False):
    """Called from mainpy.py after recording stops."""
    session_folder = Path(session_folder)
    print(f"\n{'='*60}")
    print(f"  EXPORT REPORT: {session_folder.name}")
    print(f"{'='*60}")
    data = process_session(session_folder)
    if data is None:
        return
    output_path = session_folder / "motion_report.html"
    generate_html(data, output_path)
    _write_viewer_bat(session_folder)
    if open_browser:
        launch_pyqt_viewer(session_folder, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze spike motion & open the PyQt6 viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python dataprocessing.py recording/Hoang_20260612_110759
  uv run python dataprocessing.py spike.mp4
  uv run python dataprocessing.py spike.mp4 --side LEFT --complexity 2
        """
    )
    parser.add_argument("input",
        help="Session folder (with camera_data.csv) OR a video file (.mp4/.avi/.mov)")
    parser.add_argument("--side", default="RIGHT", choices=["RIGHT","LEFT"],
        help="Which arm to analyze when the input is a video (default: RIGHT)")
    parser.add_argument("--complexity", type=int, default=1, choices=[0,1,2],
        help="MediaPipe complexity when the input is a video (default: 1)")
    parser.add_argument("--out", default=None,
        help="Custom session output folder (only used when the input is a video)")
    parser.add_argument("--no-viewer", action="store_true",
        help="Only export the HTML, do not open the viewer window")

    args = parser.parse_args()
    path = Path(args.input)

    # Input is a video file → extract the CSV first
    if path.is_file() and path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        from datetime import datetime
        if args.out:
            session_dir = Path(args.out)
        else:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = path.stem.replace(" ", "_")
            session_dir = Path("recording") / f"{stem}_{ts}"
        session_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  PROCESS VIDEO: {path.name}")
        print(f"  Session dir: {session_dir}")
        print(f"{'='*60}")

        ok = extract_pose_from_video(
            path, session_dir / "camera_data.csv",
            side=args.side, model_complexity=args.complexity,
        )
        if not ok:
            sys.exit(1)

        dst = session_dir / "raw_video.mp4"
        if path.resolve() != dst.resolve():
            shutil.copy2(path, dst)
            print(f"  → Video copied into the session folder.")

        path = session_dir

    elif not path.is_dir():
        print(f"[ERROR] Not found: {path}")
        sys.exit(1)

    data = process_session(path)
    if data is None:
        sys.exit(1)

    output_path = path / "motion_report.html"
    generate_html(data, output_path)
    _write_viewer_bat(path)

    if not args.no_viewer:
        launch_pyqt_viewer(path, output_path)


if __name__ == "__main__":
    main()