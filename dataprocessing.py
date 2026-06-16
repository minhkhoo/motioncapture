"""
dataprocessing.py – Motion Capture & Automated Video Analysis Report Generator
=============================================================================
HƯỚNG DẪN SỬ DỤNG:
1. Chạy phân tích video cũ/mới: python dataprocessing.py đường_dẫn_video.mp4
2. Chạy từ folder session có sẵn csv: python dataprocessing.py đường_dẫn_folder
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import base64
import time

# =====================================================================
# CẤU HÌNH BỘ LỌC ĐỘNG HỌC (EMA Filter)
# =====================================================================
TRAIL_LENGTH = 45
SMOOTH_SH = 0.85
SMOOTH_EL = 0.50
SMOOTH_WR = 0.30

def calculate_elbow_angle(shoulder, elbow, wrist):
    v1 = np.array(elbow) - np.array(shoulder)
    v2 = np.array(wrist)  - np.array(elbow)
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return 180 - np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# =====================================================================
# TÍNH NĂNG MỚI: XỬ LÝ VIDEO OFFLINE / UPLOAD BẰNG MEDIAPIPE
# =====================================================================
def extract_data_from_video(video_path: Path) -> Path:
    print(f"\n🚀 Khởi chạy hệ thống phân tích thị giác máy tính cho: {video_path.name}")
    
    # Tạo thư mục session tự động dựa trên tên video
    session_dir = video_path.parent / f"session_{video_path.stem}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 60.0
    
    raw_data = []
    frame_idx = 0
    start_time = time.time()

    print(" → Đang quét khung xương người đập bóng (Vui lòng đợi)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = frame_idx / fps
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Mặc định lấy tay phải (Các khớp: 12, 14, 16). Thay đổi nếu người chơi thuận tay trái.
            sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            raw_data.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "shoulder_x": sh.x, "shoulder_y": sh.y, "shoulder_z": sh.z,
                "elbow_x": el.x, "elbow_y": el.y, "elbow_z": el.z,
                "wrist_x": wr.x, "wrist_y": wr.y, "wrist_z": wr.z
            })
        frame_idx += 1

    cap.release()
    pose.close()
    
    # Lưu file CSV tương thích với bộ xử lý dữ liệu cũ
    df = pd.DataFrame(raw_data)
    df.to_csv(session_dir / "camera_data.csv", index=False)
    
    # Sao chép/Di chuyển video vào đúng cấu trúc thư mục
    target_video_path = session_dir / "raw_video.mp4"
    if video_path != target_video_path:
        import shutil
        shutil.copy(video_path, target_video_path)
        
    print(f" ✅ Trích xuất hoàn tất! Dữ liệu lưu tại folder: {session_dir}")
    return session_dir

# =====================================================================
# BƯỚC 1: ĐỌC VÀ XỬ LÝ DATA (EMA + DATA FUSION)
# =====================================================================
def process_session(session_folder):
    session_folder = Path(session_folder)
    csv_path   = session_folder / "camera_data.csv"
    imu_path   = session_folder / "imu_data.csv"
    video_path = session_folder / "raw_video.mp4"

    if not csv_path.exists():
        print(f"[LỖI] Không tìm thấy camera_data.csv tại: {csv_path}")
        return None

    print("  → Đọc dữ liệu chuyển động gốc...")
    df_cam = pd.read_csv(csv_path)
    cam_ts = df_cam["timestamp"].values
    frames = df_cam["frame"].values.astype(int)

    accel_mag_synced = np.zeros(len(df_cam))
    has_imu = False
    if imu_path.exists():
        print("  → Tích hợp dữ liệu cảm biến phần cứng IMU...")
        df_imu  = pd.read_csv(imu_path)
        has_imu = True
        imu_mag   = np.sqrt(df_imu["ax"]**2 + df_imu["ay"]**2 + df_imu["az"]**2)
        imu_clean = np.abs(imu_mag - np.mean(imu_mag.iloc[:15]))
        if "timestamp" in df_imu.columns:
            accel_mag_synced = np.interp(cam_ts, df_imu["timestamp"].values, imu_clean.values)
        else:
            imu_idx = np.linspace(0, len(df_cam) - 1, len(df_imu))
            accel_mag_synced = np.interp(np.arange(len(df_cam)), imu_idx, imu_clean.values)

    print("  → Khử nhiễu tín hiệu động học (EMA Filter)...")
    sh_x = sh_y = sh_z = None
    el_x = el_y = el_z = None
    wr_x = wr_y = wr_z = None
    curr_trail = []
    frame_dict = {}

    for idx, row in df_cam.iterrows():
        f_id    = int(row["frame"])
        acc_val = float(accel_mag_synced[idx])

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

        curr_trail.append((wr_rel, acc_val))
        if len(curr_trail) > TRAIL_LENGTH:
            curr_trail.pop(0)

        ang = calculate_elbow_angle([0, 0, 0], el_rel, wr_rel)

        frame_dict[f_id] = {
            "el":   [round(v, 5) for v in el_rel],
            "wr":   [round(v, 5) for v in wr_rel],
            "angle": round(float(ang), 2),
            "trail": [[round(v, 5) for v in p] for p, _ in curr_trail],
            "trail_acc": [round(float(a), 4) for _, a in curr_trail],
            "time_offset": round(float(cam_ts[idx] - cam_ts[0]), 4),
            "acc": round(float(acc_val), 4),
        }

    has_video = video_path.exists()

    valid_frames = [f for f in frames if f in frame_dict]
    time_series  = [frame_dict[f]["time_offset"] for f in valid_frames]
    angle_series = [frame_dict[f]["angle"]       for f in valid_frames]
    acc_series   = [frame_dict[f]["acc"]          for f in valid_frames]

    return {
        "session_name": session_folder.name,
        "frames":       [int(f) for f in valid_frames],
        "frame_dict":   {str(k): v for k, v in frame_dict.items()},
        "time_series":  time_series,
        "angle_series": angle_series,
        "acc_series":   acc_series,
        "has_imu":      has_imu,
        "has_video":    has_video,
    }

# =====================================================================
# BƯỚC 2: SINH HTML REPORT (ĐÃ FIX LỖI KẸT XOAY 3D & ĐÈ CHỮ)
# =====================================================================
def generate_html(data: dict, output_path: Path):
    sn       = data["session_name"]
    has_imu  = data["has_imu"]
    has_video = data.get("has_video", False)

    video_tag = (
        f'<video id="mainVideo" controls preload="auto">'
        f'<source src="raw_video.mp4" type="video/mp4"></video>'
    ) if has_video else (
        '<div class="no-video"><span>📹 Không có video đính kèm</span></div>'
    )

    imu_badge_cls  = "active"  if has_imu else "inactive"
    imu_badge_text = "● IMU FUSED" if has_imu else "○ CAM ONLY"
    chart_title    = "📈 Động học: Góc Khuỷu Tay & Gia Tốc" if has_imu else "📈 Động học: Góc Khuỷu Tay"
    
    y2_layout = (
        'yaxis2: { gridcolor:"rgba(0,0,0,0)", title: { text: "Gia tốc IMU (G)", font:{size:10,color:"#ff6b35"} },'
        ' overlaying:"y", side:"right", tickfont:{color:"#ff6b35",size:9} },'
    ) if has_imu else ""
    
    acc_trace = (
        'chartTraces.push({ type:"scatter", mode:"lines",'
        ' x:timeSeries, y:accSeries,'
        ' line:{color:"#ff6b35",width:1.5}, opacity:0.6,'
        ' name:"Gia tốc (G)", yaxis:"y2" });'
    ) if has_imu else ""

    payload = json.dumps({
        "frames":       data["frames"],
        "frame_dict":   data["frame_dict"],
        "time_series":  data["time_series"],
        "angle_series": data["angle_series"],
        "acc_series":   data["acc_series"],
        "has_imu":      has_imu,
    }, separators=(",",":"))

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Motion Capture Dashboard · {sn}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root {{
  --bg:#0b0d11; --surface:#131720; --card:#1b202e; --border:#262f42;
  --accent:#00e5ff; --accent2:#ff6b35; --green:#39d98a; --text:#e8edf5; --sub:#6c7d93;
  --mono:"JetBrains Mono",monospace; --ui:system-ui,sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--ui);height:100vh;overflow:hidden;display:flex;flex-direction:column}}
header{{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;background:var(--surface);border-bottom:1px solid var(--border);flex-shrink:0}}
.h-title{{font-size:15px;font-weight:700;letter-spacing:.5px}}
.badge{{font-family:var(--mono);font-size:11px;color:var(--sub);background:var(--card);padding:4px 10px;border-radius:4px;border:1px solid var(--border)}}
.imu{{font-size:10px;font-weight:700;padding:4px 10px;border-radius:4px;border:1px solid}}
.imu.active{{color:var(--green);border-color:var(--green);background:rgba(57,217,138,.08)}}
.imu.inactive{{color:var(--sub);border-color:var(--border)}}
.main{{display:grid;grid-template-columns:420px 1fr;flex:1;min-height:0;overflow:hidden}}
.left{{display:flex;flex-direction:column;background:var(--surface);border-right:1px solid var(--border);overflow:hidden}}
.panel-lbl{{font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--sub);padding:10px 16px;border-bottom:1px solid var(--border)}}
.vid-wrap{{flex:1;background:#000;display:flex;align-items:center;justify-content:center;min-height:0;overflow:hidden}}
#mainVideo{{width:100%;height:100%;object-fit:contain}}
.ctrl-area{{padding:16px;background:var(--card);border-top:1px solid var(--border)}}
.finfo{{display:flex;justify-content:space-between;margin-bottom:10px;font-family:var(--mono);font-size:12px}}
.fval{{color:var(--accent);font-weight:700}} .aval{{color:var(--green);font-weight:700}}
input[type=range]{{width:100%;height:4px;background:var(--border);outline:none;cursor:pointer;margin-bottom:12px}}
.ctrls{{display:flex;gap:8px;align-items:center}}
.btn{{background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:6px;cursor:pointer;font-size:12px;padding:6px 12px}}
.btn:hover{{border-color:var(--accent);background:var(--border)}}
.btn.primary{{background:var(--accent);color:#000;font-weight:700;border:none}}
select.spd{{background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:5px;font-family:var(--mono)}}
.right{{display:flex;flex-direction:column;min-height:0;overflow:hidden}}
.graphs-row{{display:grid;grid-template-columns:1fr 1fr 1fr;flex:0 0 340px;border-bottom:1px solid var(--border)}}
.gcell{{border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}}
.gcell:last-child{{border-right:none}}
.gtitle{{font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--sub);padding:10px 14px;border-bottom:1px solid var(--border)}}
.plotly-cell{{flex:1;min-height:0}}
.chart-area{{flex:1;display:flex;flex-direction:column;min-height:0;padding:8px 0}}
#chartAngle{{flex:1;min-height:0}}
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;gap:10px">
    <span class="h-title">📊 Hệ Thống Phân Tích Động Tác Đập Bóng 3D</span>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <span class="badge">{sn}</span>
    <span class="imu {imu_badge_cls}">{imu_badge_text}</span>
  </div>
</header>

<div class="main">
  <div class="left">
    <div class="panel-lbl">📹 Băng Ghi Hình Gốc</div>
    <div class="vid-wrap">{video_tag}</div>
    <div class="ctrl-area">
      <div class="finfo">
        <span>Khung hình: <span class="fval" id="dF">—</span></span>
        <span>Góc khuỷu: <span class="aval" id="dA">—°</span></span>
        <span id="dT" style="color:var(--sub)">0.000s</span>
      </div>
      <input type="range" id="slider" min="0" max="0" value="0" step="1">
      <div class="ctrls">
        <button class="btn primary" id="btnPlay" onclick="togglePlay()">▶ Play</button>
        <button class="btn" onclick="step(-1)">‹ Trước</button>
        <button class="btn" onclick="step(1)">Sau ›</button>
        <select class="spd" id="spdSel" onchange="updateSpd()" style="margin-left:auto">
          <option value="4">0.25×</option>
          <option value="2">0.50×</option>
          <option value="1" selected>1.00×</option>
        </select>
      </div>
    </div>
  </div>

  <div class="right">
    <div class="graphs-row">
      <div class="gcell">
        <div class="gtitle" style="color:var(--accent)">◈ Không Gian 3D Tương Tác</div>
        <div class="plotly-cell" id="g3d"></div>
      </div>
      <div class="gcell">
        <div class="gtitle" style="color:var(--accent2)">◈ Nhìn Ngang (Sideway Z-Y)</div>
        <div class="plotly-cell" id="gSide"></div>
      </div>
      <div class="gcell">
        <div class="gtitle" style="color:var(--green)">◈ Nhìn Từ Trên Xuống (Bird-Eye X-Z)</div>
        <div class="plotly-cell" id="gBird"></div>
      </div>
    </div>
    <div class="chart-area">
      <div class="gtitle" style="margin-left:14px">{chart_title}</div>
      <div id="chartAngle"></div>
    </div>
  </div>
</div>

<script>
const P = {payload};
const frames = P.frames, FD = P.frame_dict;
const timeSeries = P.time_series, angleSeries = P.angle_series, accSeries = P.acc_series;
const hasIMU = P.has_imu;
const minF = frames[0], maxF = frames[frames.length-1];
const FPS = 60.0;

const sl = document.getElementById('slider');
sl.min = minF; sl.max = maxF; sl.value = minF;
sl.addEventListener('input', () => seek(+sl.value));

const DARK = {{
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#6c7d93',family:'JetBrains Mono,monospace',size:9}},
  margin:{{l:45,r:45,t:15,b:35}}, showlegend:false
}};
const AX = {{ gridcolor:'#262f42', zerolinecolor:'#3e4b66', tickfont:{{size:9,color:'#6c7d93'}} }};
const R = [-0.65, 0.65];

const allX = frames.map(f => (FD[f]||{{}}).wr?.[0] ?? null);
const allY = frames.map(f => (FD[f]||{{}}).wr?.[1] ?? null);
const allZ = frames.map(f => (FD[f]||{{}}).wr?.[2] ?? null);

// --- TẠO BIỂU ĐỒ 3D ---
Plotly.newPlot('g3d', [
  {{ type:'scatter3d', mode:'lines', x:allX, y:allZ, z:allY, line:{{color:'#00e5ff', width:2}}, opacity:.15 }},
  {{ type:'scatter3d', mode:'lines+markers', x:[0,0,0], y:[0,0,0], z:[0,0,0], line:{{color:'#e8edf5',width:5}}, marker:{{size:[8,6,8], color:['#ff4d4d','#00e5ff','#ffcc00']}} }},
  {{ type:'scatter3d', mode:'lines', x:[], y:[], z:[], line:{{color:'#00e5ff',width:3.5}}, opacity:.9 }}
], {{
  ...DARK, margin:{{l:0,r:0,t:0,b:0}},
  scene:{{
    xaxis:{{...AX, title:'X', range:R}}, yaxis:{{...AX, title:'Depth Z', range:R}}, zaxis:{{...AX, title:'Height Y', range:R}},
    camera:{{eye:{{x:1.3,y:-1.3,z:0.8}}}}, aspectmode:'cube'
  }}
}}, {{responsive:true, displayModeBar:false}});

// --- TẠO BIỂU ĐỒ SIDEWAY ---
Plotly.newPlot('gSide', [
  {{ type:'scatter', mode:'lines', x:allZ, y:allY, line:{{color:'#ff6b35',width:1}}, opacity:.15 }},
  {{ type:'scatter', mode:'lines+markers', x:[0,0,0], y:[0,0,0], line:{{color:'#ff6b35',width:3}}, marker:{{size:[8,5,8],color:['#ff4d4d','#ff6b35','#ffcc00']}} }},
  {{ type:'scatter', mode:'lines', x:[], y:[], line:{{color:'#ff6b35',width:2.5}} }}
], {{ ...DARK, xaxis:{{...AX, title:'Depth Z', range:R}}, yaxis:{{...AX, title:'Height Y', range:R, scaleanchor:'x'}} }}, {{responsive:true, displayModeBar:false}});

// --- TẠO BIỂU ĐỒ BIRD-EYE ---
Plotly.newPlot('gBird', [
  {{ type:'scatter', mode:'lines', x:allX, y:allZ, line:{{color:'#39d98a',width:1}}, opacity:.15 }},
  {{ type:'scatter', mode:'lines+markers', x:[0,0,0], y:[0,0,0], line:{{color:'#39d98a',width:3}}, marker:{{size:[8,5,8],color:['#ff4d4d','#39d98a','#ffcc00']}} }},
  {{ type:'scatter', mode:'lines', x:[], y:[], line:{{color:'#39d98a',width:2.5}} }}
], {{ ...DARK, xaxis:{{...AX, title:'Lateral X', range:R}}, yaxis:{{...AX, title:'Depth Z', range:R, scaleanchor:'x'}} }}, {{responsive:true, displayModeBar:false}});

// --- BIỂU ĐỒ PHÂN TÍCH ĐỘNG HỌC ---
const chartTraces = [
  {{ type:'scatter', mode:'lines', x:timeSeries, y:angleSeries, line:{{color:'#39d98a',width:2}}, name:'Góc khuỷu (°)' }}
];
{acc_trace}
chartTraces.push({{ type:'scatter', mode:'lines', x:[0,0], y:[0,180], line:{{color:'#00e5ff',width:1.5,dash:'dash'}}, name:'Hiện tại' }});
const cursorIdx = chartTraces.length - 1;

Plotly.newPlot('chartAngle', chartTraces, {{
  ...DARK, margin:{{l:50,r:50,t:10,b:35}},
  xaxis:{{...AX, title:'Thời gian (s)'}},
  yaxis:{{...AX, title:'Góc Khuỷu (°)', range:[0,180]}},
  {y2_layout}
  showlegend:true,
  legend:{{x:0.01, y:0.99, bgcolor:'rgba(19,23,32,0.85)', bordercolor:'#262f42', borderwidth:1}}
}}, {{responsive:true, displayModeBar:false}});

// --- HÀM SEEK ĐỒNG BỘ: SỬ DỤNG RESTYLE ĐỂ TRÁNH KẸT CHUỘT XOAY 3D ---
let curF = minF;
function seek(f, fromVid) {{
  f = Math.max(minF, Math.min(maxF, f)); curF = f; sl.value = f;
  const fd = FD[f] || FD[String(f)];
  if (!fd) return;

  document.getElementById('dF').textContent = f;
  document.getElementById('dA').textContent = fd.angle.toFixed(1) + '°';
  document.getElementById('dT').textContent = fd.time_offset.toFixed(3) + 's';

  const vid = document.getElementById('mainVideo');
  if (vid && !fromVid) {{
    const tgt = (f - minF) / FPS;
    if (Math.abs(vid.currentTime - tgt) > 0.05) vid.currentTime = tgt;
  }}

  const tx = fd.trail.map(p=>p[0]), ty = fd.trail.map(p=>p[1]), tz = fd.trail.map(p=>p[2]);

  // SỬ DỤNG Plotly.restyle thay vì Plotly.update ĐỂ KHÔNG BỊ KHÓA GÓC XOAY CAMERA KHI PLAY
  Plotly.restyle('g3d', {{x: [[0, fd.el[0], fd.wr[0]]], y: [[0, fd.el[2], fd.wr[2]]], z: [[0, fd.el[1], fd.wr[1]]]}}, [1]);
  Plotly.restyle('g3d', {{x: [tx], y: [tz], z: [ty]}}, [2]);

  Plotly.restyle('gSide', {{x: [[0, fd.el[2], fd.wr[2]]], y: [[0, fd.el[1], fd.wr[1]]]}}, [1]);
  Plotly.restyle('gSide', {{x: [tz], y: [ty]}}, [2]);

  Plotly.restyle('gBird', {{x: [[0, fd.el[0], fd.wr[0]]], y: [[0, fd.el[2], fd.wr[2]]]}}, [1]);
  Plotly.restyle('gBird', {{x: [tx], y: [tz]}}, [2]);

  Plotly.restyle('chartAngle', {{x: [[fd.time_offset, fd.time_offset]]}}, [cursorIdx]);
}}

// --- ĐIỀU KHIỂN VIDEO ---
const ps = {{on:false, spd:1.0}};
const btn = document.getElementById('btnPlay');

function updateSpd() {{
  ps.spd = 1.0 / parseFloat(document.getElementById('spdSel').value);
  const v = document.getElementById('mainVideo'); if (v) v.playbackRate = ps.spd;
}}
function togglePlay() {{ ps.on ? pause() : play(); }}
function play() {{
  ps.on = true; btn.textContent = '⏸ Pause';
  const v = document.getElementById('mainVideo'); if (v) {{ v.playbackRate = ps.spd; v.play(); }}
}}
function pause() {{ ps.on = false; btn.textContent = '▶ Play'; const v = document.getElementById('mainVideo'); if (v) v.pause(); }}

const vid = document.getElementById('mainVideo');
if (vid) {{
  vid.addEventListener('timeupdate', () => {{
    if (!ps.on) return;
    const cf = Math.max(minF, Math.min(maxF, Math.round(vid.currentTime * FPS) + minF));
    if (Math.abs(cf - curF) >= 1) seek(cf, true);
  }});
  vid.addEventListener('ended', pause);
}}
function step(d) {{ pause(); seek(curF + d); }}
seek(minF);
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\n✨ Báo cáo HTML tích hợp All-In-One đã sẵn sàng tại: {output_path}")

# =====================================================================
# SYSTEM ENTRY CONTROL
# =====================================================================
def run(input_arg):
    path = Path(input_arg)
    
    # KIỂM TRA NẾU ĐƯỜNG DẪN LÀ FILE VIDEO (YÊU CẦU 4: XỬ LÝ VIDEO CŨ)
    if path.is_file() and path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        session_folder = extract_data_from_video(path)
    elif path.is_dir():
        session_folder = path
    else:
        print(f"[LỖI] Đường dẫn đầu vào không hợp lệ: {input_arg}")
        return

    data = process_session(session_folder)
    if data is None: return
    
    output_path = session_folder / "motion_report.html"
    generate_html(data, output_path)
    
    import webbrowser
    webbrowser.open(output_path.resolve().as_uri())

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "recording/20260602_160951"
    run(target)