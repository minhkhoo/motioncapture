import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button 
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# =====================================================================
# CAU HINH BO LOC DONG HOC (EMA Filter)
# =====================================================================
TRAIL_LENGTH = 45  # Do dai duong quy dao cua co tay

# Bo loc lam muot chong nhieu toa do tu MediaPipe
SMOOTH_SH = 0.85  
SMOOTH_EL = 0.50  
SMOOTH_WR = 0.30  

def calculate_elbow_angle(shoulder, elbow, wrist):
    """Tinh goc gap khuyl tay trong khong gian 3D (0 - 180 do)"""
    v1 = np.array(elbow) - np.array(shoulder)
    v2 = np.array(wrist) - np.array(elbow)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return 180 - angle 

# =====================================================================
# DO THI TUONG TAC CAO CAP PHUC VU NGHIEN CUU (INTERACTIVE GRAPH)
# =====================================================================
def interactive_replay(session_folder):
    csv_path = Path(session_folder) / "camera_data.csv"
    imu_path = Path(session_folder) / "imu_data.csv"
    
    if not csv_path.exists():
        print(f"Khong tim thay file du lieu AI tai: {csv_path}")
        return
        
    # 1. DOC DU LIEU CAMERA HAU KY
    df_cam = pd.read_csv(csv_path)
    cam_timestamps = df_cam['timestamp'].values
    frames = df_cam['frame'].values
    min_frame = int(frames.min())
    max_frame = int(frames.max())

    # 2. XU LY & DONG BO DATA FUSION VOI CAM BIEN IMU (NOI SUY LUONG THOI GIAN)
    accel_mag_synced = np.zeros(len(df_cam))
    if imu_path.exists():
        print("Dang tien hanh Data Fusion: Noi suy dong bo IMU va Camera...")
        df_imu = pd.read_csv(imu_path)
        
        imu_mag = np.sqrt(df_imu['ax']**2 + df_imu['ay']**2 + df_imu['az']**2)
        imu_mag_clean = np.abs(imu_mag - np.mean(imu_mag.iloc[:15]))
        
        if 'timestamp' in df_imu.columns:
            accel_mag_synced = np.interp(cam_timestamps, df_imu['timestamp'].values, imu_mag_clean)
        else:
            imu_indices = np.linspace(0, len(df_cam)-1, len(df_imu))
            accel_mag_synced = np.interp(np.arange(len(df_cam)), imu_indices, imu_mag_clean)
    else:
        print("Khong tim thay file imu_data.csv. Do thi chi hien thi goc xuong tay.")

    # 3. TIEN XU LY BO LOC VA TINH TOAN GOC CHO TOAN BO SESSION
    frame_dict = {}
    
    sh_x, sh_y, sh_z = None, None, None
    el_x, el_y, el_z = None, None, None
    wr_x, wr_y, wr_z = None, None, None
    curr_trail = []

    print("Dang toi uu hoa toa do dong hoc chuyen dong...")
    for idx, row in df_cam.iterrows():
        f_id = int(row["frame"])
        acc_val = accel_mag_synced[idx]
        
        raw = [[row["shoulder_x"], -row["shoulder_y"], -row["shoulder_z"]],
               [row["elbow_x"], -row["elbow_y"], -row["elbow_z"]],
               [row["wrist_x"], -row["wrist_y"], -row["wrist_z"]]]
               
        if sh_x is None: 
            sh_x, sh_y, sh_z = raw[0]; el_x, el_y, el_z = raw[1]; wr_x, wr_y, wr_z = raw[2]
            
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
            
        ang = calculate_elbow_angle([0,0,0], el_rel, wr_rel)
            
        frame_dict[f_id] = {
            "el": el_rel,
            "wr": wr_rel,
            "angle": ang,
            "trail": list(curr_trail),
            "time_offset": cam_timestamps[idx] - cam_timestamps[0]
        }

    # 4. THIET KE DO THI PHAN TICH CHUYEN NGHIEP CHIA DOI (DUAL PLOT)
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"He Thong Phan Tich Co Sinh Hoc - Session: {Path(session_folder).name}", fontsize=13, fontweight='bold')
    
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)
    ax2d_twin = ax2d.twinx() 
    
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    time_axis = [frame_dict[f]["time_offset"] for f in frames if f in frame_dict]
    angles_axis = [frame_dict[f]["angle"] for f in frames if f in frame_dict]
    accel_axis = [accel_mag_synced[i] for i, f in enumerate(frames) if f in frame_dict]
    
    line_ang, = ax2d.plot(time_axis, angles_axis, 'g-', label='Goc Khuyl Tay (do)', linewidth=2)
    line_acc, = ax2d_twin.plot(time_axis, accel_axis, 'r-', label='Gia Toc IMU (G)', alpha=0.5, linewidth=1.5)
    v_cursor = ax2d.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Frame Hien Tai')
    
    ax2d.set_xlabel("Thoi gian (Giay)", fontweight='bold')
    ax2d.set_ylabel("Goc Gap (Do)", color='g', fontweight='bold')
    ax2d_twin.set_ylabel("Gia Toc Tong Hop (Clean Magnitude)", color='r', fontweight='bold')
    ax2d.grid(True, alpha=0.3)
    
    lines = [line_ang, line_acc, v_cursor]
    ax2d.legend(lines, [l.get_label() for l in lines], loc='upper right')

    norm = mcolors.Normalize(vmin=0, vmax=max(accel_mag_synced)*0.8 if max(accel_mag_synced)>0 else 15)
    cmap = plt.get_cmap('jet')

    slider_ax = plt.axes([0.35, 0.06, 0.45, 0.03])
    slider = Slider(slider_ax, 'Khung Hinh (Frame)', min_frame, max_frame, valinit=min_frame, valfmt='%d')
    
    btn_play_ax = plt.axes([0.15, 0.05, 0.08, 0.05])
    btn_play = Button(btn_play_ax, '▶ Play', color='#e1f5fe', hovercolor='#b3e5fc')
    
    play_state = {"is_playing": False}
    timer = fig.canvas.new_timer(interval=16) 

    last_valid_data = None

    def update_graph(val):
        nonlocal last_valid_data
        target_f = int(slider.val)
        
        ax3d.clear()
        if target_f in frame_dict:
            last_valid_data = frame_dict[target_f]
            
        if last_valid_data is not None:
            el = last_valid_data["el"]
            wr = last_valid_data["wr"]
            angle = last_valid_data["angle"]
            trail_data = last_valid_data["trail"]
            current_time = last_valid_data["time_offset"]
            
            ax3d.plot([0, el[0], wr[0]], [0, el[1], wr[1]], [0, el[2], wr[2]], 'k-', lw=4)
            ax3d.scatter([0, el[0], wr[0]], [0, el[1], wr[1]], [0, el[2], wr[2]], c=['r','g','b'], s=120)
            
            for i in range(len(trail_data)-1):
                p1, _ = trail_data[i]
                p2, acc = trail_data[i+1]
                ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=cmap(norm(acc)), lw=3)
                
            ax3d.text2D(0.05, 0.95, f"Goc Khuyl: {angle:.1f}°\nThoi gian: {current_time:.3f}s", 
                        transform=ax3d.transAxes, bbox=dict(facecolor='w', alpha=0.85), fontsize=11)
            
            v_cursor.set_xdata([current_time, current_time])

        ax3d.set_xlim(-0.6, 0.6); ax3d.set_ylim(-0.6, 0.6); ax3d.set_zlim(-0.6, 0.6)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.view_init(elev=10, azim=-90)
        ax3d.set_title(f"Mo Hinh Xuong Tay 3D - Frame: {target_f}")
        fig.canvas.draw_idle()

    def on_step():
        curr = int(slider.val)
        if curr < max_frame:
            slider.set_val(curr + 1)
        else:
            timer.stop()
            play_state["is_playing"] = False
            btn_play.label.set_text('▶ Play')

    def toggle_play(event):
        if play_state["is_playing"]:
            timer.stop()
            btn_play.label.set_text('▶ Play')
        else:
            timer.start()
            btn_play.label.set_text('⏸ Pause')
        play_state["is_playing"] = not play_state["is_playing"]

    def on_key_press(event):
        if event.key == ' ': 
            toggle_play(None)
        elif event.key == 'right': 
            curr = int(slider.val)
            if curr < max_frame: slider.set_val(curr + 1)
        elif event.key == 'left': 
            curr = int(slider.val)
            if curr > min_frame: slider.set_val(curr - 1)

    timer.add_callback(on_step)
    btn_play.on_clicked(toggle_play)
    slider.on_changed(update_graph)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    update_graph(min_frame)
    print("\n MEO DIEU KHIEN PHAN TICH:")
    print("   - Phim [Spacebar]: Bat/Tat Replay tu dong.")
    print("   - Phim [->] hoac [<-]: Di chuyen tung khung hinh.")
    print("="*80)
    plt.show()

if __name__ == "__main__":
    interactive_replay(sys.argv[1] if len(sys.argv) > 1 else "recording/20260602_160951")