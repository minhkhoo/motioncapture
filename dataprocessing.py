import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.figure import Figure 
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.widgets import Slider, Button  

import numpy as np
import pandas as pd
from pathlib import Path
import sys

from shareddata import shared_data

# =========================
# CONFIG
# =========================
TRAIL_LENGTH = 30

# Tách riêng hệ số lọc cho từng khớp dựa trên tốc độ chuyển động thực tế
SMOOTH_SH = 0.85  # Vai di chuyển chậm 
SMOOTH_EL = 0.45  # Khuỷu tay di chuyển vừa 
SMOOTH_WR = 0.15  # Cổ tay vung cực nhanh 

# =========================
# GLOBALS (Filters added for all 3 joints)
# =========================
smooth_shoulder = None
smooth_elbow = None
smooth_wrist = None
wrist_trail = []

# =========================
# GET LATEST CAMERA POSE
# =========================
def get_pose():
    if not shared_data.camera_buffer:
        return None
    return shared_data.camera_buffer[-1]


# =====================================================================
# CHỨC NĂNG 1: live graph 
# =====================================================================
def live_graph():
    global smooth_shoulder, smooth_elbow, smooth_wrist
    global wrist_trail

    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    while (
    plt.fignum_exists(fig.number)
    and shared_data.program_running
    ):
        # =========================
        # WAIT UNTIL RECORDING
        # =========================
        if not shared_data.graph_running:
            plt.pause(0.1)
            continue

        pose = get_pose()

        if pose is not None:
            shoulder = np.array(pose["shoulder"])
            elbow = np.array(pose["elbow"])
            wrist = np.array(pose["wrist"])

            # =========================
            # NOISE FILTER WITH SPLIT COEFFICIENTS
            # =========================
            if smooth_shoulder is None:
                smooth_shoulder = shoulder.copy()
                smooth_elbow = elbow.copy()
                smooth_wrist = wrist.copy()

            smooth_shoulder = SMOOTH_SH * smooth_shoulder + (1 - SMOOTH_SH) * shoulder
            smooth_elbow = SMOOTH_EL * smooth_elbow + (1 - SMOOTH_EL) * elbow
            smooth_wrist = SMOOTH_WR * smooth_wrist + (1 - SMOOTH_WR) * wrist

            # =========================
            # COORDINATE TRANSFORMATION
            # =========================
            sh_x, sh_y, sh_z = smooth_shoulder[0], -smooth_shoulder[1], -smooth_shoulder[2]
            el_x, el_y, el_z = smooth_elbow[0], -smooth_elbow[1], -smooth_elbow[2]
            wr_x, wr_y, wr_z = smooth_wrist[0], -smooth_wrist[1], -smooth_wrist[2]

            # =========================
            # RELATIVE COORDINATES (Shoulder locked at 0,0,0)
            # =========================
            shoulder_rel = np.array([0.0, 0.0, 0.0])
            elbow_rel = np.array([el_x - sh_x, el_y - sh_y, el_z - sh_z])
            wrist_rel = np.array([wr_x - sh_x, wr_y - sh_y, wr_z - sh_z])

            # Save wrist trajectory trail
            wrist_trail.append(wrist_rel.copy())
            if len(wrist_trail) > TRAIL_LENGTH:
                wrist_trail.pop(0)

            # =========================
            # RENDER ARM
            # =========================
            ax.clear()
            xs = [shoulder_rel[0], elbow_rel[0], wrist_rel[0]]
            ys = [shoulder_rel[1], elbow_rel[1], wrist_rel[1]]
            zs = [shoulder_rel[2], elbow_rel[2], wrist_rel[2]]

            ax.scatter(xs, ys, zs, s=120, c=['red', 'green', 'blue'])
            ax.plot(xs, ys, zs, linewidth=4, color='black')

            # =========================
            # RENDER WRIST TRAIL
            # =========================
            if len(wrist_trail) > 1:
                tx = [p[0] for p in wrist_trail]
                ty = [p[1] for p in wrist_trail]
                tz = [p[2] for p in wrist_trail]
                ax.plot(tx, ty, tz, alpha=0.6, linewidth=2, color='magenta', linestyle='--')

            # =========================
            # VIEW LIMITS & CONFIG
            # =========================
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(-0.6, 0.6)
            ax.set_zlim(-0.6, 0.6)
            ax.set_box_aspect([1, 1, 1])

            ax.set_xlabel("X (Left - Right)")
            ax.set_ylabel("Y (Down - Up)")
            ax.set_zlabel("Z (Far - Near)")

            ax.view_init(elev=0, azim=-90)

            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.pause(0.01)

    plt.ioff()
    plt.close(fig)


# =====================================================================
# CHỨC NĂNG 2: interactive replay 
# =====================================================================
def interactive_replay(session_folder):
    csv_path = Path(session_folder) / "camera_data.csv"
    if not csv_path.exists():
        print(f"Khong tim thay file du lieu camera_data.csv tai: {csv_path}")
        return

    print(f"Dang doc va phan tich du lieu tu: {session_folder}...")
    history = pd.read_csv(csv_path)

    if history.empty:
        print("File dữ liệu CSV rỗng.")
        return

    # 1. Tính toán trước (Pre-computation) toàn bộ tọa độ
    sh_x_s, sh_y_s, sh_z_s = None, None, None
    el_x_s, el_y_s, el_z_s = None, None, None
    wr_x_s, wr_y_s, wr_z_s = None, None, None

    el_rels, wr_rels, trails = [], [], []
    current_trail = []
    
    max_frames = len(history)

    for frame in range(max_frames):
        row = history.iloc[frame]
        
        raw_sh_x, raw_sh_y, raw_sh_z = row["shoulder_x"], -row["shoulder_y"], -row["shoulder_z"]
        raw_el_x, raw_el_y, raw_el_z = row["elbow_x"], -row["elbow_y"], -row["elbow_z"]
        raw_wr_x, raw_wr_y, raw_wr_z = row["wrist_x"], -row["wrist_y"], -row["wrist_z"]

        if sh_x_s is None:
            sh_x_s, sh_y_s, sh_z_s = raw_sh_x, raw_sh_y, raw_sh_z
            el_x_s, el_y_s, el_z_s = raw_el_x, raw_el_y, raw_el_z
            wr_x_s, wr_y_s, wr_z_s = raw_wr_x, raw_wr_y, raw_wr_z

        sh_x_s = SMOOTH_SH * sh_x_s + (1 - SMOOTH_SH) * raw_sh_x
        sh_y_s = SMOOTH_SH * sh_y_s + (1 - SMOOTH_SH) * raw_sh_y
        sh_z_s = SMOOTH_SH * sh_z_s + (1 - SMOOTH_SH) * raw_sh_z

        el_x_s = SMOOTH_EL * el_x_s + (1 - SMOOTH_EL) * raw_el_x
        el_y_s = SMOOTH_EL * el_y_s + (1 - SMOOTH_EL) * raw_el_y
        el_z_s = SMOOTH_EL * el_z_s + (1 - SMOOTH_EL) * raw_el_z

        wr_x_s = SMOOTH_WR * wr_x_s + (1 - SMOOTH_WR) * raw_wr_x
        wr_y_s = SMOOTH_WR * wr_y_s + (1 - SMOOTH_WR) * raw_wr_y
        wr_z_s = SMOOTH_WR * wr_z_s + (1 - SMOOTH_WR) * raw_wr_z

        el_rel = [el_x_s - sh_x_s, el_y_s - sh_y_s, el_z_s - sh_z_s]
        wr_rel = [wr_x_s - sh_x_s, wr_y_s - sh_y_s, wr_z_s - sh_z_s]

        el_rels.append(el_rel)
        wr_rels.append(wr_rel)

        current_trail.append(wr_rel)
        if len(current_trail) > TRAIL_LENGTH:
            current_trail.pop(0)
        trails.append(list(current_trail))

    # 2. Khởi tạo giao diện tương tác Matplotlib
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2) 

    #  TẠO NÚT PLAY / PAUSE
    ax_play = plt.axes([0.15, 0.05, 0.1, 0.04])
    btn_play = Button(ax_play, 'Play', hovercolor='0.9')

    # Dịch Slider sang phải nhường chỗ cho Nút Play
    ax_slider = plt.axes([0.35, 0.05, 0.5, 0.04])
    slider = Slider(ax_slider, 'Khung hinh', 0, max_frames - 1, valinit=0, valfmt='%d')
    
    ax.view_init(elev=0, azim=-90)

    #  LOGIC CHO BỘ ĐẾM THỜI GIAN VÀ AUTO PLAY
    play_state = {"is_playing": False}
    timer = fig.canvas.new_timer(interval=33) # 33ms ~ 30FPS

    def update_graph(val):
        frame_idx = int(slider.val)
        
        # Giữ lại góc xoay hiện tại khi chuột xoay dở
        current_elev, current_azim = ax.elev, ax.azim
        
        ax.clear()
        
        el_rel = el_rels[frame_idx]
        wr_rel = wr_rels[frame_idx]
        wrist_trail_frame = trails[frame_idx]
        
        xs = [0, el_rel[0], wr_rel[0]]
        ys = [0, el_rel[1], wr_rel[1]]
        zs = [0, el_rel[2], wr_rel[2]]

        # Vẽ các khớp và xương cánh tay
        ax.scatter(xs, ys, zs, s=120, c=['red', 'green', 'blue'])
        ax.plot(xs, ys, zs, linewidth=4, color='black')

        # Vẽ đường vết quỹ đạo vung tay (Trail)
        if len(wrist_trail_frame) > 1:
            tx = [p[0] for p in wrist_trail_frame]
            ty = [p[1] for p in wrist_trail_frame]
            tz = [p[2] for p in wrist_trail_frame]
            ax.plot(tx, ty, tz, alpha=0.6, linewidth=2, color='magenta', linestyle='--')

        # Cấu hình không gian cố định đồng bộ
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_xlabel("X (Left - Right)")
        ax.set_ylabel("Y (Down - Up)")
        ax.set_zlabel("Z (Far - Near)")
        
        # Áp ngược góc xoay tự do vào frame mới
        ax.view_init(elev=current_elev, azim=current_azim)
        fig.canvas.draw_idle()

    #  HÀM TỰ ĐỘNG TUA KHUNG HÌNH
    def auto_step():
        current_val = int(slider.val)
        if current_val < max_frames - 1:
            slider.set_val(current_val + 1)
        else:
            slider.set_val(0)
            toggle_play(None) # Dừng lại khi hết video

    #  HÀM XỬ LÝ KHI BẤM NÚT PLAY/PAUSE
    def toggle_play(event):
        play_state["is_playing"] = not play_state["is_playing"]
        if play_state["is_playing"]:
            btn_play.label.set_text("Pause")
            timer.start()
        else:
            btn_play.label.set_text("Play")
            timer.stop()
        fig.canvas.draw_idle()

    # Kết nối các sự kiện
    timer.add_callback(auto_step)
    btn_play.on_clicked(toggle_play)
    slider.on_changed(update_graph)
    
    update_graph(0) # Hiển thị frame đầu tiên làm mẫu

    plt.show()


# =====================================================================
# LUỒNG CHẠY TỰ ĐỘNG: Phân tách gọi từ File .bat hoặc chạy trực tiếp
# =====================================================================
if __name__ == "__main__":
    # Nếu file .bat gọi và truyền đường dẫn folder vào qua sys.argv
    if len(sys.argv) > 1:
        TARGET_SESSION = sys.argv[1]
    else:
        # Đường dẫn dự phòng nếu bạn lỡ bấm chạy tay trực tiếp file này
        TARGET_SESSION = "recording/20260602_160951" 
    
    interactive_replay(TARGET_SESSION)