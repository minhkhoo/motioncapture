import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def sync_and_visualize(session_folder):
    session_dir = Path(session_folder)
    cam_path = session_dir / "camera_data.csv"
    imu_path = session_dir / "imu_data.csv" # Sửa lại tên file IMU nếu bạn đặt tên khác

    if not cam_path.exists() or not imu_path.exists():
        print("❌ Không tìm thấy đủ 2 file camera_data.csv và imu_data.csv trong thư mục!")
        return

    print("⏳ Đang đồng bộ dữ liệu Camera và IMU...")

    # 1. ĐỌC DỮ LIỆU
    df_cam = pd.read_csv(cam_path)
    df_imu = pd.read_csv(imu_path)

    # 2. TÍNH TOÁN GIA TỐC TỔNG CỦA IMU (Tổng hợp lực 3 trục)
    # Công thức: a_total = căn bậc 2 của (ax^2 + ay^2 + az^2)
    df_imu['a_total'] = np.sqrt(df_imu['ax']**2 + df_imu['ay']**2 + df_imu['az']**2)

    # 3. CHUẨN BỊ ĐỒNG BỘ (Bắt buộc phải sort theo thời gian)
    df_cam = df_cam.sort_values('timestamp')
    df_imu = df_imu.sort_values('pc_time')

    # 4. GỘP DỮ LIỆU BẰNG NỘI SUY (MERGE ASOF)
    # Lấy Camera làm gốc, tìm dòng IMU có thời gian gần nhất ghép vào
    df_synced = pd.merge_asof(
        df_cam, 
        df_imu, 
        left_on='timestamp', 
        right_on='pc_time', 
        direction='nearest',
        tolerance=0.05 # Cảnh báo nếu 2 thiết bị lệch nhau quá 50ms
    )

    # Lưu lại file đã gộp để sau này làm AI nhận diện
    sync_save_path = session_dir / "synced_master.csv"
    df_synced.to_csv(sync_save_path, index=False)
    print(f"✅ Đã gộp thành công! File lưu tại: {sync_save_path}")

    # =========================================================
    # 5. VẼ ĐỒ THỊ NHẬN DIỆN CÚ ĐẬP (BIOMECHANICS GRAPH)
    # =========================================================
    print("📈 Đang mở biểu đồ phân tích...")
    
    # Tạo 2 biểu đồ xếp chồng lên nhau, dùng chung trục ngang (Thời gian)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('PHÂN TÍCH CHUYỂN ĐỘNG ĐẬP BÓNG (SYNCED DATA)', fontsize=16, fontweight='bold')

    # Trừ đi thời gian gốc để đồ thị bắt đầu từ 0s cho dễ nhìn
    time_zero = df_synced['timestamp'].iloc[0]
    plot_time = df_synced['timestamp'] - time_zero

    # --- ĐỒ THỊ 1: QUỸ ĐẠO CỔ TAY (CAMERA) ---
    # Trong OpenCV, Y hướng xuống, nên ta thêm dấu âm (-) để Y hướng lên trời giống đời thực
    ax1.plot(plot_time, -df_synced['wrist_y'], label='Độ cao Cổ tay (Wrist Y)', color='blue', linewidth=2)
    ax1.plot(plot_time, -df_synced['shoulder_y'], label='Độ cao Vai (Shoulder Y)', color='gray', linestyle='--')
    ax1.set_ylabel('Độ cao trên video (Pixels/Tương đối)')
    ax1.set_title('Quỹ đạo tay (Nhô lên cao là lúc vung tay)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- ĐỒ THỊ 2: LỰC GIA TỐC (IMU) ---
    ax2.plot(plot_time, df_synced['a_total'], label='Gia tốc tổng hợp (Lực vung)', color='red', linewidth=2)
    ax2.set_ylabel('Gia tốc (G)')
    ax2.set_xlabel('Thời gian (Giây)')
    ax2.set_title('Lực cổ tay (Các đỉnh nhọn là lúc phát lực đập)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Thay tên folder dưới đây bằng tên folder bạn vừa record
    # Hoặc gõ lệnh: python sync_data.py recording/ten_folder_cua_ban
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Tự động tìm thư mục mới nhất trong thư mục recording
        recording_dir = Path("recording")
        subfolders = [f for f in recording_dir.iterdir() if f.is_dir()]
        if subfolders:
            target = max(subfolders, key=lambda x: x.stat().st_mtime)
        else:
            target = ""
            print("Chưa có data!")
            
    if target:
        sync_and_visualize(target)