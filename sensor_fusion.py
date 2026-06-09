import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
from pathlib import Path
import sys

# ==========================================
# CÁC HÀM LỌC TÍN HIỆU (SIGNAL FILTERS)
# ==========================================
def low_pass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def high_pass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def sensor_fusion(session_folder):
    session_dir = Path(session_folder)
    cam_path = session_dir / "camera_data.csv"
    imu_path = session_dir / "imu_data.csv" 

    if not cam_path.exists() or not imu_path.exists():
        print(f"❌ Không tìm thấy đủ 2 file CSV trong: {session_dir}")
        return

    print(f"⏳ Đang thực hiện Sensor Fusion cho: {session_folder}...")

    # 1. Đọc và đồng bộ dữ liệu theo mốc thời gian
    df_cam = pd.read_csv(cam_path).sort_values('timestamp')
    df_imu = pd.read_csv(imu_path).sort_values('pc_time')

    df_synced = pd.merge_asof(
        df_cam, df_imu, 
        left_on='timestamp', right_on='pc_time', 
        direction='nearest'
    )
    
    # 2. Lấy dữ liệu thô
    fps = 60.0  # Tần số lấy mẫu gốc của hệ thống
    dt = 1.0 / fps
    
    # Trục Z của Camera (Chiều sâu bị nhiễu cần sửa)
    cam_z = df_synced['wrist_z'].values
    
    # Tính Gia tốc động của IMU (Loại bỏ trọng lực 1G)
    # Công thức: a_dynamic = sqrt(ax^2 + ay^2 + az^2) - 1.0
    a_total = np.sqrt(df_synced['ax']**2 + df_synced['ay']**2 + df_synced['az']**2)
    a_dynamic = a_total - np.mean(a_total[:20]) # Lấy vài frame đầu lúc đứng yên làm mốc trừ nhiễu tĩnh

    # 3. TÍCH PHÂN IMU ĐỂ TÌM QUÃNG ĐƯỜNG (DOUBLE INTEGRATION)
    # Tích phân lần 1: Gia tốc -> Vận tốc
    velocity = cumulative_trapezoid(a_dynamic, dx=dt, initial=0)
    # Khử trôi dạt vận tốc bằng High-pass filter
    velocity_filtered = high_pass_filter(velocity, cutoff=0.5, fs=fps)
    
    # Tích phân lần 2: Vận tốc -> Quãng đường (Displacement)
    displacement = cumulative_trapezoid(velocity_filtered, dx=dt, initial=0)

    # 4. CHUẨN HÓA TỶ LỆ (AUTO-SCALE)
    # Vì Displacement của IMU đo bằng mét, còn Camera đo bằng Pixel (0-1), 
    # ta phải ép độ biến thiên của IMU cho bằng với Camera để mix được với nhau.
    cam_variance = np.max(cam_z) - np.min(cam_z)
    imu_variance = np.max(displacement) - np.min(displacement)
    scale_factor = cam_variance / (imu_variance + 1e-6)
    
    scaled_displacement = displacement * scale_factor

    # 5. THỰC HIỆN COMPLEMENTARY FILTER (CỐT LÕI SENSOR FUSION)
    # Lấy nền của Camera (Lọc bỏ nhiễu giật cục)
    cam_z_lpf = low_pass_filter(cam_z, cutoff=2.0, fs=fps)
    
    # Lấy đỉnh nhọn của IMU (Lọc bỏ trôi dạt)
    imu_z_hpf = high_pass_filter(scaled_displacement, cutoff=1.0, fs=fps)
    
    # Lai tạo: Trục Z hoàn hảo
    fused_z = cam_z_lpf - imu_z_hpf # Dùng dấu trừ vì Z camera tiến về phía trước là số âm

    # Lưu kết quả để sau này nhúng vào file 3D
    df_synced['fused_wrist_z'] = fused_z
    master_path = session_dir / "master_fused_data.csv"
    df_synced.to_csv(master_path, index=False)
    
    # ==========================================
    # VẼ ĐỒ THỊ CHỨNG MINH KẾT QUẢ
    # ==========================================
    time_axis = df_synced['timestamp'] - df_synced['timestamp'].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.title("SO SÁNH TRỤC SÂU (Z-AXIS): CAMERA GỐC vs SENSOR FUSION", fontsize=14, fontweight='bold')
    
    # Đường 1: Camera gốc (Nhiễu, méo mó khi vung tay nhanh)
    plt.plot(time_axis, cam_z, label='Camera Gốc (Nhiễu & Chậm)', color='lightgray', linestyle='--', linewidth=2)
    
    # Đường 2: Dữ liệu đã hợp nhất (Cực mượt, giữ được biên độ đỉnh của IMU)
    plt.plot(time_axis, fused_z, label='Fused Z (Camera + IMU)', color='blue', linewidth=2.5)
    
    # Đường 3: Gia tốc IMU để đối chiếu
    # Chuẩn hóa đồ thị IMU xuống đáy màn hình để dễ nhìn
    plot_imu = (a_dynamic / np.max(np.abs(a_dynamic))) * cam_variance * 0.5 + np.min(cam_z)
    plt.plot(time_axis, plot_imu, label='Gia tốc phát lực (IMU)', color='red', alpha=0.5)

    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Tọa độ Z (Chiều sâu)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TARGET_SESSION = sys.argv[1]
    else:
        # Tự động tìm thư mục record mới nhất
        recording_dir = Path("recording")
        subfolders = [f for f in recording_dir.iterdir() if f.is_dir()]
        if subfolders:
            TARGET_SESSION = max(subfolders, key=lambda x: x.stat().st_mtime)
        else:
            TARGET_SESSION = ""
            print("Chưa có data!")
            
    if TARGET_SESSION:
        sensor_fusion(TARGET_SESSION)