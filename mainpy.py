import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

import sys
import asyncio
import signal
import threading

if sys.platform == "win32":
    os.environ["PYTHONNET_INITIALIZE"] = "0"
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from datetime import datetime
from pathlib import Path

from imuclient import IMUClient
from cameraclient import CameraClient
from shareddata import shared_data

# =========================
# RECORDING CONFIG
# =========================
recording_dir = Path("recording")
recording = False

# =========================
# HARD KILL
# =========================
signal.signal(signal.SIGINT, lambda *_: os._exit(1))

# =========================
# SHUTDOWN
# =========================
async def shutdown(use_imu, imu_obj):
    print("\nShutting down...")
    if use_imu and imu_obj:
        try:
            await imu_obj.send_stop()
        except Exception:
            pass

        try:
            await imu_obj.stop()
        except Exception:
            pass

# =========================
# MAIN ASYNC LOOP
# =========================

# =====================================================================
# HELPER: Tạo shortcut .bat để double-click mở viewer
# =====================================================================
def _write_viewer_bat(session_dir):
    """Tạo file Open_Viewer.bat trong session folder.
    Double-click là mở viewer, không cần gõ lệnh terminal.
    """
    from pathlib import Path
    session_dir = Path(session_dir)
    # Tính đường dẫn tương đối từ session_dir lên root project
    # session_dir = recording/TenVDV_20260612.../  → cần cd lên 2 cấp
    bat_content = (
        "@echo off\n"
        "cd /d \"%~dp0..\\..\"\n"
        f'uv run python dataprocessing.py "recording\\{session_dir.name}"\n'
        "pause\n"
    )
    bat_path = session_dir / "Open_Viewer.bat"
    bat_path.write_text(bat_content, encoding="utf-8")

async def main():
    global recording

    # ========================================================
    #  BƯỚC 1: MENU CHỌN CHẾ ĐỘ CHẠY TRUYỀN THỐNG
    # ========================================================
    print("\n" + "="*40)
    print(" KHỞI ĐỘNG HỆ THỐNG THU THẬP DỮ LIỆU")
    print("="*40)
    print("1. Chỉ quay CAMERA")
    print("2. Chỉ lưu CẢM BIẾN IMU")
    print("3. Chạy CẢ HAI thiết bị đồng thời")
    print("="*40)

    def get_mode_input():
        return input("Nhập lựa chọn của bạn (1/2/3): ").strip()

    mode_choice = await asyncio.get_event_loop().run_in_executor(None, get_mode_input)
    
    use_camera = mode_choice in ["1", "3"]
    use_imu = mode_choice in ["2", "3"]

    print(f"\n➔ Trạng thái kích hoạt: CAMERA={use_camera} | IMU={use_imu}\n")

    # Khởi tạo các đối tượng dựa trên lựa chọn
    imu = None
    cam = None
    cam_thread = None

    if use_camera:
        cam = CameraClient()
        # START CAMERA THREAD (Đảm bảo chạy ngầm bắt khung hình 60 FPS)
        cam_thread = threading.Thread(target=cam.run, daemon=True)
        cam_thread.start()
        print("Luồng Camera 60 FPS đã sẵn sàng.")

    if use_imu:
        try:
            imu = IMUClient()
            await imu.start_client() # CHỈ KẾT NỐI BLUETOOTH 1 LẦN DUY NHẤT Ở ĐÂY
            print("Kết nối IMU Arduino thành công và đang giữ liên lạc.")
        except Exception as e:
            print(f"Thất bại khi kết nối Arduino: {e}")
            print("Hệ thống tự động chuyển về chế độ CAMERA ONLY.")
            use_imu = False

    # Tạo thư mục gốc lưu dữ liệu
    recording_dir.mkdir(parents=True, exist_ok=True)

    try:
        # VÒNG LẶP CHÍNH: TRẠNG THÁI CHỜ QUAY HOẶC THOÁT
        while True:
            print("\n=== HỆ THỐNG SẴN SÀNG (Gõ: start / exit) ===")
            cmd = await asyncio.get_event_loop().run_in_executor(None, input)
            cmd = cmd.strip().lower()

            # ========================================================
            # LỆNH: EXIT (CHỈ THOÁT KHI ĐANG RẢNH)
            # ========================================================
            if cmd == "exit":
                print("Đang chuẩn bị đóng toàn bộ hệ thống...")
                break

            # ========================================================
            # LỆNH: START RECORDING
            # ========================================================
            elif cmd == "start":
                # Hỏi tên VĐV để đặt tên thư mục thông minh
                def get_player_name():
                    return input("Nhập Tên hoặc Mã số VĐV (Bấm Enter để bỏ qua): ").strip()
                
                vđv_name = await asyncio.get_event_loop().run_in_executor(None, get_player_name)
                if not vđv_name:
                    vđv_name = "Player"

                recording = True
                shared_data.graph_running = True

                # Xóa dữ liệu cũ trên bộ đệm biểu đồ
                shared_data.camera_buffer.clear()
                shared_data.imu_buffer.clear()

                # Tạo thư mục phiên có tên VĐV
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_dir = recording_dir / f"{vđv_name}_{time_str}"
                session_dir.mkdir(parents=True, exist_ok=True)
                shared_data.session_dir = session_dir

                # (HTML report sẽ được tạo tự động sau khi stop, bởi dataprocessing.py)

                print(f"SESSION DIR: {session_dir}")

                # Kích hoạt bộ ghi cho từng thiết bị
                if use_imu and imu:
                    imu.open_session(session_dir)
                    await imu.start_session() # Gửi lệnh START qua BLE công suất thực tế

                if use_camera and cam:
                    cam.open_session(session_dir)
                    cam.start()

                print("\n>>> ĐANG QUAY DATA (Gõ lệnh 'stop' để kết thúc) <<<")

                # ----------------------------------------------------
                # VÒNG LẶP LỒNG: BẮT BUỘC ĐỢI LỆNH STOP KHI ĐANG QUAY
                # ----------------------------------------------------
                while True:
                    sub_cmd = await asyncio.get_event_loop().run_in_executor(None, input)
                    sub_cmd = sub_cmd.strip().lower()

                    if sub_cmd == "stop":
                        recording = False
                        shared_data.graph_running = False
                        print("\n[STOPPING SYSTEM] - Đang dừng thiết bị và kết xuất dữ liệu...")

                        if use_imu and imu:
                            await imu.stop_session() # Dừng luồng và ghi RAM -> CSV (Không ngắt BLE)

                        if use_camera and cam:
                            cam.stop()
                            cam.close_session() # Kích hoạt hậu kỳ AI cho Camera

                        # Xuất HTML report (không mở browser vì đang ở CLI loop)
                        if use_camera and (session_dir / "camera_data.csv").exists():
                            print("  → Đang xuất motion_report.html ...")
                            try:
                                import dataprocessing
                                dataprocessing.run(str(session_dir), open_browser=False)
                                _write_viewer_bat(session_dir)
                                report_path = session_dir / "motion_report.html"
                                print(f"  → Xem report: uv run python dataprocessing.py {session_dir}")
                            except Exception as e:
                                print(f"  ⚠️  Không tạo được report: {e}")

                        print("=== 100% DATA SAVED & POST-PROCESSING COMPLETED ===")
                        print("Hệ thống tự động đưa bạn quay lại Menu chính.")
                        break # Bẻ gãy vòng lặp phụ, quay về Menu chính chờ lệnh start/exit tiếp theo
                    
                    elif sub_cmd == "exit":
                        print(" CẢNH BÁO: Hệ thống đang ghi dữ liệu, gõ 'stop' trước khi muốn thoát.")
                    else:
                        print(" Lệnh không hợp lệ, chỉ nhận lệnh 'stop'.")

            else:
                print(" Lệnh không hợp lệ, chỉ nhận lệnh 'start' hoặc 'exit'.")

    except (KeyboardInterrupt, EOFError):
        pass

    finally:
        # ========================================================
        # DỌN DẸP VÀ NGẮT KẾT NỐI HOÀN TOÀN KHI THOÁT APP
        # ========================================================
        try:
            if recording:
                print("Auto stopping recording due to abrupt exit...")
                recording = False
                shared_data.graph_running = False

                if use_imu and imu:
                    await imu.stop_session()

                if use_camera and cam:
                    cam.stop()
                    cam.close_session()

            # ĐÓNG KẾT NỐI BLUETOOTH AN TOÀN TRƯỚC KHI KHAI TỬ THREAD
            if use_imu and imu:
                print("Đang ngắt kết nối Bluetooth với Arduino...")
                await imu.exit_client()

        except Exception as e:
            print("Shutdown error:", e)

        finally:
            shared_data.program_running = False

            if use_camera and cam:
                cam.shutdown()

            if use_camera and cam_thread:
                cam_thread.join(timeout=3)

            print("=== ĐÃ THOÁT CHƯƠNG TRÌNH AN TOÀN ===")

# =====================================================================
# KHỞI CHẠY TRỰC TIẾP TRÊN LUỒNG CHÍNH (MAIN THREAD)
# =====================================================================
if __name__ == "__main__":
    shared_data.program_running = True
    try:
        # Chạy trực tiếp main() trên Luồng chính để lắng nghe tín hiệu CLI và giữ ứng dụng không bị tắt
        asyncio.run(main())
    except Exception:
        import traceback
        traceback.print_exc()