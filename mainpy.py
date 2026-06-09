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
            await imu.start()
            print("Kết nối IMU Arduino thành công.")
        except Exception as e:
            print(f"Thất bại khi kết nối Arduino: {e}")
            print("Hệ thống tự động chuyển về chế độ CAMERA ONLY.")
            use_imu = False

    # Tạo thư mục gốc lưu dữ liệu
    recording_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== HỆ THỐNG SẴN SÀNG (Gõ: start / stop / exit) ===")

    try:
        while True:
            cmd = await asyncio.get_event_loop().run_in_executor(None, input)
            cmd = cmd.strip().lower()

            # ========================================================
            # LỆNH: START RECORDING
            # ========================================================
            if cmd == "start":
                if recording:
                    print("Already recording")
                    continue

                # BƯỚC 2: HỎI TÊN VĐV ĐỂ ĐẶT TÊN THƯ MỤC THÔNG MINH
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

                # Tạo thư mục phiên có tên VĐV cấu trúc cũ của bạn
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_dir = recording_dir / f"{vđv_name}_{time_str}"
                session_dir.mkdir(parents=True, exist_ok=True)
                shared_data.session_dir = session_dir

                # Tự động tạo file .bat xem nhanh đồ thị 3D sau khi thu dữ liệu
                bat_file_path = session_dir / "Xem_Do_Thi_3D.bat"
                bat_content = f"""@echo off
echo =======================================================
echo   DANG KHOI DONG DO THI 3D TUONG TAC CHO SESSION NAY   
echo =======================================================
cd /d "%~dp0..\\.."
uv run python dataprocessing.py "recording/{vđv_name}_{time_str}"
"""
                with open(bat_file_path, "w", encoding="utf-8") as f:
                    f.write(bat_content)

                print(f"SESSION DIR: {session_dir}")

                # Kích hoạt bộ ghi cho từng thiết bị được chọn
                if use_imu and imu:
                    imu.open_session(session_dir)
                    # Giữ nguyên cấu trúc check writer của bạn
                    if hasattr(imu, 'writer'): print("imu writer:", imu.writer)
                    await imu.send_start()

                if use_camera and cam:
                    cam.open_session(session_dir)
                    if hasattr(cam, 'writer'): print("cam writer:", cam.writer)
                    cam.start()

                print("[RECORDING STARTED]")

            # ========================================================
            # LỆNH: STOP RECORDING
            # ========================================================
            elif cmd == "stop":
                if not recording:
                    print("Not recording")
                    continue

                recording = False
                print("[STOPPING SYSTEM]")

                shared_data.graph_running = False

                if use_imu and imu:
                    await imu.send_stop()
                    imu.close_session()

                if use_camera and cam:
                    cam.stop()
                    cam.close_session()  # Kích hoạt tiến trình quét AI hậu kỳ tự động

                print("=== DATA SAVED & POST-PROCESSING COMPLETED ===")

            # ========================================================
            # LỆNH: EXIT
            # ========================================================
            elif cmd == "exit":
                break

    except (KeyboardInterrupt, EOFError):
        pass

    finally:
        try:
            # TỰ ĐỘNG DỪNG NẾU ĐANG GHI MÀ THOÁT ĐỘT NGỘT
            if recording:
                print("Auto stopping recording...")
                recording = False
                shared_data.graph_running = False

                if use_imu and imu:
                    await imu.send_stop()
                    imu.close_session()

                if use_camera and cam:
                    cam.stop()
                    cam.close_session()

            # SHUTDOWN AN TOÀN
            await asyncio.wait_for(
                shutdown(use_imu, imu),
                timeout=5
            )

        except Exception as e:
            print("Shutdown error:", e)

        finally:
            shared_data.program_running = False

            if use_camera and cam:
                cam.shutdown()

            if use_camera and cam_thread:
                cam_thread.join(timeout=3)

            print("Ended")

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