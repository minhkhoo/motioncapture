import cv2
import mediapipe as mp
import csv
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime


# =====================================================================
# PIPELINE MediPipe (giống _run_offline_post_processing trong cameraclient.py)
# =====================================================================
def extract_pose_from_video(
    video_path: Path,
    output_csv: Path,
    side: str = "RIGHT",
    model_complexity: int = 1,
):
    """
    Chạy MediaPipe Pose trên video, xuất camera_data.csv.

    Args:
        video_path:       Đường dẫn file video (.mp4, .avi, .mov, ...)
        output_csv:       Đường dẫn xuất camera_data.csv
        side:             "RIGHT" hoặc "LEFT" – tay nào cần phân tích
        model_complexity: 0 (nhanh) / 1 (cân bằng) / 2 (chính xác nhất)
    """
    mp_pose = mp.solutions.pose

    # Map side string → landmark enum
    side = side.upper()
    if side == "RIGHT":
        SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
        ELBOW    = mp_pose.PoseLandmark.RIGHT_ELBOW
        WRIST    = mp_pose.PoseLandmark.RIGHT_WRIST
    elif side == "LEFT":
        SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
        ELBOW    = mp_pose.PoseLandmark.LEFT_ELBOW
        WRIST    = mp_pose.PoseLandmark.LEFT_WRIST
    else:
        print(f"[LỖI] --side phải là RIGHT hoặc LEFT, nhận được: {side}")
        return False

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[LỖI] Không mở được video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  → Video: {width}×{height} @ {src_fps:.1f} FPS · {total_frames} frames")
    print(f"  → Phân tích tay: {side}")
    print(f"  → Model complexity: {model_complexity}")

    # Tạo synthetic timestamps từ FPS của video gốc
    # (Không có đồng hồ thực tế như khi quay bằng hệ thống, nhưng relative timing chính xác)
    start_ts = time.perf_counter()

    rows_written = 0
    f_idx        = 0
    last_report  = -1

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

            # Synthetic timestamp: frame position / fps
            synthetic_ts = start_ts + (f_idx / src_fps)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                s  = lm[SHOULDER]
                e  = lm[ELBOW]
                w  = lm[WRIST]

                writer.writerow([
                    f_idx, synthetic_ts,
                    s.x, s.y, s.z, s.visibility,
                    e.x, e.y, e.z, e.visibility,
                    w.x, w.y, w.z, w.visibility,
                ])
                rows_written += 1

            f_idx += 1

            pct = int((f_idx / total_frames) * 100) if total_frames > 0 else 0
            if pct != last_report and pct % 5 == 0:
                last_report = pct
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r  → [{bar}] {pct:3d}%  ({f_idx}/{total_frames} frames, {rows_written} poses)", end="", flush=True)

    print()  # newline sau progress bar
    cap.release()
    pose.close()

    if rows_written == 0:
        print("  ⚠️  MediaPipe không phát hiện được tư thế nào trong video.")
        print("     Thử dùng --complexity 2 hoặc kiểm tra góc quay / ánh sáng.")
        return False

    detection_rate = rows_written / f_idx * 100 if f_idx > 0 else 0
    print(f"  ✅ Đã ghi {rows_written} dòng data ({detection_rate:.1f}% detection rate)")
    return True


# =====================================================================
# COPY VIDEO VÀO SESSION FOLDER (để HTML report embed được)
# =====================================================================
def copy_video(src: Path, dst: Path):
    import shutil
    print(f"  → Copy video vào session folder ...")
    shutil.copy2(src, dst)


# =====================================================================
# ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Xử lý video hậu kỳ và tạo motion_report.html",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  uv run python process_video.py spike_clip.mp4
  uv run python process_video.py spike_clip.mp4 --side LEFT
  uv run python process_video.py spike_clip.mp4 --out recording/tuan_spike1 --complexity 2
        """
    )
    parser.add_argument("video",       type=str, help="Đường dẫn file video đầu vào")
    parser.add_argument("--side",      type=str, default="RIGHT", choices=["RIGHT","LEFT"],
                        help="Tay cần phân tích (mặc định: RIGHT)")
    parser.add_argument("--out",       type=str, default=None,
                        help="Thư mục session output (mặc định: recording/<tên_file>_<timestamp>)")
    parser.add_argument("--complexity", type=int, default=1, choices=[0,1,2],
                        help="MediaPipe model complexity: 0=nhanh, 1=cân bằng, 2=chính xác (mặc định: 1)")
    parser.add_argument("--no-report", action="store_true",
                        help="Chỉ trích xuất CSV, không tạo HTML report")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[LỖI] Không tìm thấy file video: {video_path}")
        sys.exit(1)

    # Xác định session folder
    if args.out:
        session_dir = Path(args.out)
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem     = video_path.stem.replace(" ", "_")
        session_dir = Path("recording") / f"{stem}_{ts}"

    session_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*60)
    print(f"  XỬ LÝ VIDEO HẬU KỲ")
    print("="*60)
    print(f"  Video input : {video_path.resolve()}")
    print(f"  Session dir : {session_dir.resolve()}")

    # Bước 1: Trích xuất pose data
    csv_path = session_dir / "camera_data.csv"
    ok = extract_pose_from_video(video_path, csv_path, side=args.side, model_complexity=args.complexity)
    if not ok:
        sys.exit(1)

    # Bước 2: Copy video vào session folder
    dst_video = session_dir / "raw_video.mp4"
    copy_video(video_path, dst_video)

    # Bước 3: Tạo HTML report (trừ khi --no-report)
    if not args.no_report:
        print("\n  → Tạo HTML report ...")
        from dataprocessing import run as dp_run
        dp_run(session_dir)
    else:
        print(f"\n  ✅ CSV đã xuất tại: {csv_path}")
        print("  (Dùng 'uv run python dataprocessing.py <session_dir>' để tạo report sau)")

    print(f"\n  📁 Session folder: {session_dir.resolve()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
