import cv2
import mediapipe as mp
import csv
import time
import threading
from queue import Queue

from shareddata import shared_data


class CameraClient:

    def __init__(self):
        camera_index = 1 
        print(f"Connecting to device camera (Index: {camera_index})...")
        
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        if not self.cap.isOpened():
            print(f"Unable to connect to device camera (index {camera_index}).")
            print("Automatically falling back to using the default system webcam (index 0)")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

        self._lock = threading.Lock()

        self.file = None
        self.writer = None
        self.video_writer = None
        
        # TỐI ƯU AI CHO CHẠY REALTIME 60 FPS
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,       # Bản Lite siêu tốc độ để đỡ nghẽn máy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.frame = 0
        self.collect_event = threading.Event()
        self.stop_event = threading.Event()

        # Hàng đợi chứa ảnh để AI xử lý ngầm
        self.processing_queue = Queue(maxsize=128)
        
        # Khởi chạy luồng xử lý AI ngầm
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

        self.TARGET_FPS = 60.0

    # =======================
    # SESSION CONTROL
    # =======================
    def open_session(self, session_dir):
        # Làm sạch hàng đợi trước khi bấm record mới
        with self.processing_queue.mutex:
            self.processing_queue.queue.clear()

        with self._lock:
            if self.file:
                self.file.close()

            self.file = open(session_dir / "camera_data.csv", "w", newline="")
            self.writer = csv.writer(self.file)

            # Khởi tạo Header CSV
            self.writer.writerow([
                "frame", "timestamp",
                "shoulder_x", "shoulder_y", "shoulder_z", "shoulder_visibility",
                "elbow_x", "elbow_y", "elbow_z", "elbow_visibility",
                "wrist_x", "wrist_y", "wrist_z", "wrist_visibility"
            ])

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width == 0 or actual_height == 0:
                actual_width, actual_height = 1280, 720

            print(f"Khởi tạo Video Writer tại Luồng Chính: {actual_width}x{actual_height} @ {self.TARGET_FPS} FPS")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_path = str(session_dir / "raw_video.mp4")
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.TARGET_FPS, (actual_width, actual_height))

            self.frame = 0

    def close_session(self):
        print("Đang đợi luồng ngầm giải quyết các khung hình còn lại...")
        while not self.processing_queue.empty():
            time.sleep(0.05)

        with self._lock:
            if self.file:
                self.file.flush()
                self.file.close()
                self.file = None
                self.writer = None
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        print("Đã đóng và lưu file Video + CSV.")

    def start(self):
        self.collect_event.set()

    def stop(self):
        self.collect_event.clear()
        self.close_session()

    def shutdown(self):
        self.collect_event.clear()
        self.stop_event.set()

    # =====================================================================
    # LUỒNG AI NGẦM (BÂY GIỜ CHỈ CHẠY AI, KHÔNG ÔM ĐỒM GHI VIDEO NỮA)
    # =====================================================================
    def _processing_worker(self):
        while not self.stop_event.is_set():
            if self.processing_queue.empty():
                time.sleep(0.005)
                continue
                
            try:
                frame, now, current_frame_idx = self.processing_queue.get(timeout=0.1)
            except:
                continue
            
            # Chạy AI trích xuất tọa độ khớp
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                s = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                e = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                w = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                with self._lock:
                    if self.writer is not None:
                        self.writer.writerow([
                            current_frame_idx,
                            now,
                            s.x, s.y, s.z, s.visibility,
                            e.x, e.y, e.z, e.visibility,
                            w.x, w.y, w.z, w.visibility
                        ])

                        shared_data.camera_buffer.append({
                            "timestamp": now,
                            "shoulder": (s.x, s.y, s.z),
                            "elbow": (e.x, e.y, e.z),
                            "wrist": (w.x, w.y, w.z)
                        })
            
            self.processing_queue.task_done()

    # =====================================================================
    # LUỒNG CHÍNH ĐỌC FRAME (GHI VIDEO TRỰC TIẾP TẠI ĐÂY - CHỐNG TUA NHANH)
    # =====================================================================
    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            if self.collect_event.is_set():
                now = time.perf_counter()

                # 🔴 THAY ĐỔI CỐT LÕI: Ghi thẳng vào file .mp4 ngay tại luồng chính (Siêu tốc < 1ms)
                with self._lock:
                    if self.video_writer is not None:
                        self.video_writer.write(frame)

                # Ném ảnh copy vào hàng đợi cho AI xử lý ngầm nếu hàng đợi còn chỗ
                # Nếu AI quá tải, nó sẽ tự drop frame của AI, nhưng file VIDEO gốc vẫn hoàn hảo 60 FPS
                if not self.processing_queue.full():
                    self.processing_queue.put((frame.copy(), now, self.frame))
                
                # Tăng số thứ tự khung hình (Đảm bảo khớp chuẩn 1-1 với file video .mp4)
                self.frame += 1
        
            cv2.imshow("Camera 60 FPS Preview", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()

        self.cap.release()
        self.close_session()
        cv2.destroyAllWindows()