import cv2
import numpy as np
import mediapipe as mp
import csv
import time
import threading
from pathlib import Path


class CameraClient:
    def __init__(self):
        # CHANGE HERE: set to 1 to use the Iriun Webcam instead of 0
        camera_index = 1
        print(f"Connecting to Iriun Camera (Index: {camera_index})...")

        self.cap = cv2.VideoCapture(camera_index)

        # High-quality configuration for the Iriun camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # Fallback: if camera 1 fails to open, fall back to the machine's camera 0
        if not self.cap.isOpened():
            print("⚠️ Could not open Iriun Camera (Index 1). Falling back to default webcam (Index 0)")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.video_writer = None
        self.ts_file = None
        self.ts_writer = None
        self.current_session_dir = None

        self.frame_idx = 0
        self.collect_event = threading.Event()
        self.stop_event = threading.Event()
        self.TARGET_FPS = 60.0

    @property
    def writer(self):
        """Kept in sync with the print(cam.writer) call in mainpy.py."""
        return self.video_writer

    def open_session(self, session_dir):
        self.current_session_dir = Path(session_dir)
        self.frame_idx = 0

        self.ts_file = open(self.current_session_dir / "temp_timestamps.csv", "w", newline="")
        self.ts_writer = csv.writer(self.ts_file)
        self.ts_writer.writerow(["frame", "timestamp"])

        ret, dummy_frame = self.cap.read()
        if ret:
            actual_height, actual_width = dummy_frame.shape[:2]
        else:
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width == 0: actual_width, actual_height = 1280, 720

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = str(self.current_session_dir / "raw_video.mp4")
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.TARGET_FPS, (actual_width, actual_height))

        print(f"📹 [Camera] Raw recording file opened successfully ({actual_width}x{actual_height}).")

    def close_session(self):
        """Close the recording session and trigger standalone offline AI post-processing."""
        if self.ts_file:
            self.ts_file.flush()
            self.ts_file.close()
            self.ts_file = None
            self.ts_writer = None
            
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.current_session_dir and (self.current_session_dir / "raw_video.mp4").exists():
            self._run_offline_post_processing(self.current_session_dir)
        self.current_session_dir = None

    def _run_offline_post_processing(self, session_dir):
        video_path       = session_dir / "raw_video.mp4"
        ts_path          = session_dir / "temp_timestamps.csv"
        output_csv_path  = session_dir / "camera_data.csv"
        annotated_path   = session_dir / "annotated_video.mp4"

        if not video_path.exists() or not ts_path.exists() or video_path.stat().st_size == 0:
            print("⚠️ Raw video data is empty or missing. Skipping AI post-processing.")
            return

        print("\n" + "="*60)
        print(" RUNNING AUTOMATIC AI POST-PROCESSING (OFFLINE) ")
        print("="*60)

        timestamp_map = {}
        with open(ts_path, "r") as tf:
            reader = csv.reader(tf)
            next(reader)
            for row in reader:
                if len(row) == 2:
                    timestamp_map[int(row[0])] = float(row[1])

        mp_pose    = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose_processor = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap_video    = cv2.VideoCapture(str(video_path))
        total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w        = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h        = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # TRUE FPS for annotated_video, computed from the real capture timestamps.
        # Do NOT use TARGET_FPS (60): USB/Iriun cameras rarely reach a full 60fps,
        # so declaring 60 makes the video play faster than real life and drift from
        # the graph. Compute from frame-index span / time span (robust to dropped frames).
        actual_fps = 30.0   # fallback when it can't be computed (do NOT fall back to 60)
        if len(timestamp_map) > 1:
            _fidx = sorted(timestamp_map)
            _span = timestamp_map[_fidx[-1]] - timestamp_map[_fidx[0]]
            if _span > 0:
                _r = (_fidx[-1] - _fidx[0]) / _span
                if 0 < _r <= 240:
                    actual_fps = _r
        print(f" annotated_video true FPS: {actual_fps:.1f}")

        # VideoWriter for the annotated output (mp4v = H.264-compatible)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(annotated_path), fourcc, actual_fps, (vid_w, vid_h))

        # Highlight color for the tracked arm skeleton
        TRACKED_COLOR = (0, 229, 255)   # cyan BGR
        TRACKED_THICK = 4

        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "timestamp",
                "shoulder_x", "shoulder_y", "shoulder_z", "shoulder_visibility",
                "elbow_x", "elbow_y", "elbow_z", "elbow_visibility",
                "wrist_x", "wrist_y", "wrist_z", "wrist_visibility"
            ])

            f_idx = 0
            while True:
                ret, frame = cap_video.read()
                if not ret:
                    break

                true_timestamp = timestamp_map.get(f_idx, time.perf_counter())
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = pose_processor.process(image_rgb)
                annotated = frame.copy()

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    s  = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    e  = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    w  = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                    # CSV
                    writer.writerow([
                        f_idx, true_timestamp,
                        s.x, s.y, s.z, s.visibility,
                        e.x, e.y, e.z, e.visibility,
                        w.x, w.y, w.z, w.visibility
                    ])

                    # ── SKELETON OVERLAY ──────────────────────────────
                    # Full body dim
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(60, 60, 60), thickness=1, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(60, 60, 60), thickness=1),
                    )

                    # Highlight tracked arm
                    sx, sy = int(s.x * vid_w), int(s.y * vid_h)
                    ex, ey = int(e.x * vid_w), int(e.y * vid_h)
                    wx, wy = int(w.x * vid_w), int(w.y * vid_h)

                    cv2.line(annotated, (sx, sy), (ex, ey), TRACKED_COLOR, TRACKED_THICK)
                    cv2.line(annotated, (ex, ey), (wx, wy), TRACKED_COLOR, TRACKED_THICK)
                    for (px, py), r in [((sx, sy), 8), ((ex, ey), 6), ((wx, wy), 8)]:
                        cv2.circle(annotated, (px, py), r + 2, (0, 0, 0), -1)
                        cv2.circle(annotated, (px, py), r,     TRACKED_COLOR, -1)

                    # Elbow angle label (inlined to avoid a circular import with dataprocessing)
                    def _calc_angle(a, b, c):
                        v1 = np.array(b) - np.array(a)
                        v2 = np.array(c) - np.array(b)
                        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                        return 180 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                    ang = _calc_angle(
                        [s.x, s.y, s.z], [e.x, e.y, e.z], [w.x, w.y, w.z]
                    )
                    label = f"{ang:.1f} deg"
                    cv2.putText(annotated, label, (ex + 13, ey - 11),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(annotated, label, (ex + 12, ey - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TRACKED_COLOR, 2)

                    # Frame watermark
                    info = f"F:{f_idx}"
                    cv2.putText(annotated, info, (11, 29),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                    cv2.putText(annotated, info, (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                vw.write(annotated)
                f_idx += 1
                if f_idx % 30 == 0 or f_idx == total_frames:
                    print(f" Progress AI: {f_idx}/{total_frames} "
                          f"({(f_idx/total_frames)*100:.1f}%)")

        cap_video.release()
        pose_processor.close()
        vw.release()
        ts_path.unlink(missing_ok=True)
        print(" EXPORTED: camera_data.csv + annotated_video.mp4\n")

    def start(self):
        self.collect_event.set()

    def stop(self):
        """Only lower the collection flag; let mainpy orchestrate the file finalization."""
        self.collect_event.clear()

    def shutdown(self):
        self.collect_event.clear()
        self.stop_event.set()

    def run(self):
        self.stop_event.clear()
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            if self.collect_event.is_set():
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                if self.ts_writer is not None:
                    self.ts_writer.writerow([self.frame_idx, time.perf_counter()])
                self.frame_idx += 1

            cv2.imshow("Camera 60 FPS (Press 'q' to Stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()

        self.cap.release()
        cv2.destroyAllWindows()