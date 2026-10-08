import cv2
import numpy as np
import mediapipe as mp
import csv
import time
import threading
import shutil
import pandas as pd
import argparse
from pathlib import Path


# Camera index of the phone webcam app (Iriun); falls back to index 0 (default webcam).
CAMERA_INDEX = 1


class CameraClient:
    def __init__(self):
        camera_index = CAMERA_INDEX
        print(f"Connecting to Iriun Camera (Index: {camera_index})...")

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

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

        # Liveness: perf_counter of the last frame actually written while recording
        # (updated on the camera thread, read by the mid-session watchdog on the main
        # thread). None = no frame written since the current session started.
        self.last_frame_perf = None

    @property
    def writer(self):
        return self.video_writer

    def open_session(self, session_dir):
        self.current_session_dir = Path(session_dir)
        self.frame_idx = 0
        # Reset liveness so the watchdog measures silence from the start of THIS session
        self.last_frame_perf = None

        # FIX 1: named frame_timestamps.csv and NOT deleted at the end
        self.ts_file = open(self.current_session_dir / "frame_timestamps.csv", "w", newline="")
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

    @staticmethod
    def _run_offline_post_processing(session_dir):
        # staticmethod: --reprocess mode does NOT need to open the camera (avoids hangs/errors
        # on a machine without Iriun). Still callable via self in close_session.
        session_dir      = Path(session_dir)
        video_path       = session_dir / "raw_video.mp4"
        ts_path          = session_dir / "frame_timestamps.csv"
        output_csv_path  = session_dir / "camera_data.csv"
        annotated_path   = session_dir / "annotated_video.mp4"

        if not video_path.exists() or video_path.stat().st_size == 0:
            print("⚠️ Raw video data is empty or missing. Skipping AI post-processing.")
            return

        print("\n" + "="*60)
        print(" RUNNING AUTOMATIC AI POST-PROCESSING (OFFLINE) ")
        print("="*60)

        cap_video    = cv2.VideoCapture(str(video_path))
        total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w        = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h        = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FIX 3 & 4(e): safe timestamp handling and Reprocess support
        #
        # Reprocess = camera_data.csv already existed. Back it up to .bak ONLY ONCE
        # (never overwrite an existing .bak) so the original is not lost on a second reprocess.
        is_reprocess = output_csv_path.exists()
        if is_reprocess:
            bak_path = session_dir / "camera_data.csv.bak"
            if not bak_path.exists():
                shutil.copy2(output_csv_path, bak_path)
                print(f"💾 Backed up camera_data.csv -> {bak_path.name} (original kept).")
            else:
                print(f"💾 {bak_path.name} already exists — original kept, not overwritten.")

        timestamp_map = {}
        if ts_path.exists():
            # Best source: frame_timestamps.csv (real perf_counter values from recording)
            df_ts = pd.read_csv(ts_path)
            timestamp_map = dict(zip(df_ts['frame'], df_ts['timestamp']))
        elif is_reprocess:
            # OLD session: frame_timestamps.csv was deleted -> recover from the original camera_data.csv
            # (read from the .bak just made, to be sure it is the original, not overwritten).
            print("🔄 Reprocess: recovering timestamps from the original camera_data.csv (interpolation)...")
            df_old = pd.read_csv(session_dir / "camera_data.csv.bak")
            if {'frame', 'timestamp'}.issubset(df_old.columns):
                df_old_valid = df_old.dropna(subset=['frame', 'timestamp'])
                if not df_old_valid.empty:
                    f_known = df_old_valid['frame'].values.astype(float)
                    t_known = df_old_valid['timestamp'].values.astype(float)
                    # np.interp requires increasing xp -> sort first to be safe
                    order = np.argsort(f_known)
                    f_known, t_known = f_known[order], t_known[order]
                    f_all = np.arange(total_frames)
                    # Linear interpolation BETWEEN known points; OUTSIDE the range => NaN (no extrapolation)
                    t_interp = np.interp(f_all, f_known, t_known, left=np.nan, right=np.nan)
                    timestamp_map = dict(zip(f_all, t_interp))
                else:
                    print("⚠️ WARNING: the original camera_data.csv has no valid timestamps to interpolate!")
            else:
                print("⚠️ WARNING: the original camera_data.csv is missing the 'frame'/'timestamp' columns!")
        else:
            print("⚠️ WARNING: no timestamp file found to map!")

        actual_fps = 30.0
        if len(timestamp_map) > 1:
            valid_ts = [t for t in timestamp_map.values() if not np.isnan(t)]
            if len(valid_ts) > 1:
                _span = valid_ts[-1] - valid_ts[0]
                if _span > 0:
                    _r = len(valid_ts) / _span
                    if 0 < _r <= 240:
                        actual_fps = _r

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(annotated_path), fourcc, actual_fps, (vid_w, vid_h))

        TRACKED_COLOR = (0, 229, 255)
        TRACKED_THICK = 4

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # OPTIMISATION (1): tiered model complexity.
        # Pass 1 runs on EVERY frame -> model_complexity=1 (fast, catches most easy
        # frames). Hard frames that fail pass 1 are handled by Pass 2 with the heaviest model.
        pose_processor_pass1 = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # FIX 4(b): recovery pass (Pass 2) configuration — most accurate model for hard frames
        pose_processor_pass2 = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # OPTIMISATION (3): if the athlete has left the frame, pass 2b (CLAHE) cannot rescue it
        # and only wastes time. After OUT_OF_FRAME_STREAK CONSECUTIVE frames fail both
        # pass 1 and pass 2a, assume out of frame -> skip pass 2b until detection resumes.
        OUT_OF_FRAME_STREAK = 5
        consecutive_fail = 0
        stats_clahe_skipped = 0

        # Statistics + timing
        stats = {"pass1": 0, "pass2": 0, "fail": 0, "missing_ts": 0}
        _t_proc_start = time.perf_counter()

        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # FIX 4(c): add the detection_pass column
            writer.writerow([
                "frame", "timestamp",
                "shoulder_x", "shoulder_y", "shoulder_z", "shoulder_visibility",
                "elbow_x", "elbow_y", "elbow_z", "elbow_visibility",
                "wrist_x", "wrist_y", "wrist_z", "wrist_visibility",
                "detection_pass"
            ])

            f_idx = 0
            while True:
                ret, frame = cap_video.read()
                if not ret:
                    break

                # FIX 3: use the correct timestamp, record if missing
                true_timestamp = timestamp_map.get(f_idx, np.nan)
                if np.isnan(true_timestamp):
                    stats["missing_ts"] += 1

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated = frame.copy()
                
                # ------ START DETECTION ------
                results = pose_processor_pass1.process(image_rgb)
                pass_id = 0

                if results.pose_landmarks:
                    pass_id = 1
                else:
                    # Pass 2a: try static mode (model_complexity=2)
                    results = pose_processor_pass2.process(image_rgb)
                    if results.pose_landmarks:
                        pass_id = 2
                    elif consecutive_fail < OUT_OF_FRAME_STREAK:
                        # Pass 2b: CLAHE brightening of the L channel — ONLY when not yet suspected out of frame.
                        # (pass 1 + pass 2a still run on every frame, so a return is still caught.)
                        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        l_clahe = clahe.apply(l)
                        lab_clahe = cv2.merge((l_clahe, a, b))
                        frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
                        rgb_clahe = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2RGB)

                        results = pose_processor_pass2.process(rgb_clahe)
                        if results.pose_landmarks:
                            pass_id = 2
                    else:
                        # In an out-of-frame streak -> skip pass 2b for this frame
                        stats_clahe_skipped += 1

                # (3): update the consecutive-fail streak (used to decide whether to skip pass 2b)
                if pass_id == 0:
                    consecutive_fail += 1
                else:
                    consecutive_fail = 0

                # ------ SAVE CSV & DRAW OVERLAY ------
                if pass_id > 0:
                    if pass_id == 1: stats["pass1"] += 1
                    if pass_id == 2: stats["pass2"] += 1
                    
                    lm = results.pose_landmarks.landmark
                    s = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    e = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    w = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                    # Write a valid row
                    writer.writerow([
                        f_idx, true_timestamp,
                        s.x, s.y, s.z, s.visibility,
                        e.x, e.y, e.z, e.visibility,
                        w.x, w.y, w.z, w.visibility,
                        pass_id
                    ])

                    mp_drawing.draw_landmarks(
                        annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(60, 60, 60), thickness=1),
                    )

                    sx, sy = int(s.x * vid_w), int(s.y * vid_h)
                    ex, ey = int(e.x * vid_w), int(e.y * vid_h)
                    wx, wy = int(w.x * vid_w), int(w.y * vid_h)

                    cv2.line(annotated, (sx, sy), (ex, ey), TRACKED_COLOR, TRACKED_THICK)
                    cv2.line(annotated, (ex, ey), (wx, wy), TRACKED_COLOR, TRACKED_THICK)
                    for (px, py), r in [((sx, sy), 8), ((ex, ey), 6), ((wx, wy), 8)]:
                        cv2.circle(annotated, (px, py), r + 2, (0, 0, 0), -1)
                        cv2.circle(annotated, (px, py), r, TRACKED_COLOR, -1)

                    def _calc_angle(a, b, c):
                        v1 = np.array(b) - np.array(a)
                        v2 = np.array(c) - np.array(b)
                        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                        return 180 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                    
                    ang = _calc_angle([s.x, s.y, s.z], [e.x, e.y, e.z], [w.x, w.y, w.z])
                    label = f"{ang:.1f} deg"
                    cv2.putText(annotated, label, (ex + 13, ey - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(annotated, label, (ex + 12, ey - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TRACKED_COLOR, 2)

                else:
                    # FIX 2: fill with NaN instead of skipping the row
                    stats["fail"] += 1
                    writer.writerow([
                        f_idx, true_timestamp,
                        np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan,
                        0
                    ])

                info = f"F:{f_idx} Pass:{pass_id}"
                cv2.putText(annotated, info, (11, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(annotated, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                vw.write(annotated)
                f_idx += 1
                if f_idx % 30 == 0 or f_idx == total_frames:
                    print(f"\r Progress AI: {f_idx}/{total_frames} ({(f_idx/total_frames)*100:.1f}%)", end="", flush=True)

        cap_video.release()
        pose_processor_pass1.close()
        pose_processor_pass2.close()
        vw.release()

        _elapsed = time.perf_counter() - _t_proc_start
        _fps_proc = (f_idx / _elapsed) if _elapsed > 0 else 0.0
        _detected = stats["pass1"] + stats["pass2"]

        # FIX 4(d): print the results
        print("\n\n📊 DETECTION STATISTICS:")
        print(f"   - Total frames: {total_frames}")
        print(f"   - Detected in pass 1 (high quality): {stats['pass1']} ({stats['pass1']/total_frames*100:.1f}%)")
        print(f"   - Recovered in pass 2 (low quality): {stats['pass2']} ({stats['pass2']/total_frames*100:.1f}%)")
        print(f"   - Failed completely (NaN):         {stats['fail']} ({stats['fail']/total_frames*100:.1f}%)")
        print(f"   - Total detected (pass1+pass2):    {_detected} ({_detected/total_frames*100:.1f}%)")
        print(f"   ⏱️ Processing time: {_elapsed:.1f}s "
              f"({_fps_proc:.1f} frame/s) · pass 2b skipped (out of frame): {stats_clahe_skipped} frames")
        if stats["missing_ts"] > 0:
            _why = ("outside the interpolable range, dropped" if is_reprocess
                    else "not in frame_timestamps.csv")
            print(f"   ⚠️ WARNING: {stats['missing_ts']} frames missing a timestamp "
                  f"({_why}) -> written as NaN.")
        
        print("\n✅ EXPORTED: camera_data.csv + annotated_video.mp4\n")

    def start(self):
        self.collect_event.set()

    def stop(self):
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
                _now = time.perf_counter()
                if self.video_writer is not None:
                    self.video_writer.write(frame)
                if self.ts_writer is not None:
                    self.ts_writer.writerow([self.frame_idx, _now])
                self.last_frame_perf = _now   # liveness stamp for the watchdog
                self.frame_idx += 1

            cv2.imshow("Camera 60 FPS (Press 'q' to Stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()

        self.cap.release()
        cv2.destroyAllWindows()


# FIX 4(e): run from the terminal to reprocess an old session
if __name__ == "__main__":
    import sys
    # The default Windows console is cp1252 -> the emoji in the log (💾 🔄 ⚠️ ✅ 📊) would crash.
    # Force stdout/stderr to UTF-8 when run standalone from a terminal.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Reprocess a specific session.")
    parser.add_argument("--reprocess", type=str, required=True, help="Path to the session folder")
    args = parser.parse_args()

    session_path = Path(args.reprocess)
    if session_path.exists():
        # Call the staticmethod directly — do NOT instantiate CameraClient() to avoid opening the webcam.
        CameraClient._run_offline_post_processing(session_path)
    else:
        print(f"Error: Session folder '{session_path}' does not exist.")