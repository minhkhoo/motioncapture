import cv2
import mediapipe as mp
import csv
import time
import threading

class CameraClient:

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        self._lock = threading.Lock()
        self.file = None
        self.writer = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.frame = 0

        self.collect_event = threading.Event()
        self.stop_event = threading.Event()

    # =======================
    # SESSION
    # =======================

    def open_session(self, session_dir):
        with self._lock:
            if self.file:
                self.file.close()
            self.file = open(session_dir / "camera_data.csv", "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                "frame","timestamp",
                "shoulder_x","shoulder_y","shoulder_z","shoulder_visibility",
                "elbow_x","elbow_y","elbow_z","elbow_visibility",
                "wrist_x","wrist_y","wrist_z","wrist_visibility"
            ])
            self.frame = 0

    def close_session(self):
        with self._lock:
            if self.file:
                self.file.flush()
                self.file.close()
                self.file = None
                self.writer = None

    # =======================
    # CONTROL
    # =======================

    def start(self):
        print("Camera START")
        self.collect_event.set()

    def stop(self):
        print("Camera STOP")
        self.collect_event.clear()
        self.close_session()

    def shutdown(self):
        print("Camera SHUTDOWN")
        self.collect_event.clear()
        self.stop_event.set()

    # =======================
    # MAIN LOOP
    # =======================

    def run(self):

        while not self.stop_event.is_set():

            ret, frame = self.cap.read()
            if not ret:
                continue

            if self.collect_event.is_set():

                now = time.perf_counter()

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
                                self.frame, now,
                                s.x, s.y, s.z, s.visibility,
                                e.x, e.y, e.z, e.visibility,
                                w.x, w.y, w.z, w.visibility
                            ])
                            self.frame += 1

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()

        # cleanup
        self.cap.release()
        self.close_session()
        cv2.destroyAllWindows()