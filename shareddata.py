import threading
from collections import deque


class SharedData:

    def __init__(self):

        self.lock = threading.Lock()

        self.camera_buffer = deque(maxlen=20000)
        self.imu_buffer = deque(maxlen=20000)

        self.session_dir = None

        self.graph_running = False

        # NEW
        self.program_running = True


shared_data = SharedData()