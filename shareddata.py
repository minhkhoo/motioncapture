import threading
from collections import deque

class SharedData:

    def __init__(self):

        self.lock = threading.Lock()

        self.camera_buffer = deque(maxlen=2000)
        self.imu_buffer = deque(maxlen=2000)

shared_data = SharedData()