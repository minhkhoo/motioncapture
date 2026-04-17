import sys

if sys.platform == "win32":
    import os
    # This prevents certain libraries from initializing COM in a way 
    # that breaks Bleak/WinRT
    os.environ["PYTHONNET_INITIALIZE"] = "0"

    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        from bleak.backends.winrt.util import allow_sta
        allow_sta()
    except ImportError:
        pass
else:
    import asyncio

import threading
from imuclient import IMUClient
from cameraclient import CameraClient

imu = IMUClient()
cam = CameraClient()

async def main():

    await imu.start()

    print("READY (type start / stop / exit)")

    # 🔥 chạy camera thread 1 lần duy nhất
    cam_thread = threading.Thread(target=cam.run)
    cam_thread.start()

    while True:

        cmd = input().lower()

        if cmd == "start":

            print("SYSTEM START")

            await imu.send_start()
            cam.start()

        elif cmd == "stop":

            print("SYSTEM STOP")

            await imu.send_stop()
            cam.stop()

        elif cmd == "exit":

            print("SYSTEM EXIT")

            await imu.stop()
            cam.shutdown()

            cam_thread.join()
            break

asyncio.run(main())