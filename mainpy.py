import sys
import os
import asyncio
import signal

if sys.platform == "win32":
    os.environ["PYTHONNET_INITIALIZE"] = "0"
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import threading
from datetime import datetime
from pathlib import Path

from imuclient import IMUClient
from cameraclient import CameraClient
from dataprocessing import live_graph

print("NEW VERSION RUNNING")

# =========================
# RECORDING
# =========================

recording_dir = Path("recording")

imu = IMUClient()
cam = CameraClient()

recording = False

# =========================
# GRAPH
# =========================

graph_thread = None
graph_stop_event = threading.Event()

# Hard kill on second Ctrl+C
signal.signal(signal.SIGINT, lambda *_: os._exit(1))


# =========================
# SHUTDOWN
# =========================

async def shutdown():

    print("\nShutting down...")

    try:
        await imu.send_stop()
    except Exception:
        pass

    try:
        await imu.stop()
    except Exception:
        pass


# =========================
# MAIN
# =========================

async def main():

    global recording
    global graph_thread

    # ensure recording folder exists
    recording_dir.mkdir(parents=True, exist_ok=True)

    # start camera thread
    cam_thread = threading.Thread(
        target=cam.run,
        daemon=True
    )

    cam_thread.start()

    # connect IMU
    await imu.start()

    print("READY (type start / stop / exit)")

    try:

        while True:

            cmd = await asyncio.get_event_loop().run_in_executor(
                None,
                input
            )

            cmd = cmd.strip().lower()

            # =========================
            # START
            # =========================

            if cmd == "start":

                if recording:
                    print("Already recording")
                    continue

                recording = True

                # create session folder
                session_dir = (
                    recording_dir /
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                )

                session_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                print(f"SESSION: {session_dir}")

                # open CSV files
                imu.open_session(session_dir)
                cam.open_session(session_dir)

                print("imu writer:", imu.writer)
                print("cam writer:", cam.writer)

                # start sensors
                await imu.send_start()

                cam.start()

                print("SYSTEM START")

                # =========================
                # START GRAPH
                # =========================

                graph_stop_event.clear()

                graph_thread = threading.Thread(
                    target=live_graph,
                    args=(graph_stop_event,),
                    daemon=True
                )

                graph_thread.start()

                print("GRAPH START")

            # =========================
            # STOP
            # =========================

            elif cmd == "stop":

                if not recording:
                    print("Not recording")
                    continue

                recording = False

                print("SYSTEM STOP")

                await imu.send_stop()

                cam.stop()

                imu.close_session()

                # stop graph
                graph_stop_event.set()

                print("GRAPH STOP")

            # =========================
            # EXIT
            # =========================

            elif cmd == "exit":

                break

    except (KeyboardInterrupt, EOFError):

        pass

    finally:

        graph_stop_event.set()

        try:
            await asyncio.wait_for(
                shutdown(),
                timeout=5
            )

        except (asyncio.TimeoutError, Exception):
            pass

        cam.shutdown()

        cam_thread.join(timeout=3)

        print("Ended")

        os._exit(0)


# =========================
# RUN
# =========================

asyncio.run(main())