import sys
import os
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

from dataprocessing import live_graph
from dataprocessing import save_replay

from shareddata import shared_data

print("NEW VERSION RUNNING")

# =========================
# RECORDING
# =========================

recording_dir = Path("recording")

imu = IMUClient()
cam = CameraClient()

recording = False

# =========================
# HARD KILL
# =========================

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
# MAIN ASYNC LOOP
# =========================

async def main():

    global recording

    # =========================
    # CREATE RECORDING FOLDER
    # =========================

    recording_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================
    # START CAMERA THREAD
    # =========================

    cam_thread = threading.Thread(
        target=cam.run,
        daemon=True
    )

    cam_thread.start()

    # =========================
    # CONNECT IMU
    # =========================

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
            # START RECORDING
            # =========================

            if cmd == "start":

                if recording:

                    print("Already recording")
                    continue

                recording = True

                # graph ON
                shared_data.graph_running = True

                # clear old live data
                shared_data.camera_buffer.clear()
                shared_data.imu_buffer.clear()

                # create session folder
                session_dir = (
                    recording_dir /
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                )

                session_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shared_data.session_dir = session_dir

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
            # STOP RECORDING
            # =========================

            elif cmd == "stop":

                if not recording:

                    print("Not recording")
                    continue

                recording = False

                print("SYSTEM STOP")

                # graph OFF
                shared_data.graph_running = False

                # stop imu
                await imu.send_stop()

                # stop camera recording
                cam.stop()

                # close csv
                imu.close_session()
                cam.close_session()

                # save replay
                print("Saving replay...")

                try:

                    save_replay()

                except Exception as e:

                    print("Replay save error:", e)

                print("GRAPH STOP")

            # =========================
            # EXIT
            # =========================

            elif cmd == "exit":

                break

    except (KeyboardInterrupt, EOFError):

        pass

    finally:

        try:

            # =========================
            # AUTO STOP IF RECORDING
            # =========================

            if recording:

                print("Auto stopping recording...")

                recording = False

                shared_data.graph_running = False

                await imu.send_stop()

                cam.stop()

                imu.close_session()
                cam.close_session()

                try:

                    save_replay()

                except Exception as e:

                    print("Replay save error:", e)

            # =========================
            # SHUTDOWN
            # =========================

            await asyncio.wait_for(
                shutdown(),
                timeout=5
            )

        except Exception as e:

            print("Shutdown error:", e)

        finally:

            # IMPORTANT:
            # stop graph loop completely
            shared_data.program_running = False

            cam.shutdown()

            cam_thread.join(timeout=3)

            print("Ended")
# =========================
# RUN
# =========================

if __name__ == "__main__":

    def run_async():

        try:

            asyncio.run(main())

        except Exception:

            import traceback
            traceback.print_exc()

    # async logic thread
    async_thread = threading.Thread(
        target=run_async,
        daemon=True
    )

    async_thread.start()

    # IMPORTANT:
    # matplotlib must run on MAIN THREAD
    live_graph()