import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

import sys
import json
import time
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
from shareddata import shared_data

# =========================
# RECORDING CONFIG
# =========================
recording_dir = Path("recording")
recording = False

# =========================
# HARD KILL
# =========================
signal.signal(signal.SIGINT, lambda *_: os._exit(1))

# =========================
# MAIN ASYNC LOOP
# =========================

# =====================================================================
# HELPER: create a .bat shortcut to open the viewer by double-clicking
# =====================================================================
def _write_viewer_bat(session_dir):
    """Create Open_Viewer.bat inside the session folder.
    Double-click opens the viewer without typing a terminal command.
    """
    from pathlib import Path
    session_dir = Path(session_dir)
    # Compute the relative path from session_dir up to the project root
    # session_dir = recording/PlayerName_20260612.../  → need to cd up 2 levels
    bat_content = (
        "@echo off\n"
        "cd /d \"%~dp0..\\..\"\n"
        f'uv run python dataprocessing.py "recording\\{session_dir.name}"\n'
        "pause\n"
    )
    bat_path = session_dir / "Open_Viewer.bat"
    bat_path.write_text(bat_content, encoding="utf-8")


# =====================================================================
# MID-SESSION LIVENESS WATCHDOG
# Warn (print only) when the camera or IMU stops producing data mid-session.
# Never stops recording, never touches the other source.
# =====================================================================
async def _liveness_watchdog(cam, imu, timeout_s=2.0, poll_s=1.0):
    """Poll the two sources every `poll_s` seconds while recording.

    A source is "lost" when it hasn't produced data for more than `timeout_s`.
    Prints exactly ONCE per state transition (OK→lost and lost→recovered) so it
    never spams. Both liveness stamps use time.perf_counter() (same process), and
    each source is measured from the session start until its first sample arrives.
    """
    t_start = time.perf_counter()
    imu_ok = True
    cam_ok = True
    try:
        while True:
            await asyncio.sleep(poll_s)
            now = time.perf_counter()
            wall = datetime.now().strftime("%H:%M:%S")

            # --- IMU ---
            if imu is not None:
                last = imu.last_rx_perf
                silent = now - (last if last is not None else t_start)
                if imu_ok and silent > timeout_s:
                    imu_ok = False
                    print(f"\n[{wall}] ⚠ IMU: no data received for {silent:.1f}s "
                          f"— check the BLE connection/distance. (Recording continues)")
                elif (not imu_ok) and silent <= timeout_s:
                    imu_ok = True
                    print(f"\n[{wall}] ✓ IMU reconnected.")

            # --- CAMERA ---
            if cam is not None:
                last = cam.last_frame_perf
                silent = now - (last if last is not None else t_start)
                if cam_ok and silent > timeout_s:
                    cam_ok = False
                    print(f"\n[{wall}] ⚠ Camera: no new frame written for {silent:.1f}s "
                          f"— check the webcam/Iriun. (Recording continues)")
                elif (not cam_ok) and silent <= timeout_s:
                    cam_ok = True
                    print(f"\n[{wall}] ✓ Camera is working again.")
    except asyncio.CancelledError:
        # Normal shutdown of the watchdog when recording stops
        raise


async def _stop_watchdog(task):
    """Cancel the watchdog task and wait for it to finish. Safe to call with None
    or an already-finished task."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def main():
    global recording

    # ========================================================
    #  STEP 1: MODE SELECTION MENU
    # ========================================================
    print("\n" + "="*40)
    print(" STARTING THE DATA COLLECTION SYSTEM")
    print("="*40)
    print("1. CAMERA only")
    print("2. IMU SENSOR only")
    print("3. Run BOTH devices simultaneously")
    print("="*40)

    def get_mode_input():
        return input("Enter your choice (1/2/3): ").strip()

    mode_choice = await asyncio.get_event_loop().run_in_executor(None, get_mode_input)

    use_camera = mode_choice in ["1", "3"]
    use_imu = mode_choice in ["2", "3"]

    print(f"\n➔ Active devices: CAMERA={use_camera} | IMU={use_imu}\n")

    # Initialize objects based on the chosen mode
    imu = None
    cam = None
    cam_thread = None

    if use_camera:
        cam = CameraClient()
        # START CAMERA THREAD (runs in the background capturing frames at 60 FPS)
        cam_thread = threading.Thread(target=cam.run, daemon=True)
        cam_thread.start()
        print("Camera 60 FPS thread is ready.")

    if use_imu:
        try:
            imu = IMUClient()
            await imu.start_client() # CONNECT OVER BLUETOOTH ONLY ONCE, HERE
            print("IMU Arduino connected successfully and holding the link.")
        except Exception as e:
            print(f"Failed to connect to the Arduino: {e}")
            print("Falling back to CAMERA ONLY mode automatically.")
            use_imu = False

    # Create the root data folder
    recording_dir.mkdir(parents=True, exist_ok=True)

    try:
        # MAIN LOOP: idle state, waiting to record or exit
        while True:
            print("\n=== SYSTEM READY (type: start / exit) ===")
            cmd = await asyncio.get_event_loop().run_in_executor(None, input)
            cmd = cmd.strip().lower()

            # ========================================================
            # COMMAND: EXIT (only allowed while idle)
            # ========================================================
            if cmd == "exit":
                print("Preparing to shut down the whole system...")
                break

            # ========================================================
            # COMMAND: START RECORDING
            # ========================================================
            elif cmd == "start":
                # Ask for the session KIND before recording. The session_info.json
                # written here is the source of truth that build_tracking.py uses to tell
                # real data from test sessions to drop.
                def get_session_kind():
                    while True:
                        print("\nSession kind?  [1] professional   [2] amateur   "
                              "[3] test")
                        c = input("Enter your choice (1/2/3): ").strip()
                        if c in ("1", "2", "3"):
                            return {"1": "professional", "2": "amateur", "3": "test"}[c]
                        print("  Invalid input. Only 1, 2 or 3 are accepted.")

                session_kind = await asyncio.get_event_loop().run_in_executor(
                    None, get_session_kind)

                # Test session: do NOT ask for a name. Real session: ask for the athlete name as before.
                player_id = None
                if session_kind in ("professional", "amateur"):
                    def get_player_name():
                        return input("Enter athlete name or ID (press Enter to skip): ").strip()
                    player_id = await asyncio.get_event_loop().run_in_executor(
                        None, get_player_name)
                    if not player_id:
                        player_id = "Player"

                recording = True
                shared_data.graph_running = True

                # Clear the old chart buffers
                shared_data.camera_buffer.clear()
                shared_data.imu_buffer.clear()

                # Folder name: test sessions start with TEST_ so they're obvious at a glance.
                # Do NOT append a primary/secondary suffix — that is assigned later after
                # reviewing the video; it isn't known at recording time.
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                if session_kind == "test":
                    session_dir = recording_dir / f"TEST_{time_str}"
                else:
                    session_dir = recording_dir / f"{player_id}_{time_str}"
                session_dir.mkdir(parents=True, exist_ok=True)
                shared_data.session_dir = session_dir

                # session_info.json — source of truth about the session kind
                session_info = {
                    "session_kind": session_kind,
                    "player_id":    player_id,      # None if this is a test
                    "recorded_at":  datetime.now().isoformat(),
                }
                (session_dir / "session_info.json").write_text(
                    json.dumps(session_info, ensure_ascii=False, indent=2),
                    encoding="utf-8")

                # (The HTML report is generated automatically after stop, by dataprocessing.py)

                print(f"SESSION DIR: {session_dir}  [{session_kind}]")

                # Activate the recorder for each device.
                # CAMERA FIRST, IMU SECOND — the order matters:
                #   cam.start() only sets a flag (returns instantly), so the camera starts
                #   recording immediately. imu.start_session() sends START over BLE and AWAITS
                #   it — when the BLE link is congested this blocks for seconds. If the IMU
                #   went first, the camera would miss the start of the motion (measured on a
                #   real session: the camera only began recording after the spike had ended).
                # Both timestamps use time.perf_counter() in the same process, so the delay
                # below is directly comparable and tells us how slow BLE was this time.
                t_cam_start = t_imu_start = None
                if use_camera and cam:
                    cam.open_session(session_dir)
                    cam.start()
                    t_cam_start = time.perf_counter()
                    print(f"[TIMING] Camera started @ perf_counter={t_cam_start:.6f}")

                if use_imu and imu:
                    # IMU is isolated: any failure here (BLE down, reconnecting, sensor dead)
                    # must NOT interrupt the camera, which is already recording above.
                    try:
                        imu.open_session(session_dir)
                        await imu.start_session()  # non-blocking, non-raising by design
                        t_imu_start = time.perf_counter()
                        print(f"[TIMING] IMU    started @ perf_counter={t_imu_start:.6f}")
                    except Exception as e:
                        print(f"⚠️ [IMU] start failed ({e}) — skipped, camera keeps recording normally.")

                if t_cam_start is not None and t_imu_start is not None:
                    print(f"[TIMING] BLE start delay (IMU − camera): "
                          f"{(t_imu_start - t_cam_start) * 1000:.1f} ms")

                # Mid-session liveness watchdog (BOTH mode only). Prints a warning if the
                # camera or IMU stops producing data mid-session, and a recovery note when
                # it resumes. It NEVER stops recording and never touches the other source —
                # the session runs until the user types 'stop'.
                watchdog_task = None
                if use_imu and imu and use_camera and cam:
                    watchdog_task = asyncio.create_task(_liveness_watchdog(cam, imu))

                print("\n>>> RECORDING DATA (type 'stop' to finish) <<<")

                # ----------------------------------------------------
                # NESTED LOOP: must wait for the STOP command while recording
                # ----------------------------------------------------
                try:
                    while True:
                        sub_cmd = await asyncio.get_event_loop().run_in_executor(None, input)
                        sub_cmd = sub_cmd.strip().lower()

                        if sub_cmd == "stop":
                            # Stop the watchdog FIRST so the intentional stop below (which
                            # naturally halts both sources) doesn't trigger a false alarm.
                            await _stop_watchdog(watchdog_task)
                            watchdog_task = None

                            recording = False
                            shared_data.graph_running = False
                            print("\n[STOPPING SYSTEM] - Stopping devices and exporting data...")

                            if use_imu and imu:
                                # Isolated: a failed IMU stop must not block camera export
                                try:
                                    await imu.stop_session() # Stop stream + write RAM→CSV (does not disconnect BLE)
                                except Exception as e:
                                    print(f"⚠️ [IMU] stop failed ({e}) — still saving the camera data.")

                            if use_camera and cam:
                                cam.stop()
                                cam.close_session() # Trigger AI post-processing for the camera

                            # Export the HTML report (don't open a browser — we're in the CLI loop)
                            if use_camera and (session_dir / "camera_data.csv").exists():
                                print("  → Exporting motion_report.html ...")
                                try:
                                    import dataprocessing
                                    dataprocessing.run(str(session_dir), open_browser=False)
                                    _write_viewer_bat(session_dir)
                                    report_path = session_dir / "motion_report.html"
                                    print(f"  → View report: uv run python dataprocessing.py {session_dir}")
                                except Exception as e:
                                    print(f"  ⚠️  Could not create the report: {e}")

                            print("=== 100% DATA SAVED & POST-PROCESSING COMPLETED ===")
                            print("Returning you to the main menu automatically.")
                            break # Break the inner loop, back to the main menu for the next start/exit

                        elif sub_cmd == "exit":
                            print(" WARNING: recording in progress — type 'stop' before exiting.")
                        else:
                            print(" Invalid command, only 'stop' is accepted.")
                finally:
                    # Safety net: make sure the watchdog is stopped on any exit path
                    # (e.g. Ctrl-C / EOF while recording), not just the clean 'stop'.
                    await _stop_watchdog(watchdog_task)

            else:
                print(" Invalid command, only 'start' or 'exit' are accepted.")

    except (KeyboardInterrupt, EOFError):
        pass

    finally:
        # ========================================================
        # CLEAN UP AND FULLY DISCONNECT WHEN EXITING THE APP
        # ========================================================
        try:
            if recording:
                print("Auto stopping recording due to abrupt exit...")
                recording = False
                shared_data.graph_running = False

                if use_imu and imu:
                    await imu.stop_session()

                if use_camera and cam:
                    cam.stop()
                    cam.close_session()

            # Safely close the Bluetooth connection before killing the thread
            if use_imu and imu:
                print("Disconnecting Bluetooth from the Arduino...")
                await imu.exit_client()

        except Exception as e:
            print("Shutdown error:", e)

        finally:
            shared_data.program_running = False

            if use_camera and cam:
                cam.shutdown()

            if use_camera and cam_thread:
                cam_thread.join(timeout=3)

            print("=== PROGRAM EXITED SAFELY ===")

# =====================================================================
# RUN DIRECTLY ON THE MAIN THREAD
# =====================================================================
if __name__ == "__main__":
    shared_data.program_running = True
    try:
        # Run main() directly on the main thread to listen for CLI signals and keep the app alive
        asyncio.run(main())
    except Exception:
        import traceback
        traceback.print_exc()