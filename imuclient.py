import sys
import asyncio
import struct
import csv
import time
import threading
from shareddata import shared_data

DEVICE_NAME = "Nano33BLE"
ADDRESS = ""  # Optional fallback: set your board's BLE MAC address if scanning by name fails

class IMUClient:
    def __init__(self):
        self.collecting = False
        self._client = None
        self._imu_handle = None
        self._cmd_handle = None
        
        # Buffer in RAM to avoid bandwidth stalls during recording
        self.ram_buffer = []
        self.csv_path = None

        # Liveness: perf_counter of the most recently received BLE packet (updated on the
        # BLE thread, read by the mid-session watchdog on the main thread). None = nothing
        # received since the current session started.
        self.last_rx_perf = None

        # Auto-reconnect state. _want_connection = we intend to stay connected (set once the
        # app connects, cleared on exit). _reconnecting = a background reconnect loop is
        # already running (avoid launching several). _device = the last found BLE device.
        # All reconnect work runs on the BLE thread/loop below — it NEVER blocks the main
        # (camera) thread, so a dropped/failed IMU can't affect camera recording.
        self._want_connection = False
        self._reconnecting = False
        self._device = None

        # Private event loop + thread for BLE
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def open_session(self, session_dir):
        # Only store the path and clear RAM — do NOT open file I/O here
        self.csv_path = session_dir / "imu_data.csv"
        self.ram_buffer.clear()

    def close_session(self):
        # END OF RECORDING: flush the entire RAM buffer to CSV in one write
        if self.csv_path is not None and len(self.ram_buffer) > 0:
            print(f"Writing {len(self.ram_buffer)} IMU rows to disk...")
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["pc_time", "arduino_time", "ax", "ay", "az", "gx", "gy", "gz"])
                writer.writerows(self.ram_buffer)
        
        self.ram_buffer.clear()
        self.csv_path = None

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run_loop(self):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0x0)
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro, timeout=120):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Notification handler (unpacks batches of any number of 28-byte rows; the current firmware sends 3)
    # ------------------------------------------------------------------

    def handler(self, sender, data):
        if not self.collecting:
            return

        # Moment the BLE packet arrives at the PC (84 bytes with the current firmware: 3 rows x 28 bytes)
        pc_time_arrival = time.perf_counter()
        # Liveness stamp for the mid-session watchdog (last time any packet arrived)
        self.last_rx_perf = pc_time_arrival

        row_size = 28 # Size of one IMU row (4-byte int + 6 * 4-byte float)
        num_rows = len(data) // row_size 

        # Unpack each data row inside the packet
        for i in range(num_rows):
            start_idx = i * row_size
            end_idx = (i + 1) * row_size
            chunk = data[start_idx:end_idx]

            # Unpack that individual 28-byte row
            timestamp, ax, ay, az, gx, gy, gz = struct.unpack("<I6f", chunk)

            # Rows in a packet were sampled exactly 10 ms (0.01 s) apart on the Arduino,
            # so reconstruct each row's pc_time by counting backwards from arrival.
            row_pc_time = pc_time_arrival - (num_rows - 1 - i) * 0.010

            # =========================
            # STORE IN RAM
            # =========================
            self.ram_buffer.append([
                row_pc_time, timestamp, ax, ay, az, gx, gy, gz
            ])

            # =========================
            # LIVE SHARED DATA (pushed continuously so the camera thread isn't delayed)
            # =========================
            shared_data.imu_buffer.append({
                "timestamp": row_pc_time,
                "wrist": (ax, ay, az)
            })

    # ------------------------------------------------------------------
    # BLE coroutines (run on the BLE thread)
    # ------------------------------------------------------------------

    async def _connect_async(self):
        from bleak import BleakClient, BleakScanner

        print("Scanning for BLE devices...")
        devices = await BleakScanner.discover(timeout=10.0)

        device = None
        for d in devices:
            print(f"Found: {d.address} - {d.name}")
            if d.name == DEVICE_NAME:
                device = d
                break
        if device is None:
            for d in devices:
                if d.address == ADDRESS:
                    device = d
                    break

        if device is None:
            raise RuntimeError(f"Device '{DEVICE_NAME}' not found")
        self._device = device

        print(f"Connecting to {device.name} ({device.address})...")

        for attempt in range(1, 4):
            print(f"  attempt {attempt}/3...")
            try:
                # disconnected_callback fires (on the BLE loop) when the link drops → triggers
                # the background reconnect loop. Camera is untouched.
                if sys.platform == "win32":
                    client = BleakClient(device, timeout=20.0, use_cached_services=False,
                                         disconnected_callback=self._on_disconnect)
                else:
                    client = BleakClient(device.address, timeout=20.0,
                                         disconnected_callback=self._on_disconnect)

                await client.connect()

                if not client.is_connected:
                    raise RuntimeError("not connected after connect()")

                imu_char = cmd_char = None
                for svc in client.services:
                    for ch in svc.characteristics:
                        if "notify" in ch.properties and imu_char is None:
                            imu_char = ch
                        elif "write" in ch.properties and cmd_char is None:
                            cmd_char = ch

                if imu_char is None or cmd_char is None:
                    raise RuntimeError("characteristics not found")

                await asyncio.sleep(0.5)

                if not client.is_connected:
                    raise RuntimeError("disconnected during settle")

                await client.start_notify(imu_char.handle, self.handler)

                self._client = client
                self._imu_handle = imu_char.handle
                self._cmd_handle = cmd_char.handle
                print("IMU connected, notifications started")
                return

            except Exception as e:
                print(f"  attempt {attempt} failed: {e}")
                try:
                    await client.disconnect()
                except Exception:
                    pass
                if attempt < 3:
                    await asyncio.sleep(5.0)

        raise RuntimeError("Failed to connect to IMU after 3 attempts")

    async def _write_cmd(self, payload: bytes):
        if self._client is None or not self._client.is_connected:
            raise RuntimeError("IMU not connected")
        await self._client.write_gatt_char(self._cmd_handle, payload)

    async def _disconnect_async(self):
        if self._client:
            try:
                await asyncio.wait_for(self._client.disconnect(), timeout=10)
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------
    # AUTO-RECONNECT (all on the BLE thread — never touches the camera)
    # ------------------------------------------------------------------

    def _on_disconnect(self, client):
        """Called by Bleak on the BLE loop when the link drops. Kick off a background
        reconnect. Does nothing on an intentional disconnect (exit)."""
        if not self._want_connection:
            return  # we asked to disconnect (exiting) — don't fight it
        print("\n🔌 [IMU] BLE link lost — reconnecting in the background (camera is NOT affected)...")
        try:
            self._loop.call_soon_threadsafe(
                lambda: self._loop.create_task(self._reconnect_loop()))
        except Exception:
            pass

    async def _reconnect_loop(self):
        """Keep trying to reconnect in the background until it succeeds or we exit.
        Runs entirely on the BLE loop; the main/camera thread is never blocked.
        NOTE: at long range the reconnect may keep failing — that's fine, it just keeps
        retrying quietly; camera recording continues regardless."""
        if self._reconnecting:
            return
        self._reconnecting = True
        try:
            self._client = None
            backoff = 3.0
            while self._want_connection:
                try:
                    await self._connect_async()   # scan + connect + start_notify
                    if self._client is not None and self._client.is_connected:
                        print("✓ [IMU] Reconnected.")
                        if self.collecting:
                            # resume streaming on the Arduino for the ongoing session
                            try:
                                await self._write_cmd(b"START")
                            except Exception:
                                pass
                        return
                except Exception as e:
                    print(f"  [IMU] could not reconnect yet ({e}); retrying in {backoff:.0f}s "
                          f"(camera keeps recording normally)...")
                await asyncio.sleep(backoff)
                backoff = min(backoff + 2.0, 15.0)   # back off gradually, max 15 s per attempt
        finally:
            self._reconnecting = False

    # ------------------------------------------------------------------
    # Public API (called from main thread / main event loop)
    # ------------------------------------------------------------------

    async def start_client(self):
        """Start the BLE connection once, when the app launches."""
        self._want_connection = True   # from now on, drops trigger auto-reconnect
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._connect_async()))

    async def start_session(self):
        """Begin recording: send START. NEVER blocks the recording flow or raises — if BLE
        is down/reconnecting, camera still records and the reconnect loop re-sends START on
        success (collecting stays True so the handler captures data as soon as it's back)."""
        self.ram_buffer.clear()
        # Reset liveness so the watchdog measures silence from the start of THIS session
        self.last_rx_perf = None
        self.collecting = True
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: self._submit(self._write_cmd(b"START"), timeout=8))
        except Exception as e:
            print(f"⚠️ [IMU] Could not send START ({e}) — camera keeps recording; "
                  f"IMU data will start once the link is back.")

    async def stop_session(self):
        """Stop recording: send STOP and flush RAM to CSV. Never raises — always writes
        whatever IMU rows were captured, so an IMU problem can't break the session/camera."""
        self.collecting = False
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: self._submit(self._write_cmd(b"STOP"), timeout=8))
        except Exception as e:
            print(f"⚠️ [IMU] Could not send STOP ({e}) — still saving the data collected.")
        # Save the CSV file with whatever we collected (even if partial / empty)
        self.close_session()

    async def exit_client(self):
        """Shut down completely: disconnect BLE and stop the thread."""
        self._want_connection = False   # stop any background reconnect loop
        self.collecting = False
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, lambda: self._submit(self._disconnect_async(), timeout=12))
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        