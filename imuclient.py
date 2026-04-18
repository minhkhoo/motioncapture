import sys
import asyncio
import struct
import csv
import time
import threading

DEVICE_NAME = "Nano33BLE"
ADDRESS = "FB:5E:F5:A6:04:CB"  # Fallback address


class IMUClient:
    """
    Runs all BLE operations on a private background thread with its own
    asyncio event loop.  That thread calls CoInitializeEx(MTA) before
    touching bleak, so mediapipe/TensorFlow COM pollution in the main
    thread cannot interfere with WinRT session-change callbacks.
    """

    def __init__(self):
        self.collecting = False
        self._client = None
        self._imu_handle = None
        self._cmd_handle = None
        self.file = None
        self.writer = None

        # Private event loop + thread for BLE
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def open_session(self, session_dir):
        if self.file:
            self.file.close()
        self.file = open(session_dir / "imu_data.csv", "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["pc_time", "arduino_time",
                               "ax", "ay", "az", "gx", "gy", "gz"])

    def close_session(self):
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
            self.writer = None

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run_loop(self):
        """Entry point of the BLE thread.  Sets COM=MTA then runs the loop."""
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro, timeout=120):
        """Schedule a coroutine on the BLE thread and block until done."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Notification handler (called on BLE thread)
    # ------------------------------------------------------------------

    def handler(self, sender, data):
        if not self.collecting or self.writer is None:
            return
        pc_time = time.perf_counter()
        timestamp, ax, ay, az, gx, gy, gz = struct.unpack("I6f", data)
        self.writer.writerow([pc_time, timestamp,
                               ax, ay, az, gx, gy, gz])

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

        print(f"Connecting to {device.name} ({device.address})...")

        for attempt in range(1, 4):
            print(f"  attempt {attempt}/3...")
            try:
                if sys.platform == "win32":
                    client = BleakClient(device, timeout=20.0,
                                         use_cached_services=False)
                else:
                    client = BleakClient(device.address, timeout=20.0)

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

                # Small settle: the BLE thread has proper MTA so this sleep
                # does NOT cause a disconnect — it gives the Arduino time
                # to finish connection parameter negotiation.
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
    # Public API (called from main thread / main event loop)
    # ------------------------------------------------------------------

    async def start(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._connect_async()))

    async def send_start(self):
        self.collecting = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._write_cmd(b"START")))

    async def send_stop(self):
        self.collecting = False
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._write_cmd(b"STOP")))

    async def stop(self):
        self.collecting = False
        self.close_session()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._disconnect_async()))
        self._loop.call_soon_threadsafe(self._loop.stop)