import sys
import asyncio
import struct
import csv
import time
import threading
from shareddata import shared_data

DEVICE_NAME = "Nano33BLE"
ADDRESS = "FB:5E:F5:A6:04:CB"  # Fallback address

class IMUClient:
    def __init__(self):
        self.collecting = False
        self._client = None
        self._imu_handle = None
        self._cmd_handle = None
        
        # Buffer in RAM to avoid bandwidth stalls during recording
        self.ram_buffer = []
        self.csv_path = None

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
    # Notification handler (UPDATED TO UNPACK BATCHES OF 5 ROWS)
    # ------------------------------------------------------------------

    def handler(self, sender, data):
        if not self.collecting:
            return

        # Moment the 140-byte packet arrives at the PC
        pc_time_arrival = time.perf_counter()

        row_size = 28 # Size of one IMU row (4-byte int + 6 * 4-byte float)
        num_rows = 5  # Number of rows in one batch packet

        # Unpack each data row inside the 140-byte packet
        for i in range(num_rows):
            start_idx = i * row_size
            end_idx = (i + 1) * row_size
            chunk = data[start_idx:end_idx]

            # Unpack that individual 28-byte row
            timestamp, ax, ay, az, gx, gy, gz = struct.unpack("<I6f", chunk)

            # These 5 rows were sampled exactly 10ms (0.01s) apart on the Arduino,
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

        print(f"Connecting to {device.name} ({device.address})...")

        for attempt in range(1, 4):
            print(f"  attempt {attempt}/3...")
            try:
                if sys.platform == "win32":
                    client = BleakClient(device, timeout=20.0, use_cached_services=False)
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

    async def start_client(self):
        """Start the BLE connection once, when the app launches."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._connect_async()))

    async def start_session(self):
        """Begin recording: send the START command and clear RAM."""
        self.ram_buffer.clear()
        self.collecting = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._write_cmd(b"START")))

    async def stop_session(self):
        """Stop recording: send STOP and flush RAM to CSV. Does NOT disconnect BLE."""
        self.collecting = False
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._write_cmd(b"STOP")))
        # Save the CSV file right after the Arduino stops streaming
        self.close_session()

    async def exit_client(self):
        """Shut down completely: disconnect BLE and stop the thread."""
        self.collecting = False
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._disconnect_async()))
        self._loop.call_soon_threadsafe(self._loop.stop)
        