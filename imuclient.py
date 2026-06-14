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
        
        # SỬ DỤNG RAM ĐỂ LƯU TẠM - TRÁNH NGHẼN BĂNG THÔNG
        self.ram_buffer = []
        self.csv_path = None

        # Private event loop + thread for BLE
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def open_session(self, session_dir):
        # Chỉ lưu đường dẫn và dọn dẹp RAM, KHÔNG MỞ FILE I/O Ở ĐÂY
        self.csv_path = session_dir / "imu_data.csv"
        self.ram_buffer.clear()

    def close_session(self):
        # KHI KẾT THÚC THU: Đổ toàn bộ data từ RAM vào file CSV 1 lần duy nhất
        if self.csv_path is not None and len(self.ram_buffer) > 0:
            print(f"Đang ghi {len(self.ram_buffer)} dòng IMU vào ổ cứng...")
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
    # Notification handler (ĐÃ CẬP NHẬT ĐỂ BÓC TÁCH BATCH 5 DÒNG)
    # ------------------------------------------------------------------

    def handler(self, sender, data):
        if not self.collecting:
            return

        # Thời điểm gói tin 140 bytes cập bến PC
        pc_time_arrival = time.perf_counter()
        
        row_size = 28 # Size của 1 dòng IMU (4 bytes int + 6 * 4 bytes float)
        num_rows = 5  # Số lượng dòng trong 1 gói batch
        
        # Vòng lặp bóc tách từng dòng dữ liệu trong gói 140 bytes
        for i in range(num_rows):
            start_idx = i * row_size
            end_idx = (i + 1) * row_size
            chunk = data[start_idx:end_idx]
            
            # Unpack riêng gói 28 bytes đó ra
            timestamp, ax, ay, az, gx, gy, gz = struct.unpack("<I6f", chunk)

            # Vì 5 dòng này được lấy mẫu cách nhau đúng 10ms (0.01s) trên Arduino
            # Ta tính toán lại chính xác pc_time cho từng dòng dựa theo thời gian lùi dần
            row_pc_time = pc_time_arrival - (num_rows - 1 - i) * 0.010

            # =========================
            # LƯU VÀO RAM 
            # =========================
            self.ram_buffer.append([
                row_pc_time, timestamp, ax, ay, az, gx, gy, gz
            ])

            # =========================
            # LIVE SHARED DATA (Đẩy liên tục để luồng camera không bị trễ)
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
        """Khởi động kết nối BLE một lần duy nhất khi bật app"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._connect_async()))

    async def start_session(self):
        """Bắt đầu thu data: Gửi lệnh START và dọn RAM"""
        self.ram_buffer.clear() 
        self.collecting = True
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._write_cmd(b"START")))

    async def stop_session(self):
        """Dừng thu data: Gửi lệnh STOP và đổ RAM ra file CSV. KHÔNG NGẮT BLE"""
        self.collecting = False
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._write_cmd(b"STOP")))
        # Lưu file CSV ngay sau khi Arduino ngừng gửi
        self.close_session()

    async def exit_client(self):
        """Đóng hoàn toàn chương trình: Ngắt BLE và tắt thread"""
        self.collecting = False
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._submit(self._disconnect_async()))
        self._loop.call_soon_threadsafe(self._loop.stop)