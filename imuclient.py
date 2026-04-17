import sys
import asyncio
import struct
import csv
import time
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "Nano33BLE"  # Device name (more stable than address)
ADDRESS = "FB:5E:F5:A6:04:CB"  # Fallback address

IMU_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
CMD_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"

class IMUClient:

    def __init__(self):
        self.collecting = False
        self.file = open("imu_data.csv", "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "pc_time","arduino_time",
            "ax","ay","az","gx","gy","gz"
        ])

    def handler(self, sender, data):

        if not self.collecting:
            return

        pc_time = time.perf_counter()
        timestamp, ax, ay, az, gx, gy, gz = struct.unpack("I6f", data)

        self.writer.writerow([
            pc_time, timestamp,
            ax, ay, az, gx, gy, gz
        ])

    async def start(self):
        # Scan for BLE devices
        print("Scanning for BLE devices...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        device = None
        
        # First, try to match by device name (more stable)
        for d in devices:
            print(f"Found: {d.address} - {d.name}")
            if d.name == DEVICE_NAME:
                device = d
                break
        
        # Fallback: try to match by address
        if device is None:
            for d in devices:
                if d.address == ADDRESS:
                    device = d
                    break
        
        if device is None:
            print(f"Device '{DEVICE_NAME}' not found!")
            return
        
        print(f"Connecting to {device.name} ({device.address})...", flush=True)
        if sys.platform == "win32":
            self.client = BleakClient(device, timeout=20.0)
        else:
            self.client = BleakClient(device.address, timeout=20.0)

        print("Waiting for BLE connection...", flush=True)
        await asyncio.sleep(0.5)

        try:
            print("Starting client.connect()", flush=True)
            await asyncio.wait_for(self.client.connect(), timeout=20.0)
            print("client.connect() returned", flush=True)
        except asyncio.TimeoutError:
            print("Connection timed out after 20 seconds", flush=True)
            return
        except Exception as e:
            print(f"Connection failed: {type(e).__name__}: {e}", flush=True)
            return

        if not self.client.is_connected:
            print("Failed to connect to IMU")
            return

        print("IMU connected")

        try:
            await self.client.start_notify(IMU_UUID, self.handler)
        except Exception as e:
            print(f"Failed to start notifications: {type(e).__name__}: {e}")
            await self.client.disconnect()
            return

    async def send_start(self):
        self.collecting = True
        await self.client.write_gatt_char(CMD_UUID, b"START")

    async def send_stop(self):
        self.collecting = False
        await self.client.write_gatt_char(CMD_UUID, b"STOP")

    async def stop(self):
        await self.client.disconnect()
        self.file.close()