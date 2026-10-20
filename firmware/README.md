# Firmware — Arduino Nano 33 BLE IMU streamer

`arduino_device.ino` turns an Arduino Nano 33 BLE into a wrist-worn IMU that streams
accelerometer + gyroscope samples over Bluetooth Low Energy. The laptop side is
`imuclient.py` (in the repository root); `mainpy.py` starts and stops it.

## What the sketch does

1. Initialises the on-board LSM9DS1 IMU and the BLE stack, advertises as **`Nano33BLE`**.
2. Waits for a central (the laptop) to connect.
3. On the command `START` it samples the IMU every 10 ms (100 Hz nominal), packs 3 samples
   into one BLE notification, and sends it. `STOP` (or a disconnect) stops streaming.

It does **not** record anything on the board and applies **no calibration**
(`setGyroSlope()` and any offset/slope calls are not used).

## How to flash

1. Install the Arduino IDE and, via Boards Manager, **Arduino Mbed OS Nano Boards 4.5.0**
   (the version used for the recordings).
2. Install the libraries:
   - **ArduinoBLE** (Library Manager).
   - **Arduino_LSM9DS1 — FemmeVerbeek fork, v2.0.0**, installed from its `.zip`
     (Sketch > Include Library > Add .ZIP Library). It *replaces* the official
     `Arduino_LSM9DS1`; the sketch needs this fork because it calls `IMU.setAccelFS()`,
     which the official library does not have.
3. Open `arduino_device.ino`, select *Arduino Nano 33 BLE*, and upload.
4. Open the Serial Monitor at 115200 baud. You should see `BLE Ready`, then `Connected`
   when the laptop connects and `START received` / `STOP received` during a session.

If the IMU fails to start the board hangs silently (no `BLE Ready`): re-seat the board
and re-upload.

## BLE interface

| Role | UUID |
|------|------|
| IMU service | `19B10000-E8F2-537E-4F6C-D104768A1214` |
| IMU data characteristic (Notify) | `19B10001-E8F2-537E-4F6C-D104768A1214` |
| Command characteristic (Write, max 20 bytes) | `19B10002-E8F2-537E-4F6C-D104768A1214` |

Commands are UTF-8 strings written to the command characteristic: `START`, `STOP`.

`imuclient.py` finds the board by its advertised name `Nano33BLE`, and does not need the
UUIDs: it subscribes to the first *notify* characteristic and writes to the first *write*
characteristic it finds. If scanning by name fails, set `ADDRESS` (the board's BLE MAC
address) at the top of `imuclient.py`.

## Notification packet

One notification = **3 samples × 28 bytes = 84 bytes** (`BATCH_SIZE = 3`).
Each 28-byte sample is little-endian:

| Offset | Type | Field |
|--------|------|-------|
| 0 | uint32 | `timestamp` — Arduino `micros()` when the sample was read |
| 4, 8, 12 | float32 | `ax`, `ay`, `az` (g) |
| 16, 20, 24 | float32 | `gx`, `gy`, `gz` (deg/s) |

Python: `struct.unpack("<I6f", chunk)`. `imuclient.py` splits the packet into
`len(data) // 28` samples, so it also works with a different `BATCH_SIZE`.

### Timestamps
- `arduino_time` = the board's `micros()` (wraps after about 71 minutes). It is used for
  spacing between samples and for the sampling-rate and packet-loss checks.
- `pc_time` = `time.perf_counter()` on the laptop when the packet arrives; each sample's
  `pc_time` is reconstructed by counting back in 10 ms steps from the arrival time.
  Camera frames use the same clock, which is how the two streams are aligned.

### Rates and link settings
| Parameter | Value |
|-----------|-------|
| Sample interval | 10 ms (`readInterval = 10000` µs), nominal 100 Hz |
| Notifications | 3 samples each, about 33 per second |
| Connection interval | 7.5 ms (`BLE.setConnectionInterval(6, 6)`, 1.25 ms units) |
| MTU | not set (stack default) |

Sampling is done in the main loop, so the achieved rate can fall below 100 Hz on a weak
link. The QC step reports the effective rate and packet loss per recording.

## Sensor settings

| Setting | Value | Where it comes from |
|---------|-------|---------------------|
| Gyroscope full scale | **±2000 deg/s** | Library default: `IMU.begin()` writes `CTRL_REG1_G = 0x78` (119 Hz, 2000 dps). The sketch never calls `setGyroFS()` |
| Output data rate | 119 Hz (sensor), read at 100 Hz | Library default |
| Accelerometer | `IMU.setAccelFS(1)` (up to about ±24 g) | Set explicitly in `setup()` |

The accelerometer is set to the library's highest range (`IMU.setAccelFS(1)`, readings up
to about ±24 g) so that hard impacts do not clip. Earlier recordings used ±4 g and then
±8 g; `build_tracking.py` estimates the range of each recording from its peak values
(above 8.5 g: 24 g; above 4.05 g: 8 g; otherwise 4 g), so older data is handled too.

The LSM9DS1 datasheet specifies 16 g as the largest accelerometer range. The library scales index 1 so that the 16-bit output reaches about 24 g; readings above 16 g are outside the specified range. The library converts gyroscope counts at 2000/32768 degrees per second per count, about 13 percent below the datasheet sensitivity of 70 mdps per count, and all angular velocities in the data are in that library scaling.

## Versions used
| Component | Version |
|-----------|---------|
| Arduino Mbed OS Nano Boards | 4.5.0 |
| Arduino_LSM9DS1 (FemmeVerbeek) | 2.0.0 |
| ArduinoBLE | 2.0.1 (as installed on the recording machine) |
