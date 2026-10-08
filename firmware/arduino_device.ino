#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>   // FemmeVerbeek fork (v2.0.0) - NOT the official Arduino_LSM9DS1

BLEService imuService("19B10000-E8F2-537E-4F6C-D104768A1214");

#define BATCH_SIZE 3   // samples per BLE notification (reduced from 5: smaller packets are more robust on a weak link)

struct IMUData {
  uint32_t timestamp;
  float ax, ay, az;
  float gx, gy, gz;
};

struct IMUBatch {
  IMUData readings[BATCH_SIZE];
};

BLECharacteristic imuChar(
  "19B10001-E8F2-537E-4F6C-D104768A1214",
  BLENotify,
  sizeof(IMUBatch)
);

BLECharacteristic cmdChar(
  "19B10002-E8F2-537E-4F6C-D104768A1214",
  BLEWrite,
  20
);

bool streaming = false;
unsigned long lastReadTime = 0;
const unsigned long readInterval = 10000;   // 10 ms -> 100 Hz
int batchIndex = 0;
IMUBatch currentBatch;

void setup() {
  Serial.begin(115200);

  if (!IMU.begin()) while (1);

  // Highest accelerometer range (code 1, up to about +-24 g) to avoid clipping on hard impacts.
  // FemmeVerbeek codes: 0 = +-2 g, 1 = highest range, 2 = +-4 g, 3 = +-8 g
  IMU.setAccelFS(1);

  if (!BLE.begin()) while (1);

  BLE.setLocalName("Nano33BLE");
  BLE.setAdvertisedService(imuService);

  imuService.addCharacteristic(imuChar);
  imuService.addCharacteristic(cmdChar);

  BLE.addService(imuService);

  // Fastest connection interval (7.5 ms = 6 x 1.25 ms).
  // Must be set AFTER addService and BEFORE advertise.
  BLE.setConnectionInterval(6, 6);

  BLE.advertise();

  Serial.println("BLE Ready");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.println("Connected");

    while (central.connected()) {

      // IMPORTANT: service the BLE stack on every loop iteration.
      // Without this the BLE buffer fills up and the Arduino freezes for 1-3 seconds.
      BLE.poll();

      if (cmdChar.written()) {
        String cmd = "";
        int len = cmdChar.valueLength();
        const uint8_t* data = cmdChar.value();

        for (int i = 0; i < len; i++) {
          cmd += (char)data[i];
        }
        cmd.trim();

        if (cmd == "START") {
          streaming = true;
          batchIndex = 0;
          lastReadTime = micros();
          Serial.println("START received");
        }
        if (cmd == "STOP") {
          streaming = false;
          Serial.println("STOP received");
        }
      }

      if (streaming) {
        unsigned long currentTime = micros();

        if (currentTime - lastReadTime >= readInterval) {
          lastReadTime = currentTime;

          float ax, ay, az, gx, gy, gz;
          IMU.readAcceleration(ax, ay, az);
          IMU.readGyroscope(gx, gy, gz);

          currentBatch.readings[batchIndex].timestamp = currentTime;
          currentBatch.readings[batchIndex].ax = ax;
          currentBatch.readings[batchIndex].ay = ay;
          currentBatch.readings[batchIndex].az = az;
          currentBatch.readings[batchIndex].gx = gx;
          currentBatch.readings[batchIndex].gy = gy;
          currentBatch.readings[batchIndex].gz = gz;

          batchIndex++;

          if (batchIndex >= BATCH_SIZE) {
            imuChar.writeValue((byte*)&currentBatch, sizeof(IMUBatch));
            batchIndex = 0;
          }
        }
      }
    }

    streaming = false;
    Serial.println("Disconnected");
  }
}
