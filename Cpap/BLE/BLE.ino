/*
  Arduino Nano 33 BLE Sense
  BLE Peripheral cho app điện thoại

  Điều kiện gửi data:
  ✅ Có kết nối BLE
  ✅ App đã bật notify/read characteristic

  Chức năng:
  - Gửi sensor data qua Notify
  - Nhận command từ app qua Write
*/

#include <ArduinoBLE.h>

// ================= UUID =================
BLEService cpapService("cb24858f-399f-4498-85e8-fea9d383d54f");

// ===== Sensor Characteristic =====
BLEStringCharacteristic sensorCharacteristic(
  "5e9e214b-124c-434d-84e5-018dccd35df1",
  BLERead | BLENotify,
  100
);

// ===== Action Characteristic =====
BLEStringCharacteristic actionCharacteristic(
  "56debc28-acab-4184-8f86-1a9c887b220a",
  BLEWrite,
  50
);

// ================= Variable =================
bool notifyEnabled = false;
unsigned long lastSend = 0;

// ================= Setup =================
void setup() {

  Serial.begin(115200);

  pinMode(LED_BUILTIN, OUTPUT);

  while (!Serial);

  // ===== BLE Init =====
  if (!BLE.begin()) {
    Serial.println("❌ BLE START FAILED");
    while (1);
  }

  // ===== BLE Device =====
  BLE.setLocalName("CPAP_VSSM");
  BLE.setDeviceName("CPAP_VSSM");

  // ===== Add Service =====
  BLE.setAdvertisedService(cpapService);

  cpapService.addCharacteristic(sensorCharacteristic);
  cpapService.addCharacteristic(actionCharacteristic);

  BLE.addService(cpapService);

  // ===== Default Value =====
  sensorCharacteristic.writeValue("ready");

  // ===== Advertising =====
  BLE.advertise();

  Serial.println("🚀 BLE Peripheral Started");
}

// ================= Loop =================
void loop() {

  // Chờ central connect
  BLEDevice central = BLE.central();

  if (central) {

    Serial.print("📱 CONNECTED: ");
    Serial.println(central.address());

    while (central.connected()) {

      // Quan trọng
      BLE.poll();

      // =========================================
      // CHECK APP ĐÃ ENABLE NOTIFY CHƯA
      // =========================================
      notifyEnabled = sensorCharacteristic.subscribed();

      if (notifyEnabled) {

        // Gửi mỗi 2 giây
        if (millis() - lastSend >= 2000) {

          lastSend = millis();

          // ===== Fake Sensor =====
          float temperature = random(300, 380) / 10.0;
          int spo2 = random(94, 100);

          // ===== JSON =====
          String sensorData =
            "{"
            "\"temperature\":" + String(temperature) + "," +
            "\"spo2\":" + String(spo2) +
            "}";

          // ===== SEND NOTIFY =====
          bool success = sensorCharacteristic.writeValue(sensorData);

          if (success) {
            Serial.print("📤 SEND: ");
            Serial.println(sensorData);
          } else {
            Serial.println("❌ SEND FAILED");
          }
        }
      } else {
        Serial.println("⏳ WAITING FOR NOTIFY...");
        delay(500);
      }

      // =========================================
      // RECEIVE ACTION
      // =========================================
      if (actionCharacteristic.written()) {

        String action = actionCharacteristic.value();

        Serial.print("📥 ACTION: ");
        Serial.println(action);

        // ===== LED =====
        if (action == "LED_ON") {

          digitalWrite(LED_BUILTIN, HIGH);
          Serial.println("💡 LED ON");
        }

        else if (action == "LED_OFF") {

          digitalWrite(LED_BUILTIN, LOW);
          Serial.println("💡 LED OFF");
        }

        // ===== START =====
        else if (action == "START") {

          Serial.println("▶ START");
        }

        // ===== STOP =====
        else if (action == "STOP") {

          Serial.println("⏹ STOP");
        }
      }

      delay(10);
    }

    Serial.println("❌ DISCONNECTED");

    notifyEnabled = false;
  }
}