
// #include <Arduino_LSM9DS1.h>
#include "Arduino_BMI270_BMM150.h"
#include <ArduinoBLE.h>


const char* deviceServiceUuid = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10001-e8f2-537e-4f6c-d104768a1214";

BLEService accelerometerService(deviceServiceUuid);
BLECharacteristic accelerometerCharacteristic(
  deviceServiceCharacteristicUuid,
  BLERead | BLEWrite | BLENotify,
  9, 3
);

void setup() {
  Serial.begin(9600);

  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }

  if (!BLE.begin()) {
    Serial.println("- Starting Bluetooth® Low Energy module failed!");
    while (1)
      ;
  }

  BLE.setLocalName("Arduino Nano 33 BLE (Peripheral) Cua Nam");
  BLE.setDeviceName("Arduino Nano 33 BLE (Peripheral) Cua Nam");


  BLE.setAdvertisedService(accelerometerService);

  accelerometerService.addCharacteristic(accelerometerCharacteristic);
  BLE.addService(accelerometerService);
  accelerometerCharacteristic.canSubscribe();
  accelerometerCharacteristic.subscribed();

  BLE.advertise();

  Serial.println("Nano 33 BLE (Peripheral Device)");
  Serial.println(" ");

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Acceleration in G's");
  Serial.println("X\tY\tZ");
}

void loop() {
  float x, y, z;
  BLEDevice central = BLE.central();
  Serial.println("- Discovering central device...");
  delay(500);

  if (central) {
    Serial.println("* Connected to central device!");
    Serial.print("* Device MAC address: ");
    Serial.println(central.address());
    Serial.println(" ");

    while (central.connected()) {
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(x, y, z);

        Serial.print(x);
        Serial.print('\t');
        Serial.print(y);
        Serial.print('\t');
        Serial.println(z);

        // int8_t accelDataX = static_cast<int8_t>(x * 10 +  120);  // Giả sử gửi giá trị X nhân 10 + 120
        // int8_t accelDataY = static_cast<int8_t>(y * 10 +  120);  // Giả sử gửi giá trị X nhân 10 + 120
        // int8_t accelDataZ = static_cast<int8_t>(z * 10 +  120);  // Giả sử gửi giá trị X nhân 10 + 120
        // accelerometerCharacteristic.writeValue(accelDataX); // Cập nhật giá trị mới
        uint8_t accelData[9] = {
            (x >= 0) ? 1 : 0,  // Sign (1 for positive, 0 for negative)
            abs((int)x),        // Integer part
            abs((int)(x * 100) % 100), // Decimal part (scaled by 100)

            (y >= 0) ? 1 : 0,
            abs((int)y),
            abs((int)(y * 100) % 100),

            (z >= 0) ? 1 : 0,
            abs((int)z),
            abs((int)(z * 100) % 100)
        };


        accelerometerCharacteristic.writeValue(accelData, sizeof(accelData)); 
      }

      delay(100);  // Điều chỉnh tốc độ gửi Notify
    }
  }
  

  
}