
#include <ArduinoBLE.h>
#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"




const char* deviceServiceUuid = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10001-e8f2-537e-4f6c-d104768a1214";

BLEService accelerometerService(deviceServiceUuid);
BLECharacteristic accelerometerCharacteristic(
  deviceServiceCharacteristicUuid,
  BLERead | BLEWrite | BLENotify,
  9, 3
);
BMI270 imu;

void setup() {
  Serial.begin(9600);

  Serial.println("Started");

  Wire1.begin();
  

  if (!BLE.begin()) {
    Serial.println("- Starting Bluetooth® Low Energy module failed!");
    while (1)
      ;
  }

  BLE.setLocalName("Master_2025_BLE");
  BLE.setDeviceName("Master_2025_BLE");


  BLE.setAdvertisedService(accelerometerService);

  accelerometerService.addCharacteristic(accelerometerCharacteristic);
  BLE.addService(accelerometerService);
  accelerometerCharacteristic.canSubscribe();
  accelerometerCharacteristic.subscribed();

  BLE.advertise();

  Serial.println("Nano 33 BLE (Peripheral Device)");
  Serial.println(" ");

  if (imu.beginI2C(BMI2_I2C_PRIM_ADDR, Wire1) != BMI2_OK) {
    Serial.println("Error: BMI270 not detected!");
    while (1);
  }

  struct bmi2_sens_config config;
  config.type = BMI2_ACCEL;
  config.cfg.acc.odr = BMI2_ACC_ODR_12_5HZ;
  config.cfg.acc.range = BMI2_ACC_RANGE_2G;
  config.cfg.acc.bwp = BMI2_ACC_NORMAL_AVG4;
  config.cfg.acc.filter_perf = BMI2_POWER_OPT_MODE;

  if (imu.setConfig(config) == BMI2_OK)
    Serial.println("Accelerometer configured successfully!");
  else
    Serial.println("Failed to configure accelerometer!");

  delay(200);

  //--- Đọc lại cấu hình ---
  if (imu.getConfig(&config) == BMI2_OK) {
    Serial.println("Current Accelerometer Config:");
    Serial.print("  Range setting = ");
    switch (config.cfg.acc.range) {
      case BMI2_ACC_RANGE_2G:  Serial.println("±2g"); break;
      case BMI2_ACC_RANGE_4G:  Serial.println("±4g"); break;
      case BMI2_ACC_RANGE_8G:  Serial.println("±8g"); break;
      case BMI2_ACC_RANGE_16G: Serial.println("±16g"); break;
      default: Serial.println("Unknown"); break;
    }

    Serial.print("  ODR setting = ");
    switch (config.cfg.acc.odr) {
      case BMI2_ACC_ODR_0_78HZ: Serial.println("0.78 Hz"); break;
      case BMI2_ACC_ODR_1_56HZ: Serial.println("1.56 Hz"); break;
      case BMI2_ACC_ODR_3_12HZ: Serial.println("3.12 Hz"); break;
      case BMI2_ACC_ODR_6_25HZ: Serial.println("6.25 Hz"); break;
      case BMI2_ACC_ODR_12_5HZ: Serial.println("12.5 Hz"); break;
      case BMI2_ACC_ODR_25HZ:   Serial.println("25 Hz"); break;
      case BMI2_ACC_ODR_50HZ:   Serial.println("50 Hz"); break;
      case BMI2_ACC_ODR_100HZ:  Serial.println("100 Hz"); break;
      default: Serial.println("Unknown"); break;
    }
  } else {
    Serial.println("Failed to read accelerometer config!");
  }
}

static float x, y, z;

void loop() {
  BLEDevice central = BLE.central();
  Serial.println("- Discovering central device...");
  delay(500);

  if (central) {
    Serial.println("* Connected to central device!");
    Serial.print("* Device MAC address: ");
    Serial.println(central.address());
    Serial.println(" ");

    while (central.connected()) {
      imu.getSensorData();
      imu.getSensorData();
      x = imu.data.accelX;
      y = imu.data.accelY;
      z = imu.data.accelZ;


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
      

      delay(100);  // Điều chỉnh tốc độ gửi Notify
    }
  }
  

  
}