#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"

BMI270 imu;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Wire1.begin();
  Serial.println("Init BMI270 on Nano 33 BLE Sense (Wire1)");

  if (imu.beginI2C(BMI2_I2C_PRIM_ADDR, Wire1) != BMI2_OK) {
    Serial.println("Error: BMI270 not detected!");
    while (1);
  }

  Serial.println("BMI270 connected!");

  //--- Cấu hình gia tốc ---
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

static float prevX = 0, prevY = 0, prevZ = 0;
void loop() {
  imu.getSensorData();
  float dx = abs(imu.data.accelX - prevX);
  float dy = abs(imu.data.accelY - prevY);
  float dz = abs(imu.data.accelZ - prevZ);
  if (dx > 0.08 || dy > 0.08 || dz > 0.08) {

    Serial.print(imu.data.accelX, 5);
    Serial.print(',');
    Serial.print(imu.data.accelY, 5);
    Serial.print(',');
    Serial.println(imu.data.accelZ, 5);
  }
  delay(100);
}
