#include <Wire.h>
#include "BMI270_Sensor.h"

// Tạo đối tượng cảm biến
BMI2_X bmi270;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=== BMI270 initialization ===");

  // Khởi tạo I2C
  Wire.begin();

  // Khởi tạo cảm biến BMI270
  if (bmi270.begin(BMI2_I2C_INTERFACE, BMI2_I2C_ADDR_PRIMARY) != BMI2_OK) {
    Serial.println("BMI270 initialization failed!");
    while (1);
  }
  Serial.println("BMI270 initialized successfully!");

  // Cấu hình cảm biến gia tốc
  struct bmi2_sens_config config;
  config.type = BMI2_ACCEL;

  // Thiết lập tần số lấy mẫu (ODR) 10 Hz
  config.cfg.acc.odr = BMI2_ACC_ODR_10HZ;

  // Thiết lập dải đo ±4g
  config.cfg.acc.range = BMI2_ACC_RANGE_4G;

  // Chế độ lọc và băng thông (tuỳ chọn)
  config.cfg.acc.bwp = BMI2_ACC_NORMAL_AVG4;

  // Ghi cấu hình vào cảm biến
  if (bmi2_set_sensor_config(&config, 1, &bmi270.dev) != BMI2_OK) {
    Serial.println("Failed to set accelerometer config!");
    while (1);
  }

  // Bật cảm biến gia tốc
  if (bmi2_sensor_enable(BMI2_ACCEL, &bmi270.dev) != BMI2_OK) {
    Serial.println("Failed to enable accelerometer!");
    while (1);
  }

  Serial.println("Accelerometer configured: ±4g @ 10 Hz");
}

void loop() {
  struct bmi2_sensor_data sensor_data = {0};

  // Đọc dữ liệu từ cảm biến
  if (bmi2_get_sensor_data(&sensor_data, 1, &bmi270.dev) == BMI2_OK) {
    float ax = sensor_data.acc.x * 4.0 / 32768.0;
    float ay = sensor_data.acc.y * 4.0 / 32768.0;
    float az = sensor_data.acc.z * 4.0 / 32768.0;

    Serial.print("Accel [g] -> X: ");
    Serial.print(ax, 3);
    Serial.print("  Y: ");
    Serial.print(ay, 3);
    Serial.print("  Z: ");
    Serial.println(az, 3);
  }

  delay(100); // ~10Hz
}
