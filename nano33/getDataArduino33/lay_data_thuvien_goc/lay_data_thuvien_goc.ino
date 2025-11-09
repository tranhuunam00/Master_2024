#include <Wire.h>
#include "SparkFun_BMI270_Arduino_Library.h"

BMI270 imu;

//---------------------------------------------
// Cấu trúc Kalman Filter 1 chiều
//---------------------------------------------
struct KalmanFilter {
  float Q;   // process noise covariance
  float R;   // measurement noise covariance
  float x;   // estimated value
  float P;   // estimation error covariance
  float K;   // Kalman gain
};

void kalmanInit(KalmanFilter &kf, float Q, float R, float initValue) {
  kf.Q = Q;
  kf.R = R;
  kf.x = initValue;
  kf.P = 1.0;
  kf.K = 0.0;
}

float kalmanUpdate(KalmanFilter &kf, float measurement) {
  // Prediction update
  kf.P += kf.Q;
  // Measurement update
  kf.K = kf.P / (kf.P + kf.R);
  kf.x = kf.x + kf.K * (measurement - kf.x);
  kf.P = (1 - kf.K) * kf.P;
  return kf.x;
}

//---------------------------------------------
// Kalman cho từng trục
//---------------------------------------------
KalmanFilter kfx, kfy, kfz;

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
  imu.setConfig(config);

  //--- Khởi tạo Kalman ---
  kalmanInit(kfx, 0.02, 0.03, 0.0);
  kalmanInit(kfy, 0.02, 0.03, 0.0);
  kalmanInit(kfz, 0.02, 0.03, 0.0);

  Serial.println("System ready. Printing raw vs filtered data...");
  Serial.println("RawX,RawY,RawZ,KalX,KalY,KalZ");
}

//---------------------------------------------
void loop() {
  imu.getSensorData();
  float x = imu.data.accelX;
  float y = imu.data.accelY;
  float z = imu.data.accelZ;

  float x_kal = kalmanUpdate(kfx, x);
  float y_kal = kalmanUpdate(kfy, y);
  float z_kal = kalmanUpdate(kfz, z);

  // In dữ liệu cho Serial Plotter / Python
  // Serial.print(x, 3); Serial.print(",");
  // Serial.print(y, 3); Serial.print(",");
  Serial.print(z, 3); Serial.print(",");
  // Serial.print(x_kal, 3); Serial.print(",");
  // Serial.print(y_kal, 3); Serial.print(",");
  Serial.println(z_kal, 3);

  delay(100); // 10 Hz
}
