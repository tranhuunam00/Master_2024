#include "model.h"
#include "predict.h"
#include <math.h>
#include <algorithm>
#include "SparkFun_BMI270_Arduino_Library.h"
#include <ArduinoBLE.h>
#include <Wire.h>


#define WINDOW_SIZE 10
#define NUM_FEATURES 15
const char* deviceServiceUuid = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10001-e8f2-537e-4f6c-d104768a1214";

BLEService accelerometerService(deviceServiceUuid);
BLECharacteristic accelerometerCharacteristic(
  deviceServiceCharacteristicUuid,
  BLERead | BLEWrite | BLENotify,
  9, 3
);

BMI270 imu;

float x[WINDOW_SIZE], y[WINDOW_SIZE], z[WINDOW_SIZE];
float features[NUM_FEATURES];
float normalized[NUM_FEATURES];

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

static float xData, yData, zData;

int sampleIndex = 0;

void loop() {
  float ax, ay, az;

  imu.getSensorData();
  xData = imu.data.accelX;
  yData = imu.data.accelY;
  zData = imu.data.accelZ;

  if (sampleIndex < WINDOW_SIZE ) {
    x[sampleIndex] = xData;
    y[sampleIndex] = yData;
    z[sampleIndex] = zData;
    sampleIndex++;
  }
  if (sampleIndex == WINDOW_SIZE) {
    unsigned long t_start = micros();
    extractFeatures(x, y, z, features);
    normalizeFeatures(features, normalized);

    int pred = predict(normalized);

    unsigned long t_end = micros();
    unsigned long duration = t_end - t_start;

    const char* labels[NUM_CLASSES] = {"Ngửa", "Nghiêng phải", "Nghiêng trái", "Sấp"};
    Serial.print("[Predict] Posture: ");
    Serial.print(pred + 1);
    Serial.print(" → ");
    Serial.println(labels[pred]);

    Serial.print(" | Thời gian xử lý: ");
    Serial.print(duration);
    Serial.println(" µs");


    sampleIndex = 0; 
    delay(500);
  }
}

float median(float* arr, int size) {
  float temp[WINDOW_SIZE];
  memcpy(temp, arr, sizeof(float) * size);
  std::sort(temp, temp + size);
  if (size % 2 == 0)
    return (temp[size / 2 - 1] + temp[size / 2]) / 2.0;
  else
    return temp[size / 2];
}

float stddev(float* arr, int size, float mean) {
  float sum = 0.0;
  for (int i = 0; i < size; i++) {
    float diff = arr[i] - mean;
    sum += diff * diff;
  }
  return sqrt(sum / size);
}

void extractFeatures(float* x, float* y, float* z, float* out) {
  float sum_x = 0, sum_y = 0, sum_z = 0;
  float energy_x = 0, energy_y = 0, energy_z = 0;
  float sma = 0;
  int pos_x = 0, neg_x = 0, pos_z = 0, neg_z = 0;

  for (int i = 0; i < WINDOW_SIZE; i++) {
    sum_x += x[i];
    sum_y += y[i];
    sum_z += z[i];

    energy_x += x[i] * x[i];
    energy_y += y[i] * y[i];
    energy_z += z[i] * z[i];

    sma += (fabs(x[i]) + fabs(y[i]) + fabs(z[i])) / WINDOW_SIZE;

    if (x[i] > 0) pos_x++; else neg_x++;
    if (z[i] > 0) pos_z++; else neg_z++;
  }

  float mean_x = sum_x / WINDOW_SIZE;
  float mean_y = sum_y / WINDOW_SIZE;
  float mean_z = sum_z / WINDOW_SIZE;

  out[0] = median(z, WINDOW_SIZE); 
  out[1] = mean_z;                 
  out[2] = mean_x;                
  out[3] = energy_x / WINDOW_SIZE;
  out[4] = pos_z;                 // z_pos_count
  out[5] = neg_z;                 // z_neg_count
  out[6] = median(x, WINDOW_SIZE);// x_median
  out[7] = energy_z / WINDOW_SIZE;// z_energy
  float sum_norm = 0.0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    sum_norm += sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
  }
  out[8] = sum_norm / WINDOW_SIZE;

  out[9] = neg_x;                 // x_neg_count
  out[10] = stddev(z, WINDOW_SIZE, mean_z); // z_std
  out[11] = pos_x;                // x_pos_count
  out[12] = energy_y / WINDOW_SIZE; // y_energy
  out[13] = mean_y;               // y_mean
  out[14] = sma;                  // sma
}

void normalizeFeatures(float* in, float* out) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    if ((max_vals[i] - min_vals[i]) != 0) {
      out[i] = 2.0 * (in[i] - min_vals[i]) / (max_vals[i] - min_vals[i]) - 1.0;
    } else {
      out[i] = 0.0;
    }
  }
}
