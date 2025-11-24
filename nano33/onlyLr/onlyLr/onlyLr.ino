#include "model.h" // model2 – 15 features – phân loại 4 tư thế
#include "predict.h"

#include "model1.h" // model1 – 6 features – nằm / không nằm
#include "predict1.h"

#include <math.h>
#include <algorithm>
#include "SparkFun_BMI270_Arduino_Library.h"

#include <Wire.h>

#define WINDOW_SIZE 21
#define NUM_FEATURES2 15 // model2
#define NUM_FEATURES1 6  // model1

// LED mapping
#define LED_BACK 3
#define LED_LEFT 5
#define LED_RIGHT 17
#define LED_STOMACH 16
#define LED_OTHER 13

float x[WINDOW_SIZE], y[WINDOW_SIZE], z[WINDOW_SIZE];
float f2[NUM_FEATURES2]; // features cho 4 tư thế
float f1[NUM_FEATURES1]; // features cho nằm/không nằm
float norm2[NUM_FEATURES2];
float norm1[NUM_FEATURES1];

void setPostureLED(int pred)
{
  digitalWrite(LED_BACK, LOW);
  digitalWrite(LED_LEFT, LOW);
  digitalWrite(LED_RIGHT, LOW);
  digitalWrite(LED_STOMACH, LOW);
  digitalWrite(LED_OTHER, LOW);

  switch (pred)
  {
  case 0:
    digitalWrite(LED_BACK, HIGH);
    break;
  case 1:
    digitalWrite(LED_LEFT, HIGH);
    break;
  case 2:
    digitalWrite(LED_RIGHT, HIGH);
    break;
  case 3:
    digitalWrite(LED_STOMACH, HIGH);
    break;
  default:
    digitalWrite(LED_OTHER, HIGH);
    break;
  }
}

float median(float *arr, int size)
{
  float temp[WINDOW_SIZE];
  memcpy(temp, arr, sizeof(float) * size);
  std::sort(temp, temp + size);
  return (size % 2 == 0)
             ? (temp[size / 2 - 1] + temp[size / 2]) * 0.5f
             : temp[size / 2];
}

float stddev(float *arr, int size, float mean)
{
  float sum = 0;
  for (int i = 0; i < size; i++)
    sum += (arr[i] - mean) * (arr[i] - mean);
  return sqrt(sum / size);
}

void extractFeatures(float *x, float *y, float *z)
{
  // ---------- TÍNH CÁC THỐNG KÊ CHUNG ----------
  float sum_x = 0, sum_y = 0, sum_z = 0;
  float energy_x = 0, energy_y = 0, energy_z = 0;
  float sma = 0;
  int pos_x = 0, neg_x = 0, pos_z = 0, neg_z = 0;

  float y_min = 999, y_max = -999;
  float z_min = 999, z_max = -999;

  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    sum_x += x[i];
    sum_y += y[i];
    sum_z += z[i];

    energy_x += x[i] * x[i];
    energy_y += y[i] * y[i];
    energy_z += z[i] * z[i];

    sma += (fabs(x[i]) + fabs(y[i]) + fabs(z[i])) / WINDOW_SIZE;

    if (x[i] > 0)
      pos_x++;
    else
      neg_x++;
    if (z[i] > 0)
      pos_z++;
    else
      neg_z++;

    if (y[i] < y_min)
      y_min = y[i];
    if (y[i] > y_max)
      y_max = y[i];

    if (z[i] < z_min)
      z_min = z[i];
    if (z[i] > z_max)
      z_max = z[i];
  }

  float mean_x = sum_x / WINDOW_SIZE;
  float mean_y = sum_y / WINDOW_SIZE;
  float mean_z = sum_z / WINDOW_SIZE;

  float avg_norm = 0;
  for (int i = 0; i < WINDOW_SIZE; i++)
    avg_norm += sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
  avg_norm /= WINDOW_SIZE;

  // ===================================================
  //      MODEL 2 – 15 FEATURES (ngửa – trái – phải – sấp)
  // ===================================================
  f2[0] = median(z, WINDOW_SIZE);          // z_median
  f2[1] = mean_z;                          // z_mean
  f2[2] = mean_x;                          // x_mean
  f2[3] = energy_x / WINDOW_SIZE;          // x_energy
  f2[4] = pos_z;                           // z_pos_count
  f2[5] = neg_z;                           // z_neg_count
  f2[6] = median(x, WINDOW_SIZE);          // x_median
  f2[7] = energy_z / WINDOW_SIZE;          // z_energy
  f2[8] = avg_norm;                        // avg_result_accl
  f2[9] = neg_x;                           // x_neg_count
  f2[10] = stddev(z, WINDOW_SIZE, mean_z); // z_std
  f2[11] = pos_x;                          // x_pos_count
  f2[12] = energy_y / WINDOW_SIZE;         // y_energy
  f2[13] = mean_y;                         // y_mean
  f2[14] = sma;                            // sma

  // ===================================================
  //      MODEL 1 – 6 FEATURES (nằm / không nằm)
  // ===================================================
  f1[0] = avg_norm;               // avg_result_accl
  f1[1] = energy_y / WINDOW_SIZE; // y_energy
  f1[2] = y_min;                  // y_min
  f1[3] = mean_y;                 // y_mean
  f1[4] = z_min;                  // z_min
  f1[5] = z_max;                  // z_max
}

// ------- SCALE 6 FEATURES CHO MODEL 1 -------
void normalizeFeatures1(float *in, float *out)
{
  for (int i = 0; i < NUM_FEATURES1; i++)
    out[i] = 2.0 * (in[i] - min_vals1[i]) / (max_vals1[i] - min_vals1[i]) - 1.0;
}

// ------- SCALE 15 FEATURES CHO MODEL 2 -------
void normalizeFeatures2(float *in, float *out)
{
  for (int i = 0; i < NUM_FEATURES2; i++)
    out[i] = 2.0 * (in[i] - min_vals[i]) / (max_vals[i] - min_vals[i]) - 1.0;
}

BMI270 imu;

void setup()
{
  Serial.begin(115200);

  Wire1.begin();

  pinMode(LED_BACK, OUTPUT);
  pinMode(LED_LEFT, OUTPUT);
  pinMode(LED_RIGHT, OUTPUT);
  pinMode(LED_STOMACH, OUTPUT);
  pinMode(LED_OTHER, OUTPUT);

  if (imu.beginI2C(BMI2_I2C_PRIM_ADDR, Wire1) != BMI2_OK)
  {
    Serial.println("Error: BMI270 not detected!");
    while (1)
      ;
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
  if (imu.getConfig(&config) == BMI2_OK)
  {
    Serial.println("Current Accelerometer Config:");
    Serial.print("  Range setting = ");
    switch (config.cfg.acc.range)
    {
    case BMI2_ACC_RANGE_2G:
      Serial.println("±2g");
      break;
    case BMI2_ACC_RANGE_4G:
      Serial.println("±4g");
      break;
    case BMI2_ACC_RANGE_8G:
      Serial.println("±8g");
      break;
    case BMI2_ACC_RANGE_16G:
      Serial.println("±16g");
      break;
    default:
      Serial.println("Unknown");
      break;
    }

    Serial.print("  ODR setting = ");
    switch (config.cfg.acc.odr)
    {
    case BMI2_ACC_ODR_0_78HZ:
      Serial.println("0.78 Hz");
      break;
    case BMI2_ACC_ODR_1_56HZ:
      Serial.println("1.56 Hz");
      break;
    case BMI2_ACC_ODR_3_12HZ:
      Serial.println("3.12 Hz");
      break;
    case BMI2_ACC_ODR_6_25HZ:
      Serial.println("6.25 Hz");
      break;
    case BMI2_ACC_ODR_12_5HZ:
      Serial.println("12.5 Hz");
      break;
    case BMI2_ACC_ODR_25HZ:
      Serial.println("25 Hz");
      break;
    case BMI2_ACC_ODR_50HZ:
      Serial.println("50 Hz");
      break;
    case BMI2_ACC_ODR_100HZ:
      Serial.println("100 Hz");
      break;
    default:
      Serial.println("Unknown");
      break;
    }
  }
  else
  {
    Serial.println("Failed to read accelerometer config!");
  }

  Serial.println("=== Sleep posture detection ===");
}

int sampleIndex = 0;

void loop()
{
  float ax, ay, az;

  {
    imu.getSensorData();

    ax = imu.data.accelX;
    ay = imu.data.accelY;
    az = imu.data.accelZ;

    x[sampleIndex] = ax;
    y[sampleIndex] = ay;
    z[sampleIndex] = az;
    sampleIndex++;

    if (sampleIndex == WINDOW_SIZE)
    {
      extractFeatures(x, y, z);

      normalizeFeatures1(f1, norm1);
      int lying = predict1(norm1);

      if (lying == 0)
      {
        setPostureLED(-1);
        Serial.println("Not lying → skip model2");
      }
      else
      {
        normalizeFeatures2(f2, norm2);
        int pred = predict(norm2);
        setPostureLED(pred);

        const char *labels[4] = {"Back", "Left", "Right", "Stomach"};

        Serial.print("Posture: ");
        Serial.println(labels[pred]);
      }

      sampleIndex = 0;
      delay(300);
    }
  }

}
