#include "model.h"
#include "predict.h"
#include <math.h>
#include <algorithm>
#include "Arduino_BMI270_BMM150.h"


#define WINDOW_SIZE 21
#define NUM_FEATURES 16

float x[WINDOW_SIZE], y[WINDOW_SIZE], z[WINDOW_SIZE];
float features[NUM_FEATURES];
float normalized[NUM_FEATURES];

void setup() {
  Serial.begin(115200);
  while (!Serial);


 if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }
  Serial.println("=== Real-time Sleep Posture Detection ===");
}

int sampleIndex = 0;

void loop() {
  float ax, ay, az;

  if (IMU.accelerationAvailable()) {
   

    IMU.readAcceleration(ax, ay, az);
    if (sampleIndex < WINDOW_SIZE) {
      x[sampleIndex] = ax;
      y[sampleIndex] = ay;
      z[sampleIndex] = az;
      sampleIndex++;
    }
    if (sampleIndex == WINDOW_SIZE) {
      extractFeatures(x, y, z, features);
      normalizeFeatures(features, normalized);

      int pred = predict(normalized);
      const char* labels[NUM_CLASSES] = {"Back", "Left", "Right", "Stomach"};
      Serial.print("[Predict] Posture: ");
      Serial.print(pred + 1);
      Serial.print(" → ");
      Serial.println(labels[pred]);

      sampleIndex = 0;  // reset lại để lấy 21 mẫu tiếp theo
      delay(500);       // nghỉ 0.5s trước chu kỳ mới
    }
  }
  // nothing
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

  out[0] = median(z, WINDOW_SIZE); // z_median
  out[1] = mean_z;                 // z_mean
  out[2] = mean_x;                 // x_mean
  out[3] = energy_x / WINDOW_SIZE;// x_energy
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
  out[15] = median(y, WINDOW_SIZE); // y_median
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
