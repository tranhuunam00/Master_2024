#include "model.h"
#include "predict.h"
#include <math.h>
#include <algorithm>

#define WINDOW_SIZE 21
#define NUM_FEATURES 16

float x[WINDOW_SIZE], y[WINDOW_SIZE], z[WINDOW_SIZE];
float features[NUM_FEATURES];
float normalized[NUM_FEATURES];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=== Feature Extraction & Normalization Test ===");

  // Gán dữ liệu mẫu
  float sample_data[WINDOW_SIZE][3] = {
    {-0.03870, -0.00964, 1.00513},
    {-0.03662, -0.00769, 1.01428},
    {-0.03650, -0.01135, 1.01733},
    {-0.04089, -0.01587, 1.00879},
    {-0.03772, -0.01929, 1.00708},
    {-0.03271, -0.00488, 1.00806},
    {-0.04187, -0.01685, 1.02344},
    {-0.03625, -0.00244, 1.01563},
    {-0.03967, -0.01086, 0.99060},
    {-0.04333, -0.00854, 1.01392},
    {-0.04578, -0.01831, 1.00610},
    {-0.04431, -0.01685, 1.01990},
    {-0.05310, -0.02246, 1.01855},
    {-0.05310,  0.00452, 1.00354},
    {-0.05957,  0.02832, 0.99792},
    {-0.05579,  0.09863, 0.96973},
    {-0.05603,  0.07727, 1.00452},
    {-0.06189,  0.05505, 0.98987},
    {-0.05798, -0.00525, 1.00037},
    {-0.05579, -0.03113, 1.00171}
  };


  for (int i = 0; i < WINDOW_SIZE; i++) {
    x[i] = sample_data[i][0];
    y[i] = sample_data[i][1];
    z[i] = sample_data[i][2];
  }

  extractFeatures(x, y, z, features);
  normalizeFeatures(features, normalized);

  Serial.println("--- Features ---");
  for (int i = 0; i < NUM_FEATURES; i++) {
    Serial.print("features[");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(features[i], 6);
  }

  Serial.println("--- Normalized ---");
  for (int i = 0; i < NUM_FEATURES; i++) {
    Serial.print("normalized[");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(normalized[i], 6);
  }

  int pred = predict(normalized);
  const char* labels[NUM_CLASSES] = {"Back", "Left", "Right", "Stomach"};
  Serial.print("Predicted: ");
  Serial.print(pred + 1);
  Serial.print(" → ");
  Serial.println(labels[pred]);
}

void loop() {
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
