#include "model.h"
#include "predict.h"
#include <math.h>
#include <algorithm>
#include <Wire.h>

#define WINDOW_SIZE 10
#define NUM_FEATURES 15

float x[WINDOW_SIZE], y[WINDOW_SIZE], z[WINDOW_SIZE];
float features[NUM_FEATURES];
float normalized[NUM_FEATURES];

String line = "";
int sampleIndex = 0;

void setup()
{
  Serial.begin(115200);
  Serial.println("Started");
  while (!Serial)
    ;
}

void loop()
{
  if (!Serial.available())
    return;

  char c = Serial.read();

  if (c != '\n')
  {
    line += c;
    return;
  }

  line.trim();

  int p1 = line.indexOf(',');
  int p2 = line.indexOf(',', p1 + 1);

  if (p1 < 0 || p2 < 0)
  {
    Serial.println("ERR_FORMAT");
    line = "";
    return;
  }

  float xData = line.substring(0, p1).toFloat();
  float yData = line.substring(p1 + 1, p2).toFloat();
  float zData = line.substring(p2 + 1).toFloat();

  x[sampleIndex] = xData;
  y[sampleIndex] = yData;
  z[sampleIndex] = zData;
  sampleIndex++;

  // Nếu đủ cửa sổ 10 mẫu → chạy model
  if (sampleIndex == WINDOW_SIZE)
  {
    unsigned long t_start = micros();
    extractFeatures(x, y, z, features);
    normalizeFeatures(features, normalized);
    int pred = predict(normalized);
    unsigned long duration = micros() - t_start;

    Serial.println(pred + 1);

    sampleIndex = 0;
  }

  line = ""; // reset buffer cho dòng tiếp theo
}

float median(float *arr, int size)
{
  float temp[WINDOW_SIZE];
  memcpy(temp, arr, sizeof(float) * size);
  std::sort(temp, temp + size);
  if (size % 2 == 0)
    return (temp[size / 2 - 1] + temp[size / 2]) / 2.0;
  else
    return temp[size / 2];
}

float stddev(float *arr, int size, float mean)
{
  float sum = 0.0;
  for (int i = 0; i < size; i++)
  {
    float diff = arr[i] - mean;
    sum += diff * diff;
  }
  return sqrt(sum / size);
}

void extractFeatures(float *x, float *y, float *z, float *out)
{
  float sum_x = 0, sum_y = 0, sum_z = 0;
  float energy_x = 0, energy_y = 0, energy_z = 0;
  float sma = 0;
  int pos_x = 0, neg_x = 0, pos_z = 0, neg_z = 0;

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
  }

  float mean_x = sum_x / WINDOW_SIZE;
  float mean_y = sum_y / WINDOW_SIZE;
  float mean_z = sum_z / WINDOW_SIZE;

  out[0] = median(z, WINDOW_SIZE);
  out[1] = mean_z;
  out[2] = mean_x;
  out[3] = energy_x / WINDOW_SIZE;
  out[4] = pos_z;                  // z_pos_count
  out[5] = neg_z;                  // z_neg_count
  out[6] = median(x, WINDOW_SIZE); // x_median
  out[7] = energy_z / WINDOW_SIZE; // z_energy
  float sum_norm = 0.0;
  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    sum_norm += sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
  }
  out[8] = sum_norm / WINDOW_SIZE;

  out[9] = neg_x;                           // x_neg_count
  out[10] = stddev(z, WINDOW_SIZE, mean_z); // z_std
  out[11] = pos_x;                          // x_pos_count
  out[12] = energy_y / WINDOW_SIZE;         // y_energy
  out[13] = mean_y;                         // y_mean
  out[14] = sma;                            // sma
}

void normalizeFeatures(float *in, float *out)
{
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    if ((max_vals[i] - min_vals[i]) != 0)
    {
      out[i] = 2.0 * (in[i] - min_vals[i]) / (max_vals[i] - min_vals[i]) - 1.0;
    }
    else
    {
      out[i] = 0.0;
    }
  }
}
