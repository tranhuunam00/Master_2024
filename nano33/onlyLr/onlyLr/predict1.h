#ifndef PREDICT1_H
#define PREDICT1_H

#include <math.h>
#include "model1.h"

// Hàm dự đoán cho Logistic Regression
// Input: Mảng features đã được chuẩn hóa (normalized)
// Output: 1 (Nằm), 0 (Không nằm)
int predict1(float f[])
{
  float logit = bias1;

  for (int i = 0; i < NUM_FEATURES_1; i++)
  {
    logit += weights1[i] * f[i];
  }

  float p = 1.0 / (1.0 + exp(-logit)); // sigmoid activation

  return (p > 0.5) ? 1 : 0; // Threshold 0.5
}

#endif