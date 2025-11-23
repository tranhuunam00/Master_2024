#ifndef PREDICT1_H
#define PREDICT1_H

#include <math.h>
#include "model1.h"

int predict1(float f[])
{
  float logit = bias1;

  for (int i = 0; i < NUM_FEATURES_1; i++)
  {
    logit += weights1[i] * f[i];
  }

  float p = 1.0 / (1.0 + exp(-logit)); // sigmoid

  return (p > 0.5) ? 1 : 0; // 1 = nằm, 0 = không nằm
}

#endif
