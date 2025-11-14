#ifndef PREDICT_H
#define PREDICT_H

#include <math.h>
#include "model.h"

int predict(float features[]) {
  // Compute logits (linear combination with weights)
  float logits[NUM_CLASSES] = {0};
  for (int i = 0; i < NUM_CLASSES; i++) {
    logits[i] = biases[i];
    for (int j = 0; j < NUM_FEATURES; j++) {
      logits[i] += weights[i][j] * features[j];
    }
  }

  // Softmax with numerical stability
  float max_logit = logits[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  float exp_sum = 0.0;
  float probs[NUM_CLASSES];
  for (int i = 0; i < NUM_CLASSES; i++) {
    probs[i] = exp(logits[i] - max_logit);
    exp_sum += probs[i];
  }
  for (int i = 0; i < NUM_CLASSES; i++) {
    probs[i] /= exp_sum;
  }

  // Debug (optional): Print logits and probabilities
#ifdef DEBUG
  Serial.println("Logits:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(logits[i], 4);
  }

  Serial.println("Probabilities:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(probs[i], 4);
  }
#endif

  // Argmax
  int predicted = 0;
  float max_prob = probs[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      predicted = i;
    }
  }

  return predicted;
}

#endif
