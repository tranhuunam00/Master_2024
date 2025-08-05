#include "model.h"
#include "predict.h"

float test_input[NUM_FEATURES] = {
  0.5, -0.2, 0.1, 0.3, 5, 15, -0.1, 0.25, 1.0,
  10, 0.2, 10, 0.4, -0.1, 1.2, 0.0
};

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=== Testing model with fixed input ===");

  float normalized[NUM_FEATURES];

  for (int i = 0; i < NUM_FEATURES; i++) {
    if ((max_vals[i] - min_vals[i]) != 0) {
      normalized[i] = 2.0 * (test_input[i] - min_vals[i]) / (max_vals[i] - min_vals[i]) - 1.0;
    } else {
      normalized[i] = 0.0;
    }
  }

  int predicted = predict(normalized);

  const char* labels[NUM_CLASSES] = {"Back", "Stomach", "Left", "Right"};
  Serial.print("Predicted label (1–4): ");
  Serial.print(predicted + 1);
  Serial.print(" → ");
  Serial.println(labels[predicted]);
}

void loop() {
  // nothing
}
