/*
  IMU Classifier - TensorFlow Lite for Arduino
  Updated for compatibility with model trained on (x, y, z) acceleration data only.
  Now applies new normalization: (x + 12) / 24.
  
  Compatible with:
  - Arduino Nano 33 BLE
  - Arduino Nano 33 BLE Sense Rev2
*/

#include "Arduino_BMI270_BMM150.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

const float accelerationThreshold = 2.5; // Motion detection threshold (G)
const int numSamples = 119;  // Number of IMU samples per inference
int samplesRead = numSamples;

// TensorFlow Lite globals
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Allocate memory for TensorFlow Lite model
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Gesture labels
const char* GESTURES[] = {
  "ngửa", "nghiêng trái", "nghiêng phải", "sấp"
};
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize IMU sensor
  if (!IMU.begin()) {
    Serial.println("IMU initialization failed!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  // Load TensorFlow Lite model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Initialize interpreter
  static tflite::MicroInterpreter staticInterpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter = &staticInterpreter;

  // Allocate memory for input/output tensors
  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.println("Model ready!");
}

void loop() {
  float aX, aY, aZ;

  // Wait for significant motion
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
      if (aSum >= accelerationThreshold) {
        samplesRead = 0;
        break;
      }
    }
  }

  // Collect IMU data
  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      Serial.println("getdata!");
      

      // ✅ CHUẨN HÓA DỮ LIỆU THEO CÔNG THỨC (x + 12) / 24
      tflInputTensor->data.f[samplesRead * 3 + 0] = (aX + 12.0) / 24.0;
      tflInputTensor->data.f[samplesRead * 3 + 1] = (aY + 12.0) / 24.0;
      tflInputTensor->data.f[samplesRead * 3 + 2] = (aZ + 12.0) / 24.0;

      samplesRead++;

      // If enough samples are collected, run inference
      if (samplesRead == numSamples) {
        Serial.println("Running inference...");
        
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
        }

        // Get the predicted gesture with the highest probability
        int predictedIndex = -1;
        float maxConfidence = 0.0;

        for (int i = 0; i < NUM_GESTURES; i++) {
          float confidence = tflOutputTensor->data.f[i];
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          Serial.println(confidence, 6);

          if (confidence > maxConfidence) {
            maxConfidence = confidence;
            predictedIndex = i;
          }
        }

        // Print the final prediction
        Serial.print("Predicted gesture: ");
        Serial.println(GESTURES[predictedIndex]);
        Serial.println();
      }
    }
  }
}
