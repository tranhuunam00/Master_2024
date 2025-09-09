#include "Arduino_BMI270_BMM150.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

const int WINDOW_SIZE = 10;       // gom đủ 10 mẫu mới dùng
const int FEATURE_SIZE = 3;       // aX,aY,aZ
float windowData[WINDOW_SIZE][FEATURE_SIZE];
int windowIndex = 0;

tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURES[] = { "ngửa", "nghiêng trái", "nghiêng phải", "sấp" };
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("IMU init fail!");
    while (1);
  }

  // Load model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  static tflite::MicroInterpreter staticInterpreter(
    tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter
  );
  tflInterpreter = &staticInterpreter;
  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.println("Model ready, collecting data...");
}

void loop() {
  float aX, aY, aZ;
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(aX, aY, aZ);

    // Lưu vào buffer
    windowData[windowIndex][0] = aX;
    windowData[windowIndex][1] = aY;
    windowData[windowIndex][2] = aZ;
    windowIndex++;

    // Nếu đủ 10 mẫu thì normalize + copy vào input tensor
    if (windowIndex >= WINDOW_SIZE) {
      for (int i = 0; i < WINDOW_SIZE; i++) {
        tflInputTensor->data.f[i * FEATURE_SIZE + 0] = (windowData[i][0] ) ;
        tflInputTensor->data.f[i * FEATURE_SIZE + 1] = (windowData[i][1] ) ;
        tflInputTensor->data.f[i * FEATURE_SIZE + 2] = (windowData[i][2] ) ;
      }

      Serial.println("Running inference on 10-sample window...");
      if (tflInterpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
      }

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

      Serial.print("Predicted gesture: ");
      Serial.println(GESTURES[predictedIndex]);
      Serial.println();

      // reset window
      windowIndex = 0;
    }
  }
}
