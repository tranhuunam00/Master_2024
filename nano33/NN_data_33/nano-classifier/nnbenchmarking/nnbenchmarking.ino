#include <Arduino.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

#define WINDOW_SIZE 10
#define FEATURE_SIZE 3
float windowData[WINDOW_SIZE][FEATURE_SIZE];
int windowIndex = 0;

String line = "";

// ---- TFLite Micro ----
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
tflite::AllOpsResolver tflOpsResolver;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

void setup()
{
  Serial.begin(115200);
  while (!Serial)
    ;

  Serial.println("NN Benchmark Mode\n");

  // Load model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Model schema mismatch!");
    while (1)
      ;
  }

  static tflite::MicroInterpreter staticInterpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  tflInterpreter = &staticInterpreter;
  tflInterpreter->AllocateTensors();

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.println("Model ready.");
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

  // Parse x,y,z
  int p1 = line.indexOf(',');
  int p2 = line.indexOf(',', p1 + 1);

  if (p1 < 0 || p2 < 0)
  {
    Serial.println("ERR_FORMAT");
    line = "";
    return;
  }

  float ax = line.substring(0, p1).toFloat();
  float ay = line.substring(p1 + 1, p2).toFloat();
  float az = line.substring(p2 + 1).toFloat();

  windowData[windowIndex][0] = ax;
  windowData[windowIndex][1] = ay;
  windowData[windowIndex][2] = az;
  windowIndex++;

  line = "";

  // Enough samples → run inference
  if (windowIndex == WINDOW_SIZE)
  {

    // Copy to tensor (interleaved)
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
      tflInputTensor->data.f[i * 3 + 0] = windowData[i][0];
      tflInputTensor->data.f[i * 3 + 1] = windowData[i][1];
      tflInputTensor->data.f[i * 3 + 2] = windowData[i][2];
    }

    if (tflInterpreter->Invoke() != kTfLiteOk)
    {
      Serial.println("ERR_INVOKE");
      windowIndex = 0;
      return;
    }

    int best = 0;
    float maxv = tflOutputTensor->data.f[0];
    for (int i = 1; i < 4; i++)
    {
      float v = tflOutputTensor->data.f[i];
      if (v > maxv)
      {
        maxv = v;
        best = i;
      }
    }

    Serial.println(best + 1);

    windowIndex = 0;
  }
}
