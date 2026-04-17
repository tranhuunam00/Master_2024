#include <PDM.h>
#include "SparkFun_BMI270_Arduino_Library.h"
#include <Wire.h>

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 256

short sampleBuffer[BUFFER_SIZE];
volatile int samplesRead = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  int bytesToRead = min(bytesAvailable, BUFFER_SIZE * 2);

  PDM.read(sampleBuffer, bytesToRead);
  samplesRead = bytesToRead / 2;
}

void setup() {
  Serial.begin(921600);   // ⚠️ tăng tốc độ
  PDM.onReceive(onPDMdata);
  PDM.begin(1, SAMPLE_RATE);
}

void loop() {
  if (samplesRead > 0) {
    Serial.write((byte*)sampleBuffer, samplesRead * 2);  // gửi binary
    samplesRead = 0;
  }
}