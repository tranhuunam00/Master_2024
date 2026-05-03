#include <PDM.h>
#include "SparkFun_BMI270_Arduino_Library.h"
#include <Wire.h>

#define SAMPLE_RATE 16000
#define BUFFER_SIZE 256

short sampleBuffer[BUFFER_SIZE];
volatile int samplesRead = 0;

BMI270 imu;

// ===== HEADER =====
uint8_t header[2] = {0xAA, 0x55};

// ===== PDM callback =====
void onPDMdata() {
  int bytesAvailable = PDM.available();
  int bytesToRead = min(bytesAvailable, BUFFER_SIZE * 2);

  PDM.read(sampleBuffer, bytesToRead);
  samplesRead = bytesToRead / 2;
}

void setup() {
  Serial.begin(921600);

  // PDM
  PDM.onReceive(onPDMdata);
  PDM.begin(1, SAMPLE_RATE);

  // IMU
  Wire1.begin();
  imu.beginI2C(BMI2_I2C_PRIM_ADDR, Wire1);
}

void loop() {
  if (samplesRead > 0) {

    // đọc IMU
    imu.getSensorData();
    float x = imu.data.accelX;
    float y = imu.data.accelY;
    float z = imu.data.accelZ;

    // ===== GỬI PACKET =====
    Serial.write(header, 2);                         // header
    Serial.write((byte*)sampleBuffer, samplesRead*2); // audio
    Serial.write((byte*)&x, 4);                      // float x
    Serial.write((byte*)&y, 4);                      // float y
    Serial.write((byte*)&z, 4);                      // float z

    samplesRead = 0;
  }
}