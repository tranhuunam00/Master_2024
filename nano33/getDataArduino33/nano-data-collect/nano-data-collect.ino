#include "Arduino_BMI270_BMM150.h"

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // In header khớp với file Python
  Serial.println("aX,aY,aZ");
}

void loop() {
  float aX, aY, aZ;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(aX, aY, aZ);

    // Xuất dữ liệu cảm biến dạng CSV
    Serial.print(aX, 5);
    Serial.print(',');
    Serial.print(aY, 5);
    Serial.print(',');
    Serial.println(aZ, 5);
  }

  delay(10); // tránh spam quá nhanh
}
