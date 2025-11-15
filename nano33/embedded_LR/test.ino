// #include "Arduino_BMI270_BMM150.h"

// const int sampleRateHz = 10;
// const unsigned long sampleDelay = 1000 / sampleRateHz;

// void setup() {
//   Serial.begin(9600);
//   while (!Serial);

//   if (!IMU.begin()) {
//     Serial.println("Không khởi tạo được cảm biến BMI270!");
//     while (1);
//   }

//   // Không cần setAccelerometerRange - mặc định đã là ±2g

//   Serial.println("aX,aY,aZ");  // CSV header
// }

// void loop() {
//   float ax, ay, az;

//   if (IMU.accelerationAvailable()) {
//     IMU.readAcceleration(ax, ay, az);

//     Serial.print(ax, 5);
//     Serial.print(',');
//     Serial.print(ay, 5);
//     Serial.print(',');
//     Serial.println(az, 5);
//   }

//   delay(sampleDelay); // 10 Hz
// }
