int pwmPin = 3;
int fgPin = 2;

volatile int pulseCount = 0;

void countPulse() {
  pulseCount++;
}

void setup() {
  pinMode(pwmPin, OUTPUT);
  pinMode(fgPin, INPUT);

  attachInterrupt(digitalPinToInterrupt(fgPin), countPulse, RISING);

  Serial.begin(115200);
}

void loop() {
  // ===== chạy blower =====
  analogWrite(pwmPin, 55);

  pulseCount = 0;
  delay(1000);  // đo trong 1 giây

  int rpm = (pulseCount / 2) * 60;  // giả sử 2 xung / vòng

  Serial.print("RPM: ");
  Serial.println(rpm);

  delay(4000); // đủ 5s

  // ===== dừng blower =====
  analogWrite(pwmPin, 0);

  pulseCount = 0;
  delay(1000);

  Serial.print("RPM (stop): ");
  Serial.println(pulseCount);

  delay(4000);
}