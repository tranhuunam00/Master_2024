/*
  Arduino Nano 33 BLE Sense
  SFM3300 + BLE Notify + Blower PWM + RPM

  Chức năng:
  ✅ Đọc flow từ SFM3300
  ✅ Tính volume
  ✅ Điều khiển blower bằng PWM
  ✅ Đọc RPM blower
  ✅ Gửi data BLE sang app
  ✅ Nhận command từ app

  BLE Commands:
  - START
  - STOP
  - PWM:xx    (0 -> 255)

  JSON:
  {
    "flow":xx,
    "volume":xx,
    "sign":xx,
    "rpm":xx,
    "pwm":xx
  }
*/

#include <Wire.h>
#include <math.h>
#include <ArduinoBLE.h>

// =====================================================
// ================= SFM3300 CONFIG ====================
// =====================================================

#define POLYNOMIAL 0x31
#define SFM3300_ADDR 0x40

const float FLOW_BIAS = 0.3;
const float FILTER_ALPHA = 0.2;
const int SIGN_CONFIRM = 4;

const unsigned sensorMs = 10;
const unsigned bleSendMs = 200;

// =====================================================
// ==================== BLOWER =========================
// =====================================================

int pwmPin = 3;
int fgPin = 2;

volatile int pulseCount = 0;

int blowerPwm = 0;
bool blowerRunning = false;

int currentRPM = 0;

unsigned long lastRpmCheck = 0;

// =====================================================
// ==================== BLE UUID =======================
// =====================================================

BLEService cpapService(
  "cb24858f-399f-4498-85e8-fea9d383d54f"
);

BLEStringCharacteristic sensorCharacteristic(
  "5e9e214b-124c-434d-84e5-018dccd35df1",
  BLERead | BLENotify,
  120
);

BLEStringCharacteristic actionCharacteristic(
  "56debc28-acab-4184-8f86-1a9c887b220a",
  BLEWrite,
  50
);

// =====================================================
// ==================== VARIABLES ======================
// =====================================================

float flow = 0;
float flow_prev = 0;
float volume = 0;

unsigned long mt_prev = 0;

int current_sign = 0;
int sign_counter = 0;

bool crc_error = false;

bool notifyEnabled = false;

unsigned long sensorPrev = 0;
unsigned long lastBleSend = 0;

// =====================================================
// ====================== CRC ==========================
// =====================================================

uint8_t CRC_prim(uint8_t x, uint8_t crc)
{
  crc ^= x;

  for (uint8_t i = 0; i < 8; i++)
  {
    if (crc & 0x80)
      crc = (crc << 1) ^ POLYNOMIAL;
    else
      crc <<= 1;
  }

  return crc;
}

// =====================================================
// ==================== INTERRUPT ======================
// =====================================================

void countPulse()
{
  pulseCount++;
}

// =====================================================
// ==================== SIGN ===========================
// =====================================================

int get_sign(float f)
{
  if (f > FLOW_BIAS)
    return 1;

  if (f < -FLOW_BIAS)
    return -1;

  return 0;
}

void update_sign(int new_sign)
{
  if (new_sign == 0)
    return;

  if (new_sign == current_sign)
  {
    sign_counter = 0;
    return;
  }

  sign_counter++;

  if (sign_counter >= SIGN_CONFIRM)
  {
    Serial.println("---- Direction Changed ----");

    Serial.print("Volume cycle: ");
    Serial.println(volume);

    current_sign = new_sign;
    sign_counter = 0;

    volume = 0;
  }
}

// =====================================================
// ================= READ SENSOR =======================
// =====================================================

void SFM_measure()
{
  if (Wire.requestFrom(SFM3300_ADDR, 3) != 3)
    return;

  uint8_t crc = 0;

  uint8_t a = Wire.read();
  crc = CRC_prim(a, crc);

  uint8_t b = Wire.read();
  crc = CRC_prim(b, crc);

  uint8_t c = Wire.read();

  if ((crc_error = (crc != c)))
    return;

  uint16_t raw = (a << 8) | b;

  float new_flow =
    ((float)raw - 32768.0) / 120.0;

  // FILTER
  new_flow =
    FILTER_ALPHA * new_flow +
    (1 - FILTER_ALPHA) * flow_prev;

  // DEAD ZONE
  if (fabs(new_flow) < FLOW_BIAS)
    new_flow = 0;

  // SIGN
  int new_sign = get_sign(new_flow);

  update_sign(new_sign);

  // TIME
  unsigned long mt = millis();

  float dt = (mt - mt_prev);

  mt_prev = mt;

  // VOLUME
  volume +=
    (flow_prev + new_flow) / 2.0 *
    (dt / 60000.0) *
    1000.0;

  flow_prev = new_flow;
  flow = new_flow;
}

// =====================================================
// ===================== RPM ===========================
// =====================================================

void readRPM()
{
  unsigned long now = millis();

  if (now - lastRpmCheck >= 1000)
  {
    lastRpmCheck = now;

    noInterrupts();

    int count = pulseCount;

    pulseCount = 0;

    interrupts();

    // 2 pulse / revolution
    currentRPM = (count / 2) * 60;

    Serial.print("RPM: ");
    Serial.println(currentRPM);
  }
}

// =====================================================
// =================== BLOWER ==========================
// =====================================================

void setBlowerPWM(int pwm)
{
  pwm = constrain(pwm, 0, 255);

  blowerPwm = pwm;

  analogWrite(pwmPin, blowerPwm);

  blowerRunning = (pwm > 0);

  Serial.print("PWM SET: ");
  Serial.println(blowerPwm);
}

// =====================================================
// ================= SEND BLE DATA =====================
// =====================================================

void sendBleData()
{
  if (!notifyEnabled)
    return;

  String sensorData =
    "{"
    "\"flow\":" + String(flow, 2) + "," +
    "\"volume\":" + String(volume, 2) + "," +
    "\"sign\":" + String(current_sign) + "," +
    "\"rpm\":" + String(currentRPM) + "," +
    "\"pwm\":" + String(blowerPwm) +
    "}";

  bool success =
    sensorCharacteristic.writeValue(sensorData);

  if (success)
  {
    Serial.print("SEND: ");
    Serial.println(sensorData);
  }
  else
  {
    Serial.println("SEND FAILED");
  }
}

// =====================================================
// ====================== SETUP ========================
// =====================================================

void setup()
{
  Serial.begin(115200);

  while (!Serial);

  // ===================================================
  // BLOWER
  // ===================================================

  pinMode(pwmPin, OUTPUT);

  pinMode(fgPin, INPUT);

  attachInterrupt(
    digitalPinToInterrupt(fgPin),
    countPulse,
    RISING
  );

  analogWrite(pwmPin, 0);

  // ===================================================
  // LED
  // ===================================================

  pinMode(LED_BUILTIN, OUTPUT);

  // ===================================================
  // I2C
  // ===================================================

  Wire.begin();

  // RESET SENSOR
  Wire.beginTransmission(SFM3300_ADDR);
  Wire.write(0x20);
  Wire.write(0x00);
  Wire.endTransmission();

  delay(100);

  // START MEASUREMENT
  Wire.beginTransmission(SFM3300_ADDR);
  Wire.write(0x10);
  Wire.write(0x00);
  Wire.endTransmission();

  delay(100);

  mt_prev = millis();

  Serial.println("SFM3300 READY");

  // ===================================================
  // BLE
  // ===================================================

  if (!BLE.begin())
  {
    Serial.println("BLE START FAILED");

    while (1);
  }

  BLE.setLocalName("CPAP_VSSM");

  BLE.setDeviceName("CPAP_VSSM");

  BLE.setAdvertisedService(cpapService);

  cpapService.addCharacteristic(
    sensorCharacteristic
  );

  cpapService.addCharacteristic(
    actionCharacteristic
  );

  BLE.addService(cpapService);

  sensorCharacteristic.writeValue("ready");

  BLE.advertise();

  Serial.println("BLE READY");
}

// =====================================================
// ======================= LOOP ========================
// =====================================================

void loop()
{
  BLEDevice central = BLE.central();

  if (central)
  {
    Serial.print("CONNECTED: ");
    Serial.println(central.address());

    while (central.connected())
    {
      BLE.poll();

      unsigned long now = millis();

      // ===============================================
      // READ FLOW
      // ===============================================

      if (now - sensorPrev >= sensorMs)
      {
        sensorPrev = now;

        SFM_measure();
      }

      // ===============================================
      // READ RPM
      // ===============================================

      readRPM();

      // ===============================================
      // NOTIFY
      // ===============================================

      notifyEnabled =
        sensorCharacteristic.subscribed();

      // ===============================================
      // SEND BLE
      // ===============================================

      if (notifyEnabled)
      {
        if (now - lastBleSend >= bleSendMs)
        {
          lastBleSend = now;

          sendBleData();
        }
      }

      // ===============================================
      // RECEIVE COMMAND
      // ===============================================

      if (actionCharacteristic.written())
      {
        String action =
          actionCharacteristic.value();

        Serial.print("ACTION: ");
        Serial.println(action);

        // =============================================
        // LED
        // =============================================

        if (action == "LED_ON")
        {
          digitalWrite(
            LED_BUILTIN,
            HIGH
          );
        }

        else if (action == "LED_OFF")
        {
          digitalWrite(
            LED_BUILTIN,
            LOW
          );
        }

        // =============================================
        // START
        // =============================================

        else if (action == "START")
        {
          setBlowerPWM(55);

          Serial.println("BLOWER START");
        }

        // =============================================
        // STOP
        // =============================================

        else if (action == "STOP")
        {
          setBlowerPWM(0);

          Serial.println("BLOWER STOP");
        }

        // =============================================
        // PWM:120
        // =============================================

        else if (action.startsWith("PWM:"))
        {
          String value =
            action.substring(4);

          int pwm =
            value.toInt();

          setBlowerPWM(pwm);
        }
      }

      delay(10);
    }

    Serial.println("DISCONNECTED");

    notifyEnabled = false;

    setBlowerPWM(0);
  }
}