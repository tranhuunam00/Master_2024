#include "hw.h"
#include <Wire.h>

// ======================= Pressure =======================
static float zeroPriV = 0.20f, zeroRedV = 0.20f;

static float sensorVolts(int pin) {
  // average 8 calibrated readings (eFuse-calibrated mV)
  uint32_t mv = 0;
  for (int i = 0; i < 8; i++) mv += analogReadMilliVolts(pin);
  return (mv / 8.0f) / 1000.0f * PRESS_DIVIDER;
}

static float voltsToCmH2O(float v, float zeroV) {
  // P = (V/Vs - 0.04)/0.09 kPa; zero offset replaces the nominal 0.04*Vs term.
  return (v - zeroV) / (PRESS_VS * 0.09f) * KPA_TO_CMH2O;
}

bool pressureBegin() {
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_PRESS_PRI, ADC_11db);
  analogSetPinAttenuation(PIN_PRESS_RED, ADC_11db);
  float p = 0, r = 0;
  for (int i = 0; i < 50; i++) {
    p += sensorVolts(PIN_PRESS_PRI);
    r += sensorVolts(PIN_PRESS_RED);
    delay(4);
  }
  p /= 50; r /= 50;
  bool ok = p > ZERO_V_MIN && p < ZERO_V_MAX && r > ZERO_V_MIN && r < ZERO_V_MAX;
  if (ok) { zeroPriV = p; zeroRedV = r; }
  return ok;
}

float pressureZeroV(bool primary) { return primary ? zeroPriV : zeroRedV; }

void pressureRead(PressureReading &r) {
  r.priV = sensorVolts(PIN_PRESS_PRI);
  r.redV = sensorVolts(PIN_PRESS_RED);
  r.priOk = r.priV > ZERO_V_MIN * 0.5f && r.priV < SENSOR_V_MAX;
  r.redOk = r.redV > ZERO_V_MIN * 0.5f && r.redV < SENSOR_V_MAX;
  r.pri = voltsToCmH2O(r.priV, zeroPriV);
  r.red = voltsToCmH2O(r.redV, zeroRedV);
}

// ======================= SFM3300 =======================
static uint8_t  sfmFails = 0;
static uint32_t sfmResets = 0;
static bool     sfmStarted = false;

static uint8_t crc8(const uint8_t *d, int n) {
  uint8_t crc = 0;
  for (int i = 0; i < n; i++) {
    crc ^= d[i];
    for (int b = 0; b < 8; b++) crc = (crc & 0x80) ? (crc << 1) ^ 0x31 : (crc << 1);
  }
  return crc;
}

static bool sfmCommand(uint16_t cmd) {
  Wire.beginTransmission(SFM_ADDR);
  Wire.write(cmd >> 8);
  Wire.write(cmd & 0xFF);
  return Wire.endTransmission() == 0;
}

static void sfmStart() {
  sfmCommand(0x2000);            // soft reset
  delay(100);
  sfmStarted = sfmCommand(0x1000);  // start continuous measurement
  delay(10);
  if (sfmStarted) {              // first value after start is invalid: discard it
    Wire.requestFrom((int)SFM_ADDR, 3);
    while (Wire.available()) Wire.read();
  }
}

void sfmBegin() {
  pinMode(PIN_SFM_EN, OUTPUT);
  digitalWrite(PIN_SFM_EN, HIGH);
  delay(100);
  sfmStart();
}

void sfmPowerCycle() {
  digitalWrite(PIN_SFM_EN, LOW);   // Q2 off -> sensor VDD 0 V
  delay(100);
  digitalWrite(PIN_SFM_EN, HIGH);
  delay(100);
  sfmResets++;
  sfmFails = 0;
  sfmStart();
}

void sfmPoll(FlowReading &f) {
  f.resets = sfmResets;
  uint8_t b[3];
  bool ok = Wire.requestFrom((int)SFM_ADDR, 3) == 3;
  if (ok) {
    for (int i = 0; i < 3; i++) b[i] = Wire.read();
    ok = crc8(b, 2) == b[2];
  }
  if (ok) {
    uint16_t raw = (uint16_t(b[0]) << 8) | b[1];
    f.slm = (float(raw) - SFM_OFFSET) / SFM_SCALE;
    sfmFails = 0;
  } else if (++sfmFails >= SFM_MAX_FAILS) {
    sfmPowerCycle();
  }
  f.ok = ok;
}

// ======================= Blower =======================
static volatile uint32_t fgPulses = 0;
static volatile uint32_t fgLastUs = 0;
static float    duty = 0;
static uint32_t rpm = 0, rpmT = 0, rpmPulses = 0;

static void IRAM_ATTR fgIsr() { fgPulses++; fgLastUs = micros(); }

void blowerBegin() {
  pinMode(PIN_BLOWER_PWM, OUTPUT);
  digitalWrite(PIN_BLOWER_PWM, LOW);
  ledcAttach(PIN_BLOWER_PWM, BLOWER_PWM_HZ, BLOWER_PWM_BITS);
  ledcWrite(PIN_BLOWER_PWM, 0);
  pinMode(PIN_BLOWER_FG, INPUT);            // external 10k pull-up
  attachInterrupt(digitalPinToInterrupt(PIN_BLOWER_FG), fgIsr, FALLING);
  rpmT = millis();
}

void blowerSetDuty(float d) {
  duty = constrain(d, 0.0f, 1.0f);
  ledcWrite(PIN_BLOWER_PWM, (uint32_t)(duty * BLOWER_PWM_MAX + 0.5f));
}

float blowerDuty() { return duty; }

uint32_t blowerRpm() {
  uint32_t now = millis();
  if (now - rpmT >= 1000) {
    uint32_t p = fgPulses;
    rpm = (p - rpmPulses) * 60000UL / (now - rpmT) / BLOWER_FG_PER_REV;
    rpmPulses = p;
    rpmT = now;
  }
  return rpm;
}

uint32_t blowerMsSinceLastPulse() {
  uint32_t last = fgLastUs;
  return last == 0 ? UINT32_MAX : (micros() - last) / 1000;
}

// ======================= Encoder =======================
static volatile int8_t encQuarter = 0;
static volatile uint8_t encState = 0;
static int  encSteps = 0;
static bool swDown = false, swLongFired = false, pressEvt = false, longEvt = false;
static uint32_t swT = 0;

static void IRAM_ATTR encIsr() {
  // Gray-code state table: +1 / -1 per valid transition, 4 transitions per detent.
  static const int8_t tbl[16] = {0,-1,1,0, 1,0,0,-1, -1,0,0,1, 0,1,-1,0};
  uint8_t s = (digitalRead(PIN_ENC_A) << 1) | digitalRead(PIN_ENC_B);
  encQuarter += tbl[(encState << 2) | s];
  encState = s;
}

void encoderBegin() {
  pinMode(PIN_ENC_A, INPUT);   // external 10k pull-ups
  pinMode(PIN_ENC_B, INPUT);
  pinMode(PIN_ENC_SW, INPUT);
  encState = (digitalRead(PIN_ENC_A) << 1) | digitalRead(PIN_ENC_B);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), encIsr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_B), encIsr, CHANGE);
}

static void encoderScanButton() {
  bool down = digitalRead(PIN_ENC_SW) == LOW;
  uint32_t now = millis();
  if (down && !swDown) { swDown = true; swT = now; swLongFired = false; }
  else if (down && swDown && !swLongFired && now - swT >= 1500) { swLongFired = true; longEvt = true; }
  else if (!down && swDown) {
    swDown = false;
    if (!swLongFired && now - swT >= 30) pressEvt = true;   // 30 ms debounce
  }
}

int encoderTakeSteps() {
  noInterrupts();
  int q = encQuarter;
  int det = q / 4;
  encQuarter = q - det * 4;
  interrupts();
  return det;
}

bool encoderTakePress()     { encoderScanButton(); bool e = pressEvt; pressEvt = false; return e; }
bool encoderTakeLongPress() { encoderScanButton(); bool e = longEvt;  longEvt = false;  return e; }

// ======================= Buzzer =======================
void buzzerBegin() {
  ledcAttach(PIN_BUZZER, 4000, 8);
  ledcWrite(PIN_BUZZER, 0);
}

void buzzerTone(uint16_t hz) {
  if (hz == 0) ledcWrite(PIN_BUZZER, 0);
  else { ledcChangeFrequency(PIN_BUZZER, hz, 8); ledcWrite(PIN_BUZZER, 128); }
}

void buzzerBeep(uint16_t ms) {
  buzzerTone(4000);   // PS1240 resonant ~4 kHz
  delay(ms);
  buzzerTone(0);
}
