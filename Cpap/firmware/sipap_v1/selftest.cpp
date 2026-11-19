#include <Wire.h>
#include <U8g2lib.h>
#include "config.h"
#include "hw.h"
#include "selftest.h"

static int nPass = 0, nFail = 0, nSkip = 0;

static void result(const char *name, int r, const char *detail = "") {
  const char *t = r > 0 ? "PASS" : (r < 0 ? "FAIL" : "SKIP");
  if (r > 0) nPass++; else if (r < 0) nFail++; else nSkip++;
  Serial.printf("  [%s] %-28s %s\n", t, name, detail);
}

// wait for 'y' / 'n' / 's' (skip) on Serial; returns 1 / -1 / 0
static int ask(const char *q, uint32_t timeoutMs = 30000) {
  Serial.printf("  ? %s  [y = yes / n = no / s = skip]\n", q);
  while (Serial.available()) Serial.read();
  uint32_t t0 = millis();
  while (millis() - t0 < timeoutMs) {
    if (Serial.available()) {
      char c = tolower(Serial.read());
      if (c == 'y') return 1;
      if (c == 'n') return -1;
      if (c == 's') return 0;
    }
    delay(10);
  }
  return 0;
}

static bool i2cPresent(uint8_t a) {
  Wire.beginTransmission(a);
  return Wire.endTransmission() == 0;
}

void runSelfTest(U8G2 &oled) {
  char d[96];
  blowerSetDuty(0);
  Serial.println("\n========== SiPAP Rev C SELF-TEST ==========");
  Serial.println("Blower must be OFF/idle; pressure ports open to air.\n");

  // 1. I2C bus
  Serial.println("1. I2C bus");
  bool oledOk = i2cPresent(0x3C) || i2cPresent(0x3D);
  bool sfmOk  = i2cPresent(SFM_ADDR);
  result("OLED at 0x3C", oledOk ? 1 : -1);
  result("SFM3300 at 0x40", sfmOk ? 1 : -1);

  // 2. Pressure sensors at 0 cmH2O
  Serial.println("2. Pressure sensors (ports open)");
  bool zeroOk = pressureBegin();
  PressureReading pr;
  pressureRead(pr);
  snprintf(d, sizeof d, "pri %.3f V, red %.3f V (expect 0.10-0.35, typ 0.20)", pr.priV, pr.redV);
  result("zero output in range", zeroOk ? 1 : -1, d);
  float diff = fabsf((pr.priV - pr.redV) / (PRESS_VS * 0.09f) * KPA_TO_CMH2O);
  snprintf(d, sizeof d, "raw difference %.2f cmH2O (expect < 1.0 before zeroing)", diff);
  result("sensors agree at zero", diff < 1.0f ? 1 : -1, d);

  // 3. Hardware cutoff at rest
  Serial.println("3. Safety comparator");
  result("SAFETY_TRIP low at 0 cmH2O", !safetyTripped() ? 1 : -1);

  // 4. SFM3300 power switch (Q2/Q3) and readings
  Serial.println("4. SFM3300 power switch + flow");
  digitalWrite(PIN_SFM_EN, LOW);
  delay(150);
  bool gone = !i2cPresent(SFM_ADDR);
  digitalWrite(PIN_SFM_EN, HIGH);
  delay(150);
  bool back = i2cPresent(SFM_ADDR);
  result("sensor off when IO16 = 0", gone ? 1 : -1, "U4 pin 2 should read 0 V");
  result("sensor back when IO16 = 1", back ? 1 : -1, "U4 pin 2 should read ~5 V");
  if (back) {
    sfmPowerCycle();
    FlowReading f; int good = 0; float sum = 0;
    for (int i = 0; i < 50; i++) { sfmPoll(f); if (f.ok) { good++; sum += f.slm; } delay(10); }
    snprintf(d, sizeof d, "%d/50 CRC ok, mean %.2f slm (expect |flow| < 1 at rest)", good, good ? sum / good : 0);
    result("flow readings", (good >= 45 && fabsf(sum / max(good, 1)) < 1.0f) ? 1 : -1, d);
  } else result("flow readings", 0, "sensor not present");

  // 5. OLED
  Serial.println("5. OLED");
  if (oledOk) {
    oled.clearBuffer();
    oled.drawFrame(0, 0, 128, 64);
    oled.setFont(u8g2_font_6x10_tf);
    oled.drawStr(20, 30, "SELF-TEST OLED");
    oled.drawBox(54, 40, 20, 12);
    oled.sendBuffer();
    result("frame + text visible", ask("Do you see a border, 'SELF-TEST OLED' and a box?"));
  } else result("frame + text visible", 0, "OLED not found");

  // 6. Buzzer
  Serial.println("6. Buzzer");
  buzzerBeep(300); delay(150); buzzerBeep(300);
  result("two beeps heard", ask("Did you hear two beeps?"));

  // 7. Encoder
  Serial.println("7. Encoder: turn CLOCKWISE 5 clicks within 15 s");
  encoderTakeSteps();
  int steps = 0; uint32_t t0 = millis();
  while (millis() - t0 < 15000 && abs(steps) < 5) { steps += encoderTakeSteps(); delay(10); }
  snprintf(d, sizeof d, "counted %d (clockwise must be positive)", steps);
  result("rotation + direction", steps >= 5 ? 1 : (steps == 0 ? 0 : -1), d);
  Serial.println("   now PRESS the knob within 10 s");
  encoderTakePress();
  bool pressed = false; t0 = millis();
  while (millis() - t0 < 10000 && !pressed) { pressed = encoderTakePress(); delay(10); }
  result("push switch", pressed ? 1 : 0);

  // 8. Blower
  Serial.println("8. Blower (J2)");
  if (ask("Is the blower connected and its outlet free?") > 0) {
    blowerRpm();
    blowerSetDuty(0.30f);
    delay(3000);
    uint32_t rpm = blowerRpm();
    delay(1100);
    rpm = blowerRpm();
    pressureRead(pr);
    blowerSetDuty(0);
    snprintf(d, sizeof d, "30%% duty -> %lu rpm, %.1f cmH2O at outlet", (unsigned long)rpm, pr.pri);
    result("FG pulses / rpm > 0", rpm > 0 ? 1 : -1, d);
    result("blower spins", ask("Did the blower spin and stop again?"));
  } else { result("FG pulses / rpm > 0", 0); result("blower spins", 0); }

  // 9. Independent over-pressure cutoff
  Serial.println("9. Hardware over-pressure cutoff (U3 + LM393 + Q1)");
  if (ask("Ready to raise pressure on U3's port (syringe / water column), slowly, within 60 s?") > 0) {
    Serial.println("   raise pressure slowly...");
    float tripAt = NAN, releaseAt = NAN, peak = 0;
    uint32_t t1 = millis();
    while (millis() - t1 < 60000 && isnan(tripAt)) {
      pressureRead(pr);
      peak = max(peak, pr.red);
      if (safetyTripped()) tripAt = pr.red;
      delay(20);
    }
    if (!isnan(tripAt)) {
      Serial.printf("   tripped at %.1f cmH2O, now release pressure slowly...\n", tripAt);
      t1 = millis();
      while (millis() - t1 < 60000 && isnan(releaseAt)) {
        pressureRead(pr);
        if (!safetyTripped()) releaseAt = pr.red;
        delay(20);
      }
    }
    snprintf(d, sizeof d, "trip %.1f cmH2O (expect 23.8-26.8), peak seen %.1f", tripAt, peak);
    result("trip threshold", (!isnan(tripAt) && tripAt > 23.8f && tripAt < 26.8f) ? 1 : -1, d);
    snprintf(d, sizeof d, "release %.1f cmH2O (expect 1.5-3.5 below trip)", releaseAt);
    float h = tripAt - releaseAt;
    result("hysteresis", (!isnan(releaseAt) && h > 1.5f && h < 3.5f) ? 1 : -1, d);
  } else { result("trip threshold", 0); result("hysteresis", 0); }

  Serial.printf("\n========== RESULT: %d PASS, %d FAIL, %d SKIP ==========\n", nPass, nFail, nSkip);
  Serial.println(nFail ? "BOARD NOT OK - see FAIL lines above." : "No failures.");
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x10_tf);
  snprintf(d, sizeof d, "PASS %d FAIL %d SKIP %d", nPass, nFail, nSkip);
  oled.drawStr(0, 30, "SELF-TEST DONE");
  oled.drawStr(0, 46, d);
  oled.sendBuffer();
  Serial.println("Reset the board to run normal firmware.");
  while (true) delay(1000);
}
