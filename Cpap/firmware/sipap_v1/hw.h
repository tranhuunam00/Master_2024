// SiPAP Rev C — board drivers: pressure sensors, SFM3300, blower, encoder, buzzer.
#pragma once
#include <Arduino.h>
#include "config.h"

// ---- Pressure (both MPXV5010) ----
struct PressureReading {
  float priV, redV;          // sensor output volts (after undoing the divider)
  float pri, red;            // cmH2O, zero-corrected
  bool  priOk, redOk;        // inside plausible voltage range
};
bool  pressureBegin();                    // zero calibration; call with blower OFF
void  pressureRead(PressureReading &r);
float pressureZeroV(bool primary);

// ---- SFM3300 flow ----
struct FlowReading {
  float slm;                 // + = toward patient
  bool  ok;                  // last read valid (CRC ok)
  uint32_t resets;           // power-cycle resets done
};
void sfmBegin();
void sfmPowerCycle();                     // hard reset via SFM_EN (datasheet requirement)
void sfmPoll(FlowReading &f);             // call every CONTROL_MS

// ---- Blower ----
void     blowerBegin();
void     blowerSetDuty(float duty);       // 0..1
float    blowerDuty();
uint32_t blowerRpm();                     // updated once per second
uint32_t blowerMsSinceLastPulse();

// ---- Encoder ----
void encoderBegin();
int  encoderTakeSteps();                  // detents since last call (+ = CW)
bool encoderTakePress();                  // short press event
bool encoderTakeLongPress();              // >= 1.5 s event

// ---- Buzzer ----
void buzzerBegin();
void buzzerTone(uint16_t hz);             // 0 = off
void buzzerBeep(uint16_t ms);             // blocking short beep (UI feedback)

// ---- Safety comparator ----
inline bool safetyTripped() { return digitalRead(PIN_SAFETY_TRIP) == HIGH; }
