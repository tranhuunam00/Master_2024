// SiPAP Rev C — pin map and tuning constants.
// Pin numbers are ESP32-S3 GPIOs, taken from SiPAP_Main_RevC.SchDoc
// (module pin -> GPIO per ESP32-S3-WROOM-1 datasheet table 3-1).
#pragma once
#include <Arduino.h>

// ---------------- Pins (PCB layout v3 GPIO map) ----------------
// Each signal uses the module edge that faces its part (v3 re-layout). No strapping pin
// (IO0/IO3/IO45/IO46) drives a peripheral any more; IO35-IO37 stay free (octal PSRAM on N8R8).
constexpr int PIN_PRESS_PRI   = 4;   // U1 pin 4,  U2 MPXV5010 via 10k/20k divider (ADC1)
constexpr int PIN_PRESS_RED   = 5;   // U1 pin 5,  U3 MPXV5010 via 10k/20k divider (ADC1), also LM393 input
constexpr int PIN_SAFETY_TRIP = 9;   // U1 pin 17, HIGH (~3 V) = hardware over-pressure cutoff active
constexpr int PIN_BUZZER      = 12;  // U1 pin 20, TDK PS1240 piezo, IO12 -> GND
constexpr int PIN_I2C_SDA     = 2;   // U1 pin 38, SFM3300 + OLED, 4.7k pull-ups to 3V3
constexpr int PIN_I2C_SCL     = 1;   // U1 pin 39
constexpr int PIN_SFM_EN      = 16;  // U1 pin 9,  HIGH = SFM3300 powered (Q3/Q2), 10k pull-up -> on by default
constexpr int PIN_ENC_A       = 47;  // U1 pin 24, PEC11R, 10k pull-ups, contacts to GND
constexpr int PIN_ENC_B       = 21;  // U1 pin 23
constexpr int PIN_ENC_SW      = 48;  // U1 pin 25, push = LOW
constexpr int PIN_BLOWER_PWM  = 10;  // U1 pin 18, -> R28 1k / C26 1uF -> J2.3 (VSR)
constexpr int PIN_BLOWER_FG   = 11;  // U1 pin 19, <- R29 1k <- J2.4, 10k pull-up to 3V3 (blower FG open-collector)

// ---------------- Pressure sensors (NXP MPXV5010) ----------------
// Vout = Vs * (0.09 * P[kPa] + 0.04) (MPX5010 DS p.4); ADC sees Vout * 20k / 30k.
constexpr float PRESS_DIVIDER   = 1.5f;     // sensor V = ADC V * 1.5
constexpr float PRESS_VS        = 5.077f;   // 5V rail nominal: VFB 0.596 V x (1 + R19 100k / R20 13.3k)
                                            // (TI SLVSD26C p.5 VFB typ; sensor is ratiometric, MPX5010 DS p.2 note 2)
constexpr float KPA_TO_CMH2O    = 10.1972f;
// Zero-pressure output window = the datasheet limits, not a guess: MPX5010 DS p.2 "Minimum Pressure
// Offset" Voff = 0 (min) / 0.2 (typ) / 0.425 V (max) at Vs = 5.0 V, 0-85 C. Scaled to Vs max 5.25 V
// (same table) because the output is ratiometric. A sensor whose output is open reads ~0 V and is
// caught by the U2/U3 cross-check (DISAGREE) as soon as pressure rises.
constexpr float VOFF_MAX_AT_5V  = 0.425f;
constexpr float ZERO_V_MIN      = 0.0f;                           // exclusive
constexpr float ZERO_V_MAX      = VOFF_MAX_AT_5V * 5.25f / 5.0f;  // 0.446 V
constexpr float SENSOR_V_MAX    = 4.80f;    // above = sensor/wiring fault
constexpr uint32_t REZERO_QUIET_MS = 3000;  // re-zero only after the blower has sent no FG pulse for 3 s

// ---------------- Hardware trip calibration (RV1 on COMP_REF) ----------------
constexpr float TRIP_TARGET_CMH2O = 25.0f;  // LM393 cutoff set point, adjusted with RV1 at test
constexpr float TRIP_TOL_CMH2O    = 1.0f;   // self-test pass window: 24.0-26.0 cmH2O read on U2

// ---------------- BLE ----------------
// Commands need an encrypted, MITM-authenticated, bonded link. The 6-digit passkey is derived from
// this chip's factory MAC, so it differs per board and is not in the source; it is shown on the OLED
// at boot and printed on Serial.
// Bench-only open-loop "PWM:n" command: compile it in only for bench tests, never for a patient build.
// #define SIPAP_BENCH_MODE

// ---------------- SFM3300 flow sensor ----------------
constexpr uint8_t SFM_ADDR        = 0x40;
constexpr float   SFM_OFFSET      = 32768.0f;
constexpr float   SFM_SCALE       = 120.0f;  // slm per count (air)
constexpr uint8_t SFM_MAX_FAILS   = 20;      // consecutive bad reads before a power-cycle reset
constexpr uint32_t SFM_OFF_MS     = 100;     // sensor VDD off time in a power-cycle reset
constexpr uint32_t I2C_HZ         = 400000;  // SFM3300 max 400 kHz (Sensirion SFM3300 DS p.3)

// ---------------- Blower: Wonsmart WS4540-12-NZ03 (12 V, 1.6 A max, built-in driver) ----------------
constexpr uint32_t BLOWER_PWM_HZ     = 20000;  // WS4540-12-NZ03 VSR input is 0-3.3 V speed voltage;
                                               // high-frequency 3.3 V PWM worked on the bench (dkbom.ino).
                                               // On-board R28 1k + C26 1uF (fc 159 Hz) turn it into a DC level.
constexpr uint8_t  BLOWER_PWM_BITS   = 10;
constexpr uint16_t BLOWER_PWM_MAX    = (1u << BLOWER_PWM_BITS) - 1;
constexpr uint8_t  BLOWER_FG_PER_REV = 1;      // WS4540-12-NZ03 drawing: FG "1 Hz x 60 = speed" -> 1 pulse/rev
constexpr float    BLOWER_MIN_DUTY   = 0.15f;  // below this most BLDC blowers stall

// ---------------- Therapy limits ----------------
constexpr float SET_MIN_CMH2O     = 4.0f;
constexpr float SET_MAX_CMH2O     = 20.0f;
constexpr float SET_STEP_CMH2O    = 0.5f;
constexpr float SET_DEFAULT       = 8.0f;
constexpr float RAMP_CMH2O_PER_S  = 0.5f;    // comfort ramp from 4 cmH2O to setpoint
constexpr float SOFT_LIMIT_CMH2O  = 22.0f;   // firmware cut, below the 25 cmH2O hardware trip (RV1-calibrated)

// ---------------- PI controller (pressure -> duty) ----------------
constexpr float KP = 0.020f;   // duty per cmH2O
constexpr float KI = 0.040f;   // duty per cmH2O*s
constexpr float FF = 0.030f;   // feed-forward duty per cmH2O of setpoint

// ---------------- Alarm thresholds ----------------
constexpr float    DISAGREE_CMH2O   = 2.0f;   // primary vs redundant sensor
constexpr uint32_t DISAGREE_MS      = 1000;
constexpr float    HIGH_DEV_CMH2O   = 5.0f;   // above setpoint
constexpr uint32_t HIGH_DEV_MS      = 1000;
constexpr float    LOW_DEV_CMH2O    = 3.0f;   // below setpoint (tube off / big leak)
constexpr uint32_t LOW_DEV_MS       = 10000;
constexpr uint32_t STALL_MS         = 2000;   // duty > min but no FG pulses

// ---------------- Timing ----------------
constexpr uint32_t CONTROL_MS = 10;
constexpr uint32_t DISPLAY_MS = 200;
constexpr uint32_t BLE_MS     = 200;
constexpr uint32_t WDT_MS     = 3000;
