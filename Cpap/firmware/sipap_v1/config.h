// SiPAP Rev C — pin map and tuning constants.
// Pin numbers are ESP32-S3 GPIOs, taken from SiPAP_Main_RevC.SchDoc
// (module pin -> GPIO per ESP32-S3-WROOM-1 datasheet table 3-1).
#pragma once
#include <Arduino.h>

// ---------------- Pins ----------------
constexpr int PIN_PRESS_PRI   = 4;   // U2 MPXV5010 via 10k/20k divider (ADC1)
constexpr int PIN_PRESS_RED   = 5;   // U3 MPXV5010 via 10k/20k divider (ADC1), also LM393 input
constexpr int PIN_SAFETY_TRIP = 6;   // HIGH (~3 V) = hardware over-pressure cutoff active
constexpr int PIN_BUZZER      = 7;   // TDK PS1240 piezo, IO7 -> GND
constexpr int PIN_I2C_SDA     = 8;   // SFM3300 + OLED, 4.7k pull-ups to 3V3
constexpr int PIN_I2C_SCL     = 9;
constexpr int PIN_SFM_EN      = 16;  // HIGH = SFM3300 powered (Q3/Q2), 10k pull-up -> on by default
constexpr int PIN_ENC_A       = 17;  // PEC11R, 10k pull-ups, contacts to GND
constexpr int PIN_ENC_B       = 18;
constexpr int PIN_ENC_SW      = 21;  // push = LOW
constexpr int PIN_BLOWER_PWM  = 3;   // J2.3 (strapping pin; driven LOW first thing in setup)
constexpr int PIN_BLOWER_FG   = 2;   // J2.4, 10k pull-up to 3V3 (blower FG must be open-collector)

// ---------------- Pressure sensors (NXP MPXV5010) ----------------
// Vout = Vs * (0.09 * P[kPa] + 0.04); ADC sees Vout * 20k / 30k.
constexpr float PRESS_DIVIDER   = 1.5f;     // sensor V = ADC V * 1.5
constexpr float PRESS_VS        = 5.0f;     // 5V rail (TPS54202 U8, nominal 5.08 V)
constexpr float KPA_TO_CMH2O    = 10.1972f;
constexpr float ZERO_V_MIN      = 0.10f;    // plausible sensor voltage at 0 kPa (typ 0.20 V)
constexpr float ZERO_V_MAX      = 0.35f;
constexpr float SENSOR_V_MAX    = 4.80f;    // above = sensor/wiring fault

// ---------------- SFM3300 flow sensor ----------------
constexpr uint8_t SFM_ADDR        = 0x40;
constexpr float   SFM_OFFSET      = 32768.0f;
constexpr float   SFM_SCALE       = 120.0f;  // slm per count (air)
constexpr uint8_t SFM_MAX_FAILS   = 20;      // consecutive bad reads before a power-cycle reset

// ---------------- Blower ----------------
constexpr uint32_t BLOWER_PWM_HZ     = 25000;  // check against the blower datasheet
constexpr uint8_t  BLOWER_PWM_BITS   = 10;
constexpr uint16_t BLOWER_PWM_MAX    = (1u << BLOWER_PWM_BITS) - 1;
constexpr uint8_t  BLOWER_FG_PER_REV = 2;      // FG pulses per revolution (blower datasheet)
constexpr float    BLOWER_MIN_DUTY   = 0.15f;  // below this most BLDC blowers stall

// ---------------- Therapy limits ----------------
constexpr float SET_MIN_CMH2O     = 4.0f;
constexpr float SET_MAX_CMH2O     = 20.0f;
constexpr float SET_STEP_CMH2O    = 0.5f;
constexpr float SET_DEFAULT       = 8.0f;
constexpr float RAMP_CMH2O_PER_S  = 0.5f;    // comfort ramp from 4 cmH2O to setpoint
constexpr float SOFT_LIMIT_CMH2O  = 22.0f;   // firmware cut, below the 25 cmH2O hardware trip

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
