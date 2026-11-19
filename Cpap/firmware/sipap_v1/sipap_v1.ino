// SiPAP Rev C firmware — ESP32-S3-WROOM-1-N8R8
// Board: "ESP32S3 Dev Module", PSRAM: OPI, Flash: 8MB, upload via J3 (UART0, hold IO0 at reset).
//
// What it does
//   - PI pressure control on the primary MPXV5010 (U2), blower PWM on IO3
//   - redundant sensor U3 cross-check + monitors the independent LM393 cutoff (SAFETY_TRIP)
//   - SFM3300 flow + breath volume, hard reset by power cycling when it stops answering
//   - OLED (SSD1306) status, encoder for setpoint / start-stop, piezo alarms
//   - BLE: same service/characteristic UUIDs as the earlier Nano33 app protocol
//
// The hardware cutoff (25 cmH2O on U3, Q1) works without this firmware. Firmware alarms
// are a second layer, never the only one.

#include <Wire.h>
#include <U8g2lib.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include <esp_task_wdt.h>
#include "config.h"
#include "hw.h"
#include "selftest.h"

// ======================= State =======================
enum class Mode : uint8_t { Standby, Running, Fault };
enum Alarm : uint16_t {
  AL_NONE        = 0,
  AL_HW_TRIP     = 1 << 0,   // LM393 cut the blower (> ~25 cmH2O)
  AL_SOFT_LIMIT  = 1 << 1,   // primary > SOFT_LIMIT_CMH2O
  AL_HIGH        = 1 << 2,   // > setpoint + HIGH_DEV for HIGH_DEV_MS
  AL_LOW         = 1 << 3,   // < setpoint - LOW_DEV for LOW_DEV_MS (tube off / leak)
  AL_DISAGREE    = 1 << 4,   // primary vs redundant sensor
  AL_SENSOR      = 1 << 5,   // pressure sensor out of range
  AL_STALL       = 1 << 6,   // no FG pulses while driven
  AL_FLOW        = 1 << 7,   // SFM3300 not answering (warning only)
  AL_ZERO        = 1 << 8,   // zero calibration failed at boot
};
// Alarms that stop the blower (others only warn).
constexpr uint16_t AL_STOPPING = AL_HW_TRIP | AL_SOFT_LIMIT | AL_HIGH | AL_DISAGREE | AL_SENSOR | AL_STALL | AL_ZERO;

static Mode     mode = Mode::Standby;
static uint16_t alarms = AL_NONE;
static float    setpoint = SET_DEFAULT;
static float    target = SET_MIN_CMH2O;   // ramped setpoint
static float    integ = 0;
static bool     manualPwm = false;        // BLE "PWM:n" bench mode
static float    manualDuty = 0;
static bool     muted = false;
static uint32_t muteUntil = 0;

static PressureReading pr;
static FlowReading     fl = {0, false, 0};
static float  volumeMl = 0, lastBreathMl = 0;
static int8_t flowSign = 0;
static uint8_t signCount = 0;

static uint32_t tDisagree = 0, tHigh = 0, tLow = 0, tStall = 0;

// ======================= OLED =======================
static U8G2_SSD1306_128X64_NONAME_F_HW_I2C oled(U8G2_R0, U8X8_PIN_NONE, PIN_I2C_SCL, PIN_I2C_SDA);

// ======================= BLE =======================
static const char *SVC_UUID  = "cb24858f-399f-4498-85e8-fea9d383d54f";
static const char *DATA_UUID = "5e9e214b-124c-434d-84e5-018dccd35df1";
static const char *CMD_UUID  = "56debc28-acab-4184-8f86-1a9c887b220a";
static BLECharacteristic *dataChr = nullptr;
static volatile bool bleConnected = false;
static char pendingCmd[64];
static volatile bool cmdReady = false;
static portMUX_TYPE cmdMux = portMUX_INITIALIZER_UNLOCKED;

class ServerCb : public BLEServerCallbacks {
  void onConnect(BLEServer *) override { bleConnected = true; }
  void onDisconnect(BLEServer *s) override { bleConnected = false; s->getAdvertising()->start(); }
};
class CmdCb : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    String v = c->getValue();
    portENTER_CRITICAL(&cmdMux);
    strlcpy(pendingCmd, v.c_str(), sizeof pendingCmd);
    cmdReady = true;
    portEXIT_CRITICAL(&cmdMux);
  }
};

static void bleBegin() {
  BLEDevice::init("SIPAP");
  BLEServer *srv = BLEDevice::createServer();
  srv->setCallbacks(new ServerCb());
  BLEService *svc = srv->createService(SVC_UUID);
  dataChr = svc->createCharacteristic(DATA_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  dataChr->addDescriptor(new BLE2902());
  dataChr->setValue("READY");
  BLECharacteristic *cmd = svc->createCharacteristic(CMD_UUID, BLECharacteristic::PROPERTY_WRITE);
  cmd->setCallbacks(new CmdCb());
  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SVC_UUID);
  adv->start();
}

// ======================= Helpers =======================
static void startTherapy() {
  if (alarms & AL_STOPPING) return;
  mode = Mode::Running;
  target = min(SET_MIN_CMH2O, setpoint);
  integ = 0;
  tHigh = tLow = tStall = 0;
  buzzerBeep(60);
}

static void stopTherapy() {
  mode = Mode::Standby;
  manualPwm = false;
  blowerSetDuty(0);
  integ = 0;
}

static void raise(uint16_t a) {
  if (!(alarms & a)) Serial.printf("[ALARM] 0x%03X\n", a);
  alarms |= a;
  if (a & AL_STOPPING) { blowerSetDuty(0); if (mode == Mode::Running) mode = Mode::Fault; }
}

static void handleCommand(const String &c) {
  Serial.printf("[BLE] %s\n", c.c_str());
  if (c == "START") startTherapy();
  else if (c == "STOP") stopTherapy();
  else if (c.startsWith("SET:")) setpoint = constrain(c.substring(4).toFloat(), SET_MIN_CMH2O, SET_MAX_CMH2O);
  else if (c.startsWith("PWM:")) {           // bench mode: open loop, 0..255 like the old protocol
    if (alarms & AL_STOPPING) return;
    manualDuty = constrain(c.substring(4).toInt(), 0, 255) / 255.0f;
    manualPwm = manualDuty > 0;
    mode = manualPwm ? Mode::Running : Mode::Standby;
    if (!manualPwm) blowerSetDuty(0);
  } else if (c == "MUTE") { muted = true; muteUntil = millis() + 120000; }
  else if (c == "CLEAR") { alarms = safetyTripped() ? AL_HW_TRIP : AL_NONE; if (mode == Mode::Fault) mode = Mode::Standby; }
}

// ======================= Control (every CONTROL_MS) =======================
static void controlStep() {
  const float dt = CONTROL_MS / 1000.0f;
  const uint32_t now = millis();

  pressureRead(pr);
  sfmPoll(fl);

  // ---- breath volume from flow (sign change with debounce) ----
  if (fl.ok) {
    float f = fabsf(fl.slm) < 0.3f ? 0 : fl.slm;
    volumeMl += f * dt * 1000.0f / 60.0f;
    int8_t s = f > 0 ? 1 : (f < 0 ? -1 : 0);
    if (s != 0 && s != flowSign) {
      if (++signCount >= 4) { lastBreathMl = volumeMl; volumeMl = 0; flowSign = s; signCount = 0; }
    } else signCount = 0;
  }

  // ---- safety checks (run in every mode) ----
  if (safetyTripped()) raise(AL_HW_TRIP);
  if (!pr.priOk || !pr.redOk) raise(AL_SENSOR);
  if (pr.pri > SOFT_LIMIT_CMH2O) raise(AL_SOFT_LIMIT);

  if (fabsf(pr.pri - pr.red) > DISAGREE_CMH2O) {
    if (!tDisagree) tDisagree = now;
    else if (now - tDisagree > DISAGREE_MS) raise(AL_DISAGREE);
  } else tDisagree = 0;

  if (fl.resets >= 3 && !fl.ok) alarms |= AL_FLOW; else if (fl.ok) alarms &= ~AL_FLOW;

  if (mode != Mode::Running) { blowerSetDuty(0); return; }

  // ---- blower command ----
  float d;
  if (manualPwm) {
    d = manualDuty;
  } else {
    target = min(setpoint, target + RAMP_CMH2O_PER_S * dt);
    float err = target - pr.pri;
    integ = constrain(integ + KI * err * dt, -0.3f, 0.6f);
    d = FF * target + KP * err + integ;
    d = constrain(d, BLOWER_MIN_DUTY, 1.0f);

    if (pr.pri > setpoint + HIGH_DEV_CMH2O) { if (!tHigh) tHigh = now; else if (now - tHigh > HIGH_DEV_MS) raise(AL_HIGH); }
    else tHigh = 0;
    if (target >= setpoint && pr.pri < setpoint - LOW_DEV_CMH2O) { if (!tLow) tLow = now; else if (now - tLow > LOW_DEV_MS) alarms |= AL_LOW; }
    else { tLow = 0; alarms &= ~AL_LOW; }
  }

  // ---- stall: driven but no FG pulse ----
  if (d >= BLOWER_MIN_DUTY && blowerMsSinceLastPulse() > 500) {
    if (!tStall) tStall = now; else if (now - tStall > STALL_MS) raise(AL_STALL);
  } else tStall = 0;

  if (mode == Mode::Running) blowerSetDuty(d);
}

// ======================= UI =======================
static const char *alarmText(uint16_t a) {
  if (a & AL_HW_TRIP)    return "OVER PRESSURE (HW)";
  if (a & AL_SOFT_LIMIT) return "OVER PRESSURE";
  if (a & AL_HIGH)       return "PRESSURE HIGH";
  if (a & AL_DISAGREE)   return "SENSORS DISAGREE";
  if (a & AL_SENSOR)     return "SENSOR FAULT";
  if (a & AL_ZERO)       return "ZERO CAL FAILED";
  if (a & AL_STALL)      return "BLOWER STALLED";
  if (a & AL_LOW)        return "LOW PRESS / LEAK";
  if (a & AL_FLOW)       return "FLOW SENSOR";
  return "";
}

static void drawDisplay() {
  char buf[32];
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x10_tf);
  const char *m = mode == Mode::Running ? (manualPwm ? "MANUAL" : "RUN") : (mode == Mode::Fault ? "FAULT" : "STANDBY");
  snprintf(buf, sizeof buf, "%-7s set %4.1f", m, setpoint);
  oled.drawStr(0, 9, buf);
  if (bleConnected) oled.drawStr(116, 9, "B");

  oled.setFont(u8g2_font_logisoso22_tf);
  snprintf(buf, sizeof buf, "%4.1f", pr.pri);
  oled.drawStr(0, 36, buf);
  oled.setFont(u8g2_font_6x10_tf);
  oled.drawStr(72, 36, "cmH2O");

  snprintf(buf, sizeof buf, "F%5.1f L/m V%4.0f mL", fl.ok ? fl.slm : 0.0f, lastBreathMl);
  oled.drawStr(0, 48, buf);
  snprintf(buf, sizeof buf, "%3d%% %5lu rpm", (int)(blowerDuty() * 100), (unsigned long)blowerRpm());
  oled.drawStr(0, 58, buf);

  if (alarms) {
    oled.setDrawColor(1);
    oled.drawBox(0, 50, 128, 14);
    oled.setDrawColor(0);
    oled.drawStr(2, 61, alarmText(alarms));
    oled.setDrawColor(1);
  }
  oled.sendBuffer();
}

static void handleUi() {
  int steps = encoderTakeSteps();
  if (steps && !manualPwm) setpoint = constrain(setpoint + steps * SET_STEP_CMH2O, SET_MIN_CMH2O, SET_MAX_CMH2O);

  if (encoderTakeLongPress()) {               // long press: start / stop
    if (mode == Mode::Running) stopTherapy(); else startTherapy();
  }
  if (encoderTakePress() && alarms) {         // short press with alarm: silence 2 min, clear latched alarms
    muted = true; muteUntil = millis() + 120000;
    uint16_t keep = (safetyTripped() ? AL_HW_TRIP : 0) | (alarms & AL_FLOW);
    alarms = keep;
    if (mode == Mode::Fault && !(alarms & AL_STOPPING)) mode = Mode::Standby;
  }
  if (muted && millis() > muteUntil) muted = false;

  // alarm sound: high priority = fast double beep, warning = slow beep
  uint32_t t = millis() % 1000;
  if (alarms && !muted) {
    bool high = alarms & AL_STOPPING;
    bool on = high ? (t < 120 || (t > 240 && t < 360)) : (t < 80);
    buzzerTone(on ? 4000 : 0);
  } else buzzerTone(0);
}

static void sendBle() {
  if (!bleConnected || !dataChr) return;
  char json[200];
  snprintf(json, sizeof json,
           "{\"pressure\":%.2f,\"pressure_red\":%.2f,\"flow\":%.2f,\"vol\":%.0f,\"rpm\":%lu,"
           "\"pwm\":%d,\"set\":%.1f,\"mode\":%d,\"alarm\":%u}",
           pr.pri, pr.red, fl.ok ? fl.slm : 0.0f, lastBreathMl, (unsigned long)blowerRpm(),
           (int)(blowerDuty() * 255), setpoint, (int)mode, alarms);
  dataChr->setValue(json);
  dataChr->notify();
}

// ======================= Setup / loop =======================
void setup() {
  blowerBegin();                 // PWM pin LOW before anything else
  Serial.begin(115200);
  pinMode(PIN_SAFETY_TRIP, INPUT);
  buzzerBegin();
  encoderBegin();

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
  oled.begin();
  oled.setBusClock(400000);
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x10_tf);
  oled.drawStr(0, 20, "SiPAP Rev C");
  oled.drawStr(0, 36, "zero calibration...");
  oled.sendBuffer();

  sfmBegin();

  // Self-test: hold the encoder knob at power-on, or send "TEST" within 2 s.
  bool wantTest = digitalRead(PIN_ENC_SW) == LOW;
  Serial.println("Send TEST within 2 s for the board self-test.");
  String in;
  for (uint32_t t0 = millis(); !wantTest && millis() - t0 < 2000;) {
    while (Serial.available()) in += (char)Serial.read();
    wantTest = in.indexOf("TEST") >= 0;
    delay(10);
  }
  if (wantTest) runSelfTest(oled);   // never returns

  if (!pressureBegin()) raise(AL_ZERO);
  Serial.printf("zero: pri %.3f V  red %.3f V\n", pressureZeroV(true), pressureZeroV(false));

  bleBegin();

  esp_task_wdt_config_t wdt = {.timeout_ms = WDT_MS, .idle_core_mask = 0, .trigger_panic = true};
  esp_task_wdt_reconfigure(&wdt);
  esp_task_wdt_add(NULL);

  buzzerBeep(100);
  Serial.println("SiPAP ready. Long-press encoder = start/stop.");
}

void loop() {
  static uint32_t tCtl = 0, tDisp = 0, tBle = 0;
  uint32_t now = millis();

  if (now - tCtl >= CONTROL_MS) { tCtl = now; controlStep(); }

  char cmd[64] = "";
  portENTER_CRITICAL(&cmdMux);
  if (cmdReady) { memcpy(cmd, pendingCmd, sizeof cmd); cmdReady = false; }
  portEXIT_CRITICAL(&cmdMux);
  if (cmd[0]) handleCommand(String(cmd));

  handleUi();
  if (now - tDisp >= DISPLAY_MS) { tDisp = now; drawDisplay(); }
  if (now - tBle >= BLE_MS) { tBle = now; sendBle(); }

  esp_task_wdt_reset();
}
