#define DEBUG 0

#if DEBUG
  #define DBG(x) Serial.print(x)
  #define DBGLN(x) Serial.println(x)
#else
  #define DBG(x)
  #define DBGLN(x)
#endif

#include <ArduinoBLE.h>

// =====================================================
// ================= PRESSURE SENSOR ===================
// =====================================================

#define PRESSURE_PIN A0

float pressure = 0;

const unsigned sensorMs = 10;
const unsigned bleSendMs = 200;

// =====================================================
// ==================== BLOWER =========================
// =====================================================

int pwmPin = 3;
int fgPin = 2;

volatile int pulseCount = 0;

int blowerPwm = 0;
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

bool notifyEnabled = false;

unsigned long sensorPrev = 0;
unsigned long lastBleSend = 0;

// =====================================================
// ==================== INTERRUPT ======================
// =====================================================

void countPulse()
{
  pulseCount++;
}

// =====================================================
// ================= PRESSURE READ =====================
// =====================================================

void readPressure()
{
  int adc = analogRead(PRESSURE_PIN);

  float voltage =
    adc * 3.3 / 4095.0;

  // Chia áp 10k/20k
  float sensorVoltage =
    voltage * 1.5;

  pressure =
    ((sensorVoltage / 5.0) - 0.04) / 0.09;

  if (pressure < 0)
    pressure = 0;
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

    currentRPM = (count / 2) * 60;

    DBG("RPM: ");
    DBGLN(currentRPM);
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

  DBG("PWM SET: ");
  DBGLN(blowerPwm);
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
    "\"pressure\":" + String(pressure, 2) + "," +
    "\"rpm\":" + String(currentRPM) + "," +
    "\"pwm\":" + String(blowerPwm) +
    "}";

  bool success =
    sensorCharacteristic.writeValue(sensorData);

  if (success)
  {
    DBG("SEND: ");
    DBGLN(sensorData);
  }
}

// =====================================================
// ====================== SETUP ========================
// =====================================================

void setup()
{
#if DEBUG
  Serial.begin(115200);
  delay(1000);
#endif

  analogReadResolution(12);

  pinMode(PRESSURE_PIN, INPUT);

  pinMode(pwmPin, OUTPUT);

  pinMode(fgPin, INPUT);

  attachInterrupt(
    digitalPinToInterrupt(fgPin),
    countPulse,
    RISING
  );

  analogWrite(pwmPin, 0);

  pinMode(LED_BUILTIN, OUTPUT);

  if (!BLE.begin())
  {
    DBGLN("BLE FAILED");

    while (1);
  }

  BLE.setLocalName("SIPAP");
  BLE.setDeviceName("SIPAP");

  BLE.setAdvertisedService(cpapService);

  cpapService.addCharacteristic(
    sensorCharacteristic
  );

  cpapService.addCharacteristic(
    actionCharacteristic
  );

  BLE.addService(cpapService);

  sensorCharacteristic.writeValue("READY");

  BLE.advertise();

  DBGLN("BLE READY");
}

// =====================================================
// ======================= LOOP ========================
// =====================================================

void loop()
{
  BLEDevice central = BLE.central();

  if (central)
  {
    DBG("CONNECTED: ");
    DBGLN(central.address());

    while (central.connected())
    {
      BLE.poll();

      unsigned long now =
        millis();

      // Pressure

      if (now - sensorPrev >= sensorMs)
      {
        sensorPrev = now;

        readPressure();
      }

      // RPM

      readRPM();

      // Notify

      notifyEnabled =
        sensorCharacteristic.subscribed();

      if (notifyEnabled)
      {
        if (now - lastBleSend >= bleSendMs)
        {
          lastBleSend = now;

          sendBleData();
        }
      }

      // Commands

      if (actionCharacteristic.written())
      {
        String action =
          actionCharacteristic.value();

        DBGLN(action);

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
        else if (action == "START")
        {
          setBlowerPWM(55);
        }
        else if (action == "STOP")
        {
          setBlowerPWM(0);
        }
        else if (
          action.startsWith("PWM:")
        )
        {
          int pwm =
            action.substring(4)
            .toInt();

          setBlowerPWM(pwm);
        }
      }

      delay(10);
    }

    DBGLN("DISCONNECTED");

    notifyEnabled = false;

    setBlowerPWM(0);
  }
}