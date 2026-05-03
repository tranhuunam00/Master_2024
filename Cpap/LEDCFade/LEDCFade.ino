#include <Wire.h>
#include <math.h>

#define POLYNOMIAL 0x31
#define SFM3300_ADDR 0x40

// ====== CONFIG ======
const float FLOW_BIAS = 0.3;    // ngưỡng bỏ nhiễu
const float FILTER_ALPHA = 0.2; // low-pass filter
const int SIGN_CONFIRM = 4;     // số mẫu xác nhận đảo chiều

const unsigned mms = 10;  // đo mỗi 10ms
const unsigned dms = 500; // in mỗi 500ms

// ====== CRC ======
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

// ====== GLOBAL ======
float flow = 0;
float flow_prev = 0;
float volume = 0; // ml

unsigned long mt_prev = 0;

int current_sign = 0;
int sign_counter = 0;

bool crc_error = false;

// ====== SIGN FUNCTION ======
int get_sign(float f)
{
  if (f > FLOW_BIAS)
    return 1;
  if (f < -FLOW_BIAS)
    return -1;
  return 0;
}

// ====== UPDATE SIGN (debounce 3 mẫu) ======
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

    // in volume trước khi reset
    Serial.print("Volume cycle: ");
    Serial.println(volume);

    current_sign = new_sign;
    sign_counter = 0;
    volume = 0;
  }
}

// ====== READ SENSOR ======
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

  float new_flow = ((float)raw - 32768.0) / 120.0;

  // ===== FILTER =====
  new_flow = FILTER_ALPHA * new_flow + (1 - FILTER_ALPHA) * flow_prev;

  // ===== DEAD ZONE =====
  if (fabs(new_flow) < FLOW_BIAS)
    new_flow = 0;

  // ===== SIGN =====
  int new_sign = get_sign(new_flow);
  update_sign(new_sign);

  // ===== TIME =====
  unsigned long mt = millis();
  float dt = (mt - mt_prev); // ms
  mt_prev = mt;

  // ===== VOLUME (ml) =====
  // slm → ml/ms
  volume += (flow_prev + new_flow) / 2.0 * (dt / 60000.0) * 1000.0;

  flow_prev = new_flow;
  flow = new_flow;
}

// ====== DISPLAY ======
void display_data(bool force = false)
{
  if (fabs(volume) > 5 || force)
  {
    Serial.print("Flow: ");
    Serial.print(flow);

    Serial.print(" | Volume: ");
    Serial.print(volume);

    Serial.print(" | Sign: ");
    Serial.print(current_sign);

    if (crc_error)
      Serial.print(" | CRC_ERR");

    Serial.println();
  }
}

// ====== SETUP ======
void setup()
{
  Wire.begin();
  Serial.begin(115200);
  delay(500);

  // reset sensor
  Wire.beginTransmission(SFM3300_ADDR);
  Wire.write(0x20);
  Wire.write(0x00);
  Wire.endTransmission();
  delay(100);

  // start measurement
  Wire.beginTransmission(SFM3300_ADDR);
  Wire.write(0x10);
  Wire.write(0x00);
  Wire.endTransmission();
  delay(100);

  mt_prev = millis();

  Serial.println("System ready...");
}

// ====== LOOP ======
unsigned long ms_prev = 0;
unsigned long ms_display = 0;

void loop()
{
  unsigned long now = millis();

  if (now - ms_prev >= mms)
  {
    ms_prev = now;
    SFM_measure();
  }

  if (now - ms_display >= dms)
  {
    ms_display = now;
    display_data(true);
  }
}