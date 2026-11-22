# SiPAP Rev C — sơ đồ dây nối ngoài PCB (wiring harness)

Áp dụng cho board `SiPAP_PCB.PcbDoc` Rev C **layout v3** (2026-11-21). Chân lấy từ schematic `SiPAP_Main_RevC.SchDoc`. Pin 1 của mỗi header J2/J3/U4/U5 có **chấm tròn trên silkscreen** ngay cạnh chân (pad header đều tròn); J3 có thêm nhãn từng chân GND · 3V3 · TX · RX · EN · IO0 — luôn đối chiếu trước khi bấm cáp.

**Layout v3 đổi gán GPIO** (thứ tự chân header không đổi): BLOWER_PWM = IO10, BLOWER_FG = IO11, BUZZER = IO12, SAFETY_TRIP = IO9, I2C SDA/SCL = IO2/IO1, encoder A/B/SW = IO47/IO21/IO48; giữ nguyên PRESS_PRI = IO4, PRESS_RED = IO5, SFM_EN = IO16. Firmware `firmware/sipap_v1/config.h` đã sửa theo.

**Kiểm tự động:** `python CPAP_VSSM/tools/check_harness.py` mô phỏng từng dây (chân board → net → chân thiết bị theo datasheet): loại chân, dải điện áp nguồn (trường hợp xấu nhất), mức logic, TX→RX, không 2 ngõ ra chung dây. Kết quả 2026-11-22 (OLED Adafruit 326, mạch nạp ESP-Prog): **23 PASS, 0 FAIL, 1 OPEN** (ngõ ra FG của blower — bản vẽ hãng không ghi kiểu ngõ ra). Đã thử cố tình đấu sai (đảo VCC/GND của OLED, đảo TX/RX ở J3): checker báo FAIL đúng 4 dây.

Header trên board — **đã đối chiếu datasheet gốc Würth** (lưu ở `Cpap/tools/datasheets/Wurth_*.pdf`): J2 = **61300421121** (WR-PHD dual 2x2), J3/U4 = **61300611121** (1x6), U5 = **61300411121** (1x4). Chân vuông 0.64 mm, bước 2.54 mm; Würth khuyên lỗ Ø1.10 ±0.15 mm (0.95–1.25) — board khoan **1.00 mm**, pad 1.5 mm: đạt. 3 A, 250 V, −40…105 °C (nhà phân phối RS).

```
 Adapter 12V ──► J1 ──F1──Q4──┬──► U8 (5V) ──► MPXV5010 x2, LM393, SFM3300 (qua Q2)
 (chân Ø2.0, + giữa)          ├──► U9 (3V3) ─► ESP32, OLED, pull-up
                              └──► Q1 (cắt quá áp) ──► C25 ──► J2 ──► BLOWER
                                                         J2 ◄── PWM / FG
 ESP32 ── I2C ──┬── U4 ──cáp── SFM3300-D (trên đường khí)
                └── U5 ──cáp── OLED 0.96"
 U2 / U3 (cổng áp) ──ống silicon── T ── điểm đo áp ở đầu ra máy
```

## 1. J1 — nguồn 12 V (jack DC PJ-102AH, mép trái board)
| Mục | Giá trị |
|---|---|
| Adapter | 12 V DC, **≥ 3.5 A** (giữ như cũ để có dự phòng; blower Wonsmart WS4540-12-NZ03 ăn 1.6 A ±15 % khi thổi tự do theo bản vẽ hãng, cộng mạch), **dương ở giữa**. Đầu cắm phải khớp jack PJ-102AH: chân giữa Ø2.0 mm (Same Sky PJ-102AH DS tr.2), jack chịu 24 V / 5.0 A (DS tr.1) |
| Bảo vệ trên board | F1 MF-R400 (4 A), Q4 chống ngược cực, D2 TVS SMBJ13A (kẹp ≤ 28 V) |
| Mẫu chọn | **Mean Well GSM60B12-P1J** (E4): 12 V ±3.0 %, 5 A, Class II (ổ C8, không cần dây đất), 2×MOPP, IEC 60601-1 và 60601-1-11 (GSM60B DS tr.1–2). Đầu P1J 2.1 × 5.5 × 11 mm, dương giữa; thêm đầu chuyển **DC PLUG-P1J-P1I** 2.1 × 5.5 × 9.5 mm (E4b, Mean Well DC_plug tr.1) cho đúng độ sâu 9.5 mm của jack (PJ-102AH DS tr.2). Dây nguồn C7 theo ổ cắm nơi dùng (E4c) |
| Phải thử | Không hãng nào ghi đầu 2.1 mm kẹp chắc chân 2.0 mm: cắm thử, đầu không được lắc, đo J1 11.6–12.4 V khi blower chạy |

## 2. J2 — blower (header 2x2, 2.54 mm)
| Chân J2 | Net | Nối tới blower | Ghi chú |
|---|---|---|---|
| 1 | BLOWER_V12 | **Đỏ (pin4 +12 V)** | Đã qua Q1 — mất khi mạch an toàn cắt; C25 220 µF ngay cạnh |
| 2 | GND | **Đen (pin3 GND)** | |
| 3 | BLOWER_VSR | **Vàng (pin1 VSR)** | Điện áp tốc độ **0–3.3 V một chiều**: PWM 20 kHz từ IO10 qua lọc RC R28 1 k + C26 1 µF (fc 159 Hz) trên board |
| 4 | BLOWER_FG_IN | **Trắng (pin2 FG)** | 1 xung/vòng (rpm = Hz × 60). Qua R29 1 k nối tiếp → IO11, kéo lên 3V3 bằng R12 10 k. Kiểu ngõ ra FG chưa ghi trong bản vẽ hãng — đo thử FG ≤ 3.3 V trước khi chạy lâu |

- Vị trí chân J2 nhìn từ mặt trên board (pin 1 = chân có chấm tròn silkscreen, góc trên-trái; chữ "BLOWER" in phía dưới J2), giống cách đánh số của Würth (1-2 cùng một cột):
  ```
   [1] 12V    [3] PWM
   [2] GND    [4] FG
  ```
- Trong vỏ v0.3 đường dây blower → J2 dài ~172 mm (qua grommet vách hộp tiêu âm, lên sau mép PCB, dọc mép trái) — dây zin 150 ± 20 mm nên **hàn nối thêm dây Belden 8444 22 AWG** (cũng là chỗ đổi cỡ dây: cos 90119 chỉ nhận 22–24 AWG).
- Đầu cái (E6): **2 housing Molex 90123-0102** (1x2, rộng ≤ 2.54 mm, ghép được 2 chiều — catalog 90123 tr.1) đặt cạnh nhau: housing A vào J2.1/J2.2 (12V, GND), housing B vào J2.3/J2.4 (PWM, FG). Ghép 2 housing lại và đánh dấu pin 1 — để rời thì có thể cắm đổi chỗ, đưa 12 V vào chân VSR. Cos **Molex 90119-2110** (mạ vàng, dây 22–24 AWG, vỏ dây Ø1.02–1.47 mm, cắm chân vuông 0.635 mm — SD-90119), kìm bấm Molex 63825-8100 (E11).
- Dây nối thêm: **Belden 8444** (4 × 22 AWG, vỏ Ø1.3 mm, 2.4 A/lõi ở 25 °C — techdata tr.1) ≈ 6 cm, hàn vào 4 dây 20 AWG zin của blower, bọc co nhiệt. 2.4 A > 1.84 A (1.6 A +15 % lúc thổi tự do).
- **Trước khi cắm lần đầu**: đo mức FG khi blower chạy riêng, kiểm tra blower nhận PWM 3.3 V (mục 1 trong test).

## 3. U4 — cảm biến lưu lượng SFM3300-D (header 1x6)
| Chân U4 | Net | Chân SFM3300 | Ghi chú |
|---|---|---|---|
| 1 | GND | HEAT_GND / GND | |
| 2 | SFM_5V | VDD 5 V | Qua công tắc Q2 (IO16) — firmware tắt/bật để reset cứng |
| 3 | I2C_SCL | SCL | Pull-up 4.7k → 3V3 trên board |
| 4 | GND | GND | |
| 5 | I2C_SDA | SDA | Pull-up 4.7k → 3V3 trên board |
| 6 | — | HEAT | Để trống (heater 51 Ω, chỉ dùng khi cần chống đọng hơi) |

- **Đã đối chiếu datasheet** Sensirion SFM3300 v1.4 mục 2.2 (pad layout): 6 pad bước 2.54 mm theo thứ tự HEAT_GND · VDD · SCK · GND · DATA · HEAT — trùng 1:1 với U4 chân 1→6, nên cáp đi **thẳng, không đảo dây**. Kiểm khi lắp (không cần đoán): (1) đo thông mạch từng dây từ U4 tới pad cảm biến theo bảng; (2) chạy self-test — mục 4 phải thấy cảm biến ở 0x40, tắt được khi IO16 = 0 và đọc lưu lượng có CRC đúng. **Cắm ngược đầu cáp sẽ đưa 5 V vào chân DATA và GND vào HEAT** — luôn đo thông mạch trước khi cấp điện lần đầu, và đánh dấu pin 1 trên cả 2 đầu cáp. VDD 5 V ±5 % (board 4.94–5.22 V), mức logic cao tối thiểu 2.5 V (pull-up 3V3 đạt).
- **Đầu nối (E7):** Sensirion không bán sẵn đầu cắm cho cảm biến (DS mục 3.2). Đầu cảm biến hàn thẳng vào pad; đầu board bấm housing **Molex 90156-0146** (1x6) + cos 90119-2110. Cáp **Belden 9534** (4 × 24 AWG, chống nhiễu foil 100 % + dây drain, vỏ lõi Ø1.1 mm, 110 pF/m lõi–lõi, 210 pF/m lõi–vỏ — techdata tr.1): 4 lõi vào U4.2 VDD, U4.3 SCK, U4.4 GND, U4.5 DATA; dây drain vào U4.1 ở đầu board, không nối ở đầu cảm biến. Drain là dây trần: lồng ống cách điện Ø1.0–1.5 mm trước khi bấm để tai cos kẹp được.
- **Trong vỏ v0.3 cáp này dài ~147 mm** (SFM nằm sau PCB, U4 ở góc trước-phải) → **phải dùng cáp chống nhiễu** cho SDA/SCL.
- **Cáp (DS mục 4.8):** càng ngắn càng tốt, **tối đa 30 cm; dài hơn 10 cm thì BẮT BUỘC dùng cáp chống nhiễu (shield) cho SDA và SCL**, shield nối GND ở phía board. Khuyên đặt cảm biến ≤ 10 cm từ U4 để dùng dây thường. Firmware chạy I2C 400 kHz; nếu cáp dài hơn hoặc báo lỗi `FLOW SENSOR`, hạ xuống 100 kHz trong `sipap_v1.ino` (`Wire.begin(..., 100000)`, `oled.setBusClock(100000)`) và tăng `DISPLAY_MS` lên 500.
- Lắp **trên đường khí chính** (cổng côn 22 mm), mũi tên trên thân hướng **về phía bệnh nhân**. Cách blower/humidifier ≥ 1 đoạn ống thẳng để dòng ổn định; tránh hơi nước đọng vào cảm biến (đặt trước humidifier hoặc có bẫy nước).

## 4. U5 — màn OLED 0.96" SSD1306 I2C (header 1x4)
**Mẫu chọn: Adafruit 326** (0.96" 128x64 STEMMA QT, E3) — có tài liệu hãng: nguồn Vin 3–5 V qua ổn áp trên module, an toàn với logic 3.3 V (guide tr.7); bản STEMMA QT mặc định I2C, không cần nối RST (tr.17, tr.49). Header JP2 của module (sơ đồ hãng tr.79): **1 SDA · 2 SCL · 3 DC/SA0 · 4 RST · 5 CS · 6 3.3V · 7 VCC · 8 GND**. Để hở DC/SA0 thì địa chỉ là **0x3D** (0x3C khi DC/SA0 nối GND).

| Chân U5 | Net | Chân module (JP2) |
|---|---|---|
| 1 | GND | **8** GND |
| 2 | 3V3 | **7** VCC |
| 3 | I2C_SCL | **2** SCL |
| 4 | I2C_SDA | **1** SDA |

- Cáp (E8): đầu U5 housing **Molex 90156-0144** (1x4), đầu module housing **Molex 90123-0108** (1x8) chỉ lắp cos ở hốc 1, 2, 7, 8; cos 90119-2110; dây Belden 9534. Hốc 3–6 để trống.
- Nếu dùng module khác: đọc chữ in cạnh chân, sửa bảng `HARNESS` trong `check_harness.py` rồi chạy lại trước khi cắm. Cắm ngược VCC/GND có thể hỏng module.
- Firmware dò 0x3C trước, không có thì dùng 0x3D (`setup()` trong `sipap_v1.ino`); self-test cũng kiểm cả hai địa chỉ.

## 5. J3 — cổng nạp UART (header 1x6, mép phải)
Layout v3: J3 nằm **dọc** mép phải, chân 1 (GND, có chấm tròn) ở **trên cùng**, chân 6 (IO0) dưới cùng; nhãn từng chân in bên trái header, chữ "UART" bên phải. Thứ tự chân không đổi so với Rev C:

Mạch nạp chọn: **Espressif ESP-Prog** (E5), cổng PROG 2.54 mm 2x3: 1 ESP_EN · 2 VDD · 3 ESP_TXD · 4 GND · 5 ESP_RXD · 6 ESP_IO0; tín hiệu RX/TX luôn 3.3 V (user guide). Theo sơ đồ ESP-Prog V2.1, ESP_TXD0 là chân TX của ESP (đi vào FT_RXD qua R21), nên nối **cùng tên**:

| Chân J3 | Net | ESP-Prog PROG |
|---|---|---|
| 1 | GND | 4 GND |
| 2 | 3V3 | **không nối** (chân 2 VDD để trống; board lấy nguồn từ J1) |
| 3 | UART_TX (ESP32 TXD0) | 3 ESP_TXD |
| 4 | UART_RX (ESP32 RXD0) | 5 ESP_RXD |
| 5 | EN | 1 ESP_EN |
| 6 | IO0 / BOOT | 6 ESP_IO0 |

- Cáp (E5b): đầu J3 housing **Molex 90156-0146** (1x6), đầu ESP-Prog housing **Molex 90142-0006** (2x3), 10 cos 90119-2110, dây **Belden 9535** (5 × 24 AWG, chống nhiễu) ≤ 30 cm. Hộp ESP-Prog chỉ có cáp dẹt 1.27 mm nên phải tự làm cáp này.
- Không dùng adapter USB-UART 5 V: làm hỏng ESP32.
- Self-test: giữ núm encoder khi cấp nguồn, hoặc gửi `TEST` qua Serial 115200 trong 2 s đầu.

## 6. Ống áp — U2 (chính), U3 (dự phòng)
- Cổng cảm biến có đường kính ngoài **2.92–3.17 mm** (kích thước W, bản vẽ case 482A, datasheet MPX5010 trang 7). Ống silicon ID 3 mm có thể lỏng trên cổng 2.92 mm: chọn ống lòng nhỏ hơn hoặc thêm kẹp, nhất là ống của U3. Dài ≤ 30 cm, không gập. Danh sách mua: `CPAP_VSSM/Fab_RevC_v3/BOM_RevC_v3_offboard.csv` (mục E9 — **chưa chốt mã**: ứng viên ống Tygon ND 100-65 3/32 × 5/32 in và cút T Nordson T20-6005, lòng 2.4 mm, nhưng chưa lưu được datasheet).
- Hai ống nối chung qua **1 cút chữ T** tới **1 điểm đo: núm Ø3.4 trên dock** (vỏ v0.3 — giữa đầu ra SFM3300 và vòi bình làm ẩm). Cả 2 cảm biến phải đo cùng một áp, nếu không firmware báo `SENSORS DISAGREE`. Điểm đo nằm trước bình nên tháo bình không phải rút ống; tổn áp qua bình đo một lần trên bàn, > 0.2 cmH₂O thì bù trong firmware. Chiều dài theo mô hình: 43 mm (núm → T) + 12 mm mỗi nhánh, cộng đoạn chồng lên núm.
- Điểm đo có bẫy nước / đặt cao hơn đường khí để hơi nước không chảy vào cảm biến.
- U3 là đầu vào của mạch cắt quá áp phần cứng (25 cmH₂O): **tuột ống U3 = mất lớp bảo vệ phần cứng**. Dùng ống có khoá hoặc kẹp dây rút.

## 7. Encoder và còi
- Encoder U6 và còi BZ1 hàn trực tiếp trên board — không cần dây. Trục encoder xuyên mặt vỏ (lỗ M7), còi cần lỗ thoát âm trên vỏ ngay phía trên.

## 8. Kiểm tra harness trước khi cấp điện
1. Đo thông mạch từng dây theo bảng trên; không chập giữa 12V–GND, 5V–GND, 3V3–GND.
2. Cắm OLED, SFM3300 trước; chạy self-test mục 1–7.
3. Cắm blower sau cùng, chạy self-test mục 8–9.

## Blower thật: Wonsmart WS4540-12-NZ03 (bản vẽ hãng rev 2023-07-12)
- 12 VDC, driver tích hợp; 1.6 A khi thổi tự do, 1.3 A tại điểm làm việc 3.5 kPa / 82 L/phút, 0.9 A khi bịt kín (5.0 kPa).
- Thân xoắn ốc Ø48, rộng 54 mm (kể cả ống ra), dày 45 mm (kể cả ống hút Ø10 × 10 mm). Ống ra Ø10 ngoài / Ø8 trong. Ống hút Ø7 trong.
- 4 lỗ bắt vít **Ø1.6 sâu 15 mm trên vòng Ø36.8** ở mặt motor → vít tự ren **ST2.2 × 13** qua long đen cao su.
- Dây 4 sợi 20AWG dài 150 mm: **20AWG to hơn đầu cos Dupont 2.54 (22–28AWG)** → hàn nối mỗi sợi sang dây 24AWG ngắn + co nhiệt rồi mới bấm cos, hoặc hàn thẳng vào header.
