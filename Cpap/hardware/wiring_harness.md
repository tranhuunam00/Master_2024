# SiPAP Rev C — sơ đồ dây nối ngoài PCB (wiring harness)

Áp dụng cho board `SiPAP_PCB.PcbDoc` Rev C. Chân lấy từ schematic `SiPAP_Main_RevC.SchDoc`. Pin 1 của mỗi header J2/J3/U4/U5 được **đánh số "1" trên silkscreen** ngay cạnh chân (pad header đều tròn) — luôn đối chiếu trước khi bấm cáp.

**Kiểm tự động:** `python CPAP_VSSM/tools/check_harness.py` mô phỏng từng dây (chân board → net → chân thiết bị theo datasheet): loại chân, dải điện áp nguồn (trường hợp xấu nhất), mức logic, TX→RX, không 2 ngõ ra chung dây. Kết quả 2026-11-19: **17 PASS, 0 FAIL, 3 OPEN** (3 dây blower — chờ datasheet blower). Đã thử cố tình đấu sai (OLED không đảo dây, UART TX→TX): checker báo FAIL đúng 4 dây.

Header trên board — **đã đối chiếu datasheet gốc Würth** (lưu ở `Cpap/tools/datasheets/Wurth_*.pdf`): J2 = **61300421121** (WR-PHD dual 2x2), J3/U4 = **61300611121** (1x6), U5 = **61300411121** (1x4). Chân vuông 0.64 mm, bước 2.54 mm; Würth khuyên lỗ Ø1.10 ±0.15 mm (0.95–1.25) — board khoan **1.00 mm**, pad 1.5 mm: đạt. 3 A, 250 V, −40…105 °C (nhà phân phối RS).

```
 Adapter 12V ──► J1 ──F1──Q4──┬──► U8 (5V) ──► MPXV5010 x2, LM393, SFM3300 (qua Q2)
 (2.1 mm, + giữa)             ├──► U9 (3V3) ─► ESP32, OLED, pull-up
                              └──► Q1 (cắt quá áp) ──► C25 ──► J2 ──► BLOWER
                                                         J2 ◄── PWM / FG
 ESP32 ── I2C ──┬── U4 ──cáp── SFM3300-D (trên đường khí)
                └── U5 ──cáp── OLED 0.96"
 U2 / U3 (cổng áp) ──ống silicon── T ── điểm đo áp ở đầu ra máy
```

## 1. J1 — nguồn 12 V (jack DC PJ-102AH, mép trái board)
| Mục | Giá trị |
|---|---|
| Adapter | 12 V DC, **≥ 3.5 A** (blower ~3 A + mạch ~0.5 A), jack 5.5/2.1 mm, **dương ở giữa** |
| Bảo vệ trên board | F1 MF-R400 (4 A), Q4 chống ngược cực, D2 TVS SMBJ13A (kẹp ≤ 28 V) |
| Lưu ý | Adapter phải có chứng nhận an toàn (IEC 62368-1 / 60601-1 nếu dùng y tế) |

## 2. J2 — blower (header 2x2, 2.54 mm)
| Chân J2 | Net | Nối tới blower | Ghi chú |
|---|---|---|---|
| 1 | BLOWER_V12 | +12 V (dây đỏ) | Đã qua Q1 — mất khi mạch an toàn cắt; C25 220 µF ngay cạnh |
| 2 | GND | GND (dây đen) | |
| 3 | BLOWER_PWM | PWM / speed in | Mức 3.3 V từ ESP32 IO3, 25 kHz (đổi `BLOWER_PWM_HZ` theo datasheet blower) |
| 4 | BLOWER_FG | FG / tacho out | Kéo lên 3V3 qua R12 10k — **blower phải ra FG kiểu open-collector**. Nếu FG ra 5/12 V: thêm chia áp hoặc diode, KHÔNG cắm thẳng |

- Vị trí chân J2 nhìn từ mặt trên board (pin 1 = chân có chữ "1" bên trái, góc trên-trái), giống cách đánh số của Würth (1-2 cùng một cột):
  ```
   [1] 12V    [3] PWM
   [2] GND    [4] FG
  ```
- Dây: 12V/GND **AWG20 (≥ 0.5 mm²)**, dài ≤ 30 cm; PWM/FG AWG26, xoắn với GND nếu > 15 cm.
- Đầu cái: housing 2x2 2.54 mm (Dupont / Harwin M20) có khoá nếu được — blower rung làm lỏng đầu cắm.
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
- **Đầu nối:** Sensirion không bán sẵn đầu cắm cho cảm biến (DS mục 3.2). Cách làm: hàn trực tiếp 6 dây (hoặc 1 header 1x6 2.54 mm) vào 6 pad của cảm biến, đầu kia bấm housing 1x6 2.54 mm cắm vào U4.
- **Cáp (DS mục 4.8):** càng ngắn càng tốt, **tối đa 30 cm; dài hơn 10 cm thì BẮT BUỘC dùng cáp chống nhiễu (shield) cho SDA và SCL**, shield nối GND ở phía board. Khuyên đặt cảm biến ≤ 10 cm từ U4 để dùng dây thường. Firmware chạy I2C 400 kHz; nếu cáp dài hơn hoặc báo lỗi `FLOW SENSOR`, hạ xuống 100 kHz trong `sipap_v1.ino` (`Wire.begin(..., 100000)`, `oled.setBusClock(100000)`) và tăng `DISPLAY_MS` lên 500.
- Lắp **trên đường khí chính** (cổng côn 22 mm), mũi tên trên thân hướng **về phía bệnh nhân**. Cách blower/humidifier ≥ 1 đoạn ống thẳng để dòng ổn định; tránh hơi nước đọng vào cảm biến (đặt trước humidifier hoặc có bẫy nước).

## 4. U5 — màn OLED 0.96" SSD1306 I2C (header 1x4)
**Mẫu chọn: ProtoSupplies DSP-8** (SSD1306 128x64, I2C 0x3C, có ổn áp 3.3 V trên module, nhận 3.3 V hoặc 5 V). Thứ tự chân module: **VCC · GND · SCL · SDA** — ngược 2 chân đầu so với U5, nên cáp **đảo dây 1 và 2**:

| Chân U5 | Net | Chân module DSP-8 |
|---|---|---|
| 1 | GND | **2** GND |
| 2 | 3V3 | **1** VCC |
| 3 | I2C_SCL | 3 SCL |
| 4 | I2C_SDA | 4 SDA |

- Dùng dây Dupont cái-cái rời (không dùng cáp dẹt liền) để đảo được dây 1–2.
- Nếu mua module khác: đọc chữ in cạnh 4 chân. Có loại GND · VCC · SCL · SDA (khi đó cáp đi thẳng) và loại VCC · GND · SCL · SDA (đảo như bảng trên). Sửa bảng `HARNESS` trong `check_harness.py` rồi chạy lại trước khi cắm. Cắm ngược VCC/GND có thể hỏng module.
- Firmware kiểm cả địa chỉ 0x3C và 0x3D trong self-test.

## 5. J3 — cổng nạp UART (header 1x6, mép phải)
| Chân J3 | Net | Adapter USB-UART 3.3 V |
|---|---|---|
| 1 | GND | GND |
| 2 | 3V3 | **không nối** nếu board đã cấp 12 V (chỉ nối khi nạp mà không có 12 V, adapter phải cấp được ≥ 500 mA) |
| 3 | UART_TX (ESP32 TXD0) | RX |
| 4 | UART_RX (ESP32 RXD0) | TX |
| 5 | EN | RTS (hoặc nút reset tay: chạm EN xuống GND) |
| 6 | IO0 / BOOT | DTR (hoặc giữ IO0 xuống GND khi reset để vào chế độ nạp) |

- Chỉ dùng adapter **3.3 V** (CP2102 / CH340 đặt 3.3 V). Adapter 5 V làm hỏng ESP32.
- Self-test: giữ núm encoder khi cấp nguồn, hoặc gửi `TEST` qua Serial 115200 trong 2 s đầu.

## 6. Ống áp — U2 (chính), U3 (dự phòng)
- Ống silicon **ID 3 mm** (cổng cảm biến Ø3.17 mm), dài ≤ 30 cm, không gập.
- Hai ống nối chung qua **1 cút chữ T** tới **1 điểm đo ở đầu ra máy** (sau humidifier, trước ống thở) — cả 2 cảm biến phải đo cùng một áp, nếu không firmware báo `SENSORS DISAGREE`.
- Điểm đo có bẫy nước / đặt cao hơn đường khí để hơi nước không chảy vào cảm biến.
- U3 là đầu vào của mạch cắt quá áp phần cứng (25 cmH₂O): **tuột ống U3 = mất lớp bảo vệ phần cứng**. Dùng ống có khoá hoặc kẹp dây rút.

## 7. Encoder và còi
- Encoder U6 và còi BZ1 hàn trực tiếp trên board — không cần dây. Trục encoder xuyên mặt vỏ (lỗ M7), còi cần lỗ thoát âm trên vỏ ngay phía trên.

## 8. Kiểm tra harness trước khi cấp điện
1. Đo thông mạch từng dây theo bảng trên; không chập giữa 12V–GND, 5V–GND, 3V3–GND.
2. Cắm OLED, SFM3300 trước; chạy self-test mục 1–7.
3. Cắm blower sau cùng, chạy self-test mục 8–9.
