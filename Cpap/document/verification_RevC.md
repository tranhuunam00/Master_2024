# SiPAP Rev C — kiểm chứng thiết kế & quy trình chứng minh mạch hoạt động

Ngày: 2026-11-19. Phạm vi: `CPAP_VSSM` (schematic `SiPAP_Main_RevC.SchDoc`, PCB `SiPAP_PCB.PcbDoc`, hồ sơ `Fab_RevC/`).

> **TRẠNG THÁI: CHƯA ĐƯỢC ĐẶT MẠCH.** Lần kiểm tra lại này tìm ra 2 lỗi trong file Gerber đã xuất (mục 2). Lỗi đã sửa trong Altium nhưng chưa lưu và chưa xuất lại Gerber — xem mục 5.

## 1. Kiểm tra tự động (đã chạy)

| Kiểm tra | Công cụ | Kết quả |
|---|---|---|
| ERC schematic | Altium ERC | 0 lỗi; 5 cảnh báo "no driving source" trên FB5/FB3/EN/GATE/COMP_REF (chân input cấp từ cầu chia điện trở — đúng thiết kế) |
| Schematic ↔ PCB linh kiện | Altium compare | 64/64 khớp |
| Schematic ↔ PCB từng chân | `tools/verify_revc.py` (độc lập với Altium) | 216 pad trên PCB so với 179 chân có net: **0 sai** |
| DRC điện | Altium DRC | 0 lỗi: clearance 10 mil, short, unrouted, width, hole, mask sliver, net antenna, height |
| DRC silkscreen | Altium DRC | 423 silk đè pad + 77 silk đè silk (viền silk của footprint cũ; xưởng tự cắt) |
| Va chạm thân linh kiện | `verify_revc.py` + datasheet | Không có va chạm vật lý (BZ1–U6 hở 1.1 mm, F1–J1 hở 1.2 mm; chỉ chồng khung silk) |
| Gerber / khoan | đọc lại file | Viền board 101.6 x 76.2 mm đúng; mm 4:4; khoan Excellon 4:3 mm có G85 |

## 2. Lỗi tìm ra khi kiểm tra lại

| # | Lỗi | Hậu quả | Trạng thái |
|---|---|---|---|
| 1 | 20 pad xuyên lỗ của J2, J3, U4, U5 (header 2.54) khai báo ở lớp Top thay vì MultiLayer (bản footprint cũ từ Rev B) | Mặt dưới không có đồng và không mở mask → **không hàn được header** | Đã sửa trên board (MultiLayer), **chưa lưu**; thư viện cần sửa tương tự |
| 2 | Khung silk/assembly/courtyard của J1 (PJ-102AH) vẽ đối xứng quanh gốc, trong khi thân jack nằm 0–14.4 mm (pad và khối 3D vẫn đúng) | Silk in lệch 7.2 mm ra ngoài mép board | Đang sửa trong thư viện thì script Altium crash |
| 3 | Footprint U6 trong schematic còn trỏ bản cũ `PEC11R-4220F-S0024` | Chỉ ảnh hưởng nếu chạy "Update PCB" (sẽ đề xuất đổi sang bản sai) | Chờ tắt tạm "Symbol pins vs pads" |

## 3. Tính toán kiểm chứng (theo datasheet)

### Nguồn 5V / 3V3 (TPS54202, fsw 500 kHz, VFB 0.596 V — TI SLVSD26C)
| | 5V (U8) | 3V3 (U9) |
|---|---|---|
| Vout danh định | 5.077 V | 3.293 V |
| Dải do sai số R 1% + VFB 1% | 4.94–5.22 V | 3.21–3.38 V |
| Ripple dòng cuộn cảm ΔI | 0.39 A | 0.48 A |
| Dòng đỉnh ở 2 A (max IC) | 2.20 A | 2.24 A |
| Isat cuộn cảm (Bourns SRN6045TA) | 3.8 A (150M) | 4.6 A (100M) |
| Ripple áp ra (2x22 µF, giảm 50% do DC bias) | ~4 mV | ~5 mV |
| Tải ước tính | ~30 mA (2x MPXV5010, LM393, SFM3300) | ~0.5 A đỉnh (ESP32 WiFi TX + OLED) |
→ Biên dòng cuộn cảm > 1.7x ngay cả ở tải tối đa của IC.

### Mạch an toàn độc lập firmware
- Ngưỡng cắt: Vref = 5 V × 10k/57k = 0.877 V tại PRESS_RED_ADC → cảm biến 1.316 V → **2.48 kPa = 25.3 cmH2O** (MPXV5010: Vout = Vs·(0.09·P + 0.04)).
- Ngưỡng **tỉ lệ theo nguồn 5V** (cả Vref và cảm biến cùng lấy 5V) → sai số nguồn 5V không làm lệch ngưỡng.
- Trễ (hysteresis) do R18 1M: 74 mV → **2.5 cmH2O** — blower chỉ bật lại khi áp tụt dưới ~22.8 cmH2O.
- Khi cắt: LM393 hở collector → GATE kéo lên 12V qua R13 → Vgs Q1 = 0 → blower tắt. Bình thường: GATE = 6V → Vgs = −6V → Q1 dẫn, Rds(on) < 17 mΩ (AOS AO4407A DS, VGS=−6V).
- Báo về ESP32: SAFETY_TRIP = 12 × 330k/1.33M = 2.98 V (< 3.6 V tuyệt đối của ESP32).

### Tổn hao ở dòng blower 3 A
| Linh kiện | Tổn hao | Đánh giá |
|---|---|---|
| Q1 AO4407A | 0.15 W | Tốt |
| D1 SS54 (chống ngược cực, nối tiếp) | ~1.65 W @3.3 A | **Rủi ro nhiệt** — SMC trên board 2 lớp có thể lên >100 °C. Đo nhiệt ở bước 5 |
| F1 MF-R300 | Ihold 3.0 A @23 °C, **2.49 A @40 °C, 2.31 A @50 °C** (Bourns MF-R derating) | **Sẽ ngắt nếu blower > ~2.3 A trong vỏ nóng** — đo dòng blower thật trước |

**Việc cần quyết khi có blower thật:** nếu dòng blower liên tục > 2 A → đổi F1 lên MF-R400/MF-R500 và cân nhắc thay D1 bằng P-MOSFET chống ngược cực (tổn hao ~0.1 W) ở Rev D.

## 4. Chứng minh mạch hoạt động — quy trình bring-up (từng bước, dừng nếu bước nào trượt)

Dụng cụ: nguồn DC có giới hạn dòng, đồng hồ vạn năng, oscilloscope, adapter USB-UART 3.3V, ống silicon + xi-lanh 50 ml + đồng hồ áp (hoặc cột nước) để tạo áp chuẩn.

| Bước | Làm gì | Đạt khi |
|---|---|---|
| 0. Nhìn | Soi kính lúp: cầu thiếc ở U1/U8/U9/U7, chiều D1/D2, chấm pin 1 | Không cầu thiếc, đúng chiều |
| 1. Đo trở (chưa cấp điện) | Đo 12V–GND, TP2 5V–GND, TP3 3V3–GND | > 1 kΩ (không chập) |
| 2. Cấp nguồn giới hạn | 12V qua J1, giới hạn 0.3 A, **chưa** cắm blower | Dòng < 100 mA, không nóng |
| 3. Đo nguồn | TP2, TP3 | 5V: 4.94–5.22 V; 3V3: 3.21–3.38 V |
| 4. Ripple | Scope tại C18/C21, đầu dò lò xo GND ngắn | < 30 mV p-p |
| 5. Ngược cực | Cấp −12V vào J1 (giới hạn 0.3 A) | Dòng ≈ 0, không gì nóng (D1 chặn) |
| 6. Nạp firmware | USB-UART vào J3, giữ BOOT | ESP32 nhận chip, nạp được |
| 7. Cảm biến | Firmware đọc U2, U3 ở 0 / 10 / 20 cmH2O (cột nước) | Sai khác < 0.5 cmH2O giữa 2 cảm biến và áp chuẩn |
| 8. **Mạch an toàn (quan trọng nhất)** | Cắm blower, chạy; bơm áp vào cổng U3 tăng dần | Blower tắt tại 25 ± 1.5 cmH2O, bật lại dưới ~23; SAFETY_TRIP lên ~3 V; **làm kể cả khi firmware treo / ESP32 bị giữ reset (EN=GND)** |
| 9. Tải nhiệt | Blower chạy max 30 phút trong vỏ kín | Đo nhiệt D1, Q1, F1, U8, U9 — D1 < 100 °C, F1 không ngắt |
| 10. Lưu lượng / OLED / encoder / còi | Chạy firmware chức năng | SFM3300 đọc được qua I2C 0x40, OLED 0x3C hiển thị, encoder đếm đúng chiều, còi kêu |

Ghi kết quả từng máy vào `Cpap/document/test_report_template.md` (chưa có — tạo cùng firmware).

Bước 8 là bằng chứng chính cho tính năng an toàn: nó phải đạt khi ESP32 không chạy, vì mạch cắt áp dùng phần cứng độc lập.

## 5. Việc cần làm trước khi đặt mạch
1. Khởi động lại cầu nối Altium → lưu board (giữ phần sửa pad header) → sửa xong silk J1 trong thư viện + sửa pad header trong thư viện → DRC lại.
2. Xuất lại Gerber + khoan + IPC-356 + CPL vào `Fab_RevC/` và đóng gói lại zip.
3. Relink footprint U6 trong schematic (cần tắt tạm "Symbol pins vs pads").
4. Đo dòng blower thật để chốt F1/D1 (mục 3).
