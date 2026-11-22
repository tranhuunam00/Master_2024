# SiPAP Rev C layout v3 — đối chiếu góp ý review và rà soát thiết kế

Ngày 2026-11-21. Mọi số liệu dưới đây đo trực tiếp trên `SiPAP_PCB.PcbDoc` qua Altium (bridge eda-agent v2026.11.10.02, Altium 26.11.1.5) hoặc đọc ngược từ chính file Gerber gửi xưởng. Số của v2 đo lại trên board v2 trước khi làm lại.

## 1. Góp ý review → số đo (v2 → v3)

| Góp ý | Cách đo | v2 | v3 | Cách làm ở v3 |
|---|---|---|---|---|
| Đường nét xấu, chồng chéo, khó check | `corner_audit`: điểm nối gắt hơn 45° | 84 / 460 (30 góc vuông, 10 góc nhọn) | **0 / 197** | Chỉ đi góc 45°. Mỗi bus đi thành một nhóm song song: 3 dây TX/RX/BOOT chạy thẳng hàng với chân J3 (bước 100 mil), cặp I2C song song. Tổng số đoạn dây giảm từ 641 xuống 378 |
| Kích thước dây không đồng nhất | Số cỡ rộng dây, riêng net 12V | 6 cỡ; net 12V dùng 5 cỡ (20/24/30/40/50 mil) | Luật theo nhóm net, xem cột bên | Tín hiệu luôn 10 mil (233 đoạn, 29 net). 5V/3V3 20 mil, trừ dây cảm biến áp ra về FB 10 mil (không mang dòng). 12V 50 mil, VIN_RAW 60 mil, SW 25–30 mil. Luật DRC tách 3 nhóm width |
| Dây logic to đùng | Độ rộng các net logic | 10 mil (v1 mới bị lỗi này) | **10 mil** | Như trên |
| Linh kiện công suất mà dây bé tí | Cách nối chân MOSFET và buck | Chân SO-8 của Q1/Q4 nối bằng cổ dây 24 mil | **Vùng đồng** phủ cả hàng chân S và D của Q1, Q4 | Mỗi buck là một ô copy theo mẫu layout của TI (SLVSD26C, Hình 7-16 "Board Layout", trang 21). Vùng đồng VIN/GND/SW/VOUT, vòng nóng VIN–GND ngắn nhất, FB lấy áp kiểu Kelvin |
| Dây đi dưới thân linh kiện, nhất là linh kiện nóng | Số đoạn dây của net khác nằm trong thân 3D linh kiện | 29 đoạn dưới 20 linh kiện, cộng 19 dây chui dưới ESP32 (dài tới 595 mil) | **Mặt trên: 0.** Mặt dưới: 1 đoạn EN dài 9.9 mil chạm mép thân R12 | Router chặn cứng vùng thân linh kiện, chỉ cho chân của chính linh kiện đó thoát ra. Mặt dưới gần như toàn bộ là mặt phẳng GND |
| Linh kiện giống nhau đi dây giống nhau | So vị trí và đồng tương đối | Hai buck mỗi cái một kiểu | **Giống hệt** | U8/U9: cùng ô linh kiện + đồng, dời 620 mil. U2/U3: cùng khối tụ/điện trở, dời 700 mil. Pull-up encoder R7/R8/R12/R14 xếp một hàng. Q1 và Q4 cùng kiểu vùng đồng |
| Khó check trên board thật | Silkscreen | Nhiều tên lộn ngược; thiếu dấu cực tính | Chữ đọc ở 0°/90°. Thêm 17 dấu cực tính / chân 1. Nhãn TP 12V/5V/3V3/GND, nhãn từng chân J3, chữ "12V IN", "BLOWER" | Dấu cực tính phát hiện thiếu khi đọc ngược Gerber silkscreen; xem mục 3 |

Thêm theo các cổng kiểm tra của eda-agent:

| Cổng | v2 | v3 |
|---|---|---|
| DRC trong audit (clearance) | **22 lỗi** (pad SO-8 cách GND 2.1 mil sau khi sửa footprint SOIC-8; BLOWER_PWM cách pad Q1 4.9 mil) | 0 |
| Via tín hiệu có via GND trong 40 mil | 0 / 76 | 40 / 64. 24 via còn lại là tín hiệu chậm/DC, xem mục 4 |
| Đảo đồng GND | 2 mảng | 10 mảng ở mặt trên do vùng đồng nguồn chia cắt; **cả 10 đều nối xuống mặt GND dưới qua via, 0 mảng chết**. Mặt dưới liền một khối |
| Pad-entry mức nặng / chen chúc | 0 / 5 | 0 / 1 |
| Đặt linh kiện theo vỏ hộp v0.3 | F1 bị dời khỏi hốc nắp dành cho PTC cao 24.8 mm | F1 trả về đúng hốc. U1, U2, U3, U4, U5, U6, BZ1, J1, J2 giữ đúng vị trí lỗ vỏ |

## 2. Rà schematic và BOM (trích datasheet)

Datasheet gốc lưu ở `Cpap/tools/datasheets/`.

1. **Thiếu tụ 22 µF ở chân 3V3 của ESP32.** Hình 9-1 trang 41 datasheet ESP32-S3-WROOM-1 v1.8 đặt C1 22 µF + C8 0.1 µF trên VDD33. Board cũ chỉ có 0.1 µF (C3).
   → Thêm **C27 22 µF 16 V** (Murata GRM21BR61C226ME44, cùng dòng BOM với C1/C2/C18/C21, nay số lượng 5). Đặt cạnh C3 tại chân 2 của U1.
2. **Mạch RC ở chân EN đã đúng.** Datasheet v1.8 trang 41 khuyên R = 10 kΩ, C = 1 µF; board có R9 10k + C5 1 µF. Giữ nguyên.
3. **Gán GPIO làm dây phải vòng quanh module, và dùng chân strapping.** Trước đây BLOWER_PWM nằm trên IO3, là chân strapping (Bảng 4-1 trang 13). Đổi gán để mỗi tín hiệu ra đúng cạnh module hướng về linh kiện của nó (Bảng 3-1 trang 11):
   - SAFETY_TRIP → IO9 (chân 17)
   - BLOWER_PWM → IO10 (chân 18)
   - BLOWER_FG → IO11 (chân 19)
   - BUZZER → IO12 (chân 20)
   - ENC_B / ENC_A / ENC_SW → IO21 / IO47 / IO48 (chân 23–25)
   - I2C SDA / SCL → IO2 / IO1 (chân 38 / 39)

   Giữ nguyên: PRESS_PRI = IO4 (ADC1_CH3), PRESS_RED = IO5 (ADC1_CH4), SFM_EN = IO16, EN, IO0, TXD0/RXD0.

   Không chân strapping nào (IO0 / IO3 / IO45 / IO46) còn lái ngoại vi. IO35–IO37 để trống vì trên module có PSRAM octal (R8) chúng nối vào PSRAM (chú thích b, Bảng 3-1 trang 12).
4. **IO47/IO48 chạy 3.3 V trên bản N8R8.** Chú thích ở trang 41 và trang 12 nói chỉ bản N16R16V mới chạy IO47/IO48 ở mức 1.8 V. Board dùng N8R8, nên encoder trên IO47/IO48 dùng pull-up 3V3 là đúng.
5. **Pull-up I2C lên 3V3 đủ cho SFM3300**, không cần mạch chuyển mức. Datasheet SFM3300 v1.4 trang 3 cho mức High tối thiểu 2.5 V, Low tối đa 1.1 V, nguồn 5 V ±5 %.
6. **EN của TPS54202 để hở là đúng.** TI SLVSD26C, bảng chân trang 3 ghi "Float the EN pin to enable"; trang 9 nói chân EN có nguồn dòng kéo lên nội. Đó là lý do ERC thấy chân EN của U8/U9 là net 1 chân.
7. **Mã và chiều cao encoder U6.** PCB ghi PEC11R-4220F trong khi schematic và BOM ghi 4225F; đã sửa PCB thành 4225F.
   - Datasheet Bourns PEC11R trang 2–3: thân cao 6.5 mm tới mặt lắp, bản L = 25 mm có LB = 7 mm. Vậy trục cao **31.5 mm** so với mặt PCB.
   - Model 3D cũ vẽ trục 20 mm (cao 26.5 mm), nên STEP v2 thấp hơn hàng thật 5 mm. Đã sửa model và nới luật chiều cao từ 1200 lên 1250 mil.
8. **Footprint thiếu dấu cực tính và chân 1.** Phát hiện khi đọc ngược file Gerber silkscreen:
   - D2 không có vạch cathode;
   - C25 (tụ hoá) và BZ1 không có dấu "+";
   - các SO-8, SOT-23-6, MPXV và header không có chấm chân 1.

   Đã thêm 17 dấu trên silkscreen của board. Footprint trong thư viện chưa sửa; nên sửa trước khi dùng lại cho board khác.
9. **J3 (cổng nạp):** giữ nguyên thứ tự chân GND, 3V3, TXD, RXD, EN, IO0. Header đặt dọc mép phải, chân 1 ở trên, có in tên từng chân. Bảng dây trong `Cpap/hardware/wiring_harness.md` đã cập nhật; `check_harness.py` cho 19 đạt, 0 lỗi.
10. **BOM ↔ schematic ↔ PCB:** 76 linh kiện, mỗi linh kiện đúng 1 dòng BOM. Comment trên PCB khớp schematic sau khi sửa U6. Netlist khớp từng chân (218 chân có net, 29 chân để trống).

## 3. Kiểm tra PCB v3 trước khi gửi xưởng

| Kiểm tra | Kết quả |
|---|---|
| Altium DRC (15 luật, batch) | 0 vi phạm. Verdict của bridge: `pass_unapproved_rules`, tức bộ luật chưa được bạn duyệt; xem FAB_NOTES |
| `pcb_layout_audit` | Đi dây 39/39 net. DRC 0, góc gắt 0, chồng linh kiện 0, pad-entry nặng 0 |
| Giới hạn JLCPCB tiêu chuẩn | 7/7 đạt; đồng cách mép board 0.647 mm (đo trên Gerber) |
| Đọc ngược Gerber/khoan bằng script riêng (`gerber_check.py`) | 268/268 pad đúng lớp đồng/mask/kem. 135/135 via phủ mask kín. 175/175 lỗ khớp vị trí và cỡ. Viền 101.6 x 76.2 mm |
| Silkscreen đọc ngược từ Gerber | Chữ đọc được, không đè pad (DRC silk 6 mil = 0). 17 dấu cực tính đã có trong file GTO |
| ERC | 0 lỗi, 7 cảnh báo "no driving source" trên net cầu chia (bình thường) |
| Firmware | `config.h` theo gán chân v3, biên dịch đạt (ESP32-S3, OPI PSRAM, 8 MB) |

## 4. Còn mở — cần bạn quyết

1. **Duyệt bộ luật DRC.** DRC 0 lỗi nhưng bộ luật đã sửa trong phiên này (silk 6 mil, tách 3 nhóm width, chiều cao 1250 mil) chưa có người duyệt.
2. **In mạch trần hay đặt PCBA?** Nếu PCBA ở JLC: BOM mới có mã LCSC cho 3 dòng, các dòng khác JLC khớp theo MPN và cần duyệt tay.
3. **24 via tín hiệu chưa có via GND trong 40 mil:** EN, ADC, buzzer, gate MOSFET, nguồn 3V3/5V, SW. Đây là tín hiệu chậm hoặc DC nên tôi chấp nhận. Muốn đạt tuyệt đối cổng `return_via_audit` thì phải thêm via GND cạnh từng via.
4. **Hai lỗi nhỏ chưa sửa:**
   - Một đoạn EN ở mặt dưới chạm mép thân R12 (9.9 mil).
   - Schematic: ký hiệu GND đè chữ PRESS_PRI_ADC cạnh chân 3–4 của U1. Chỉ là thẩm mỹ, netlist đúng.
5. **Chưa thử trên phần cứng:** firmware mới chỉ biên dịch. Mức FG của blower vẫn chờ đo (mục OPEN trong `check_harness.py`). Quy trình test khi nhận board nằm trong FAB_NOTES.

## 5. Review đối kháng 2026-11-22 và các sửa đổi

Một lượt review chỉ đọc, đối chiếu từng con số với datasheet gốc (ảnh các trang được trích, SHA-256 và kết quả kiểm nằm trong `review_evidence/`), tìm ra:

| # | Mức | Vấn đề | Căn cứ | Đã sửa |
|---|---|---|---|---|
| M1 | Lớn | Ngưỡng cắt áp phần cứng không chỉnh được, thực tế ~20–30 cmH2O chứ không phải ~25 | MPX5010 DS tr.2: Voff 0–0.425 V, VFSS 4.275–4.725 V | R10 36k + RV1 Bourns 3224W-1-503E 50k phía 5V; chỉnh lúc test bằng self-test mục 9 |
| M2 | Lớn | Firmware cho xoá AL_ZERO mà không hiệu chuẩn lại; BLE không xác thực; lệnh `PWM:n` chạy hở vòng đến 100 % | `sipap_v1.ino` dòng 137, 253–257, 312; `bleBegin` | AL_ZERO chỉ xoá khi hiệu chuẩn lại đạt; ghép đôi BLE bằng passkey + MITM + bonding; `PWM:n` chỉ có trong bản test bàn |
| m3 | Nhỏ | Chân VSR của blower lơ lửng khi ESP32 reset hoặc đang nạp firmware | Netlist: BLOWER_VSR/BLOWER_PWM không có điện trở kéo; bản vẽ Wonsmart chỉ ghi VSR 0–3.3 V | Thêm R30 100k xuống GND |
| m4 | Nhỏ | Datasheet lưu trong repo không phải của linh kiện JLC gắn; 2 file hỏng | Trang lcsc.com của C8545, C7955, C1973187 | Tải datasheet JSCJ, onsemi, Vishay; xoá `MPX5010.pdf` và `EC11.pdf` |
| m5 | Nhỏ | BOM ghi J1 "2.1 mm" trong khi chân giữa Ø2.0 | PJ-102AH DS tr.2 | Sửa mô tả trong BOM và `wiring_harness.md` |

Có một lỗi công cụ được ghi lại: `pdftotext -layout` làm lệch một hàng trong bảng SRN6045TA và SMBJ, nên một phát hiện nháp về cuộn cảm đã bị bỏ sau khi đối chiếu ảnh trang gốc. BOM ghi đúng.

Chi tiết layout và kết quả kiểm tra sau khi sửa (ERC 0, DRC 0, 40/40 net, Gerber đọc ngược 273 pad / 139 via / 179 lỗ, XOR với bản cũ chỉ khác quanh RV1, R10, R30): `FAB_NOTES.md`, mục "Sửa sau review đối kháng 2026-11-22".
