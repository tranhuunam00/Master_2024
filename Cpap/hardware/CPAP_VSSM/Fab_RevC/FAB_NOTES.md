# SiPAP Rev C — hồ sơ đặt mạch (fabrication + assembly)

> **CHƯA GỬI XƯỞNG BỘ FILE NÀY.** Kiểm tra lại (2026-11-19) phát hiện: header J2/J3/U4/U5 thiếu đồng mặt dưới và silk J1 lệch ra ngoài board. Phải xuất lại sau khi sửa — xem `Cpap/document/verification_RevC.md`.

Gửi xưởng file `SiPAP_RevC_Gerber.zip` (Gerber + khoan). Đặt lắp ráp (PCBA) thì gửi thêm `BOM_RevC.csv` và `SiPAP_RevC_CPL.csv`.

## Thông số board
| Mục | Giá trị |
|---|---|
| Số lớp | 2 (Top / Bottom) |
| Kích thước | 101.60 x 76.20 mm |
| Độ dày | 1.6 mm, FR-4 (Tg ≥ 130) |
| Đồng | 1 oz (35 µm) cả 2 mặt |
| Track/space nhỏ nhất | 8 / 10 mil (0.20 / 0.25 mm) |
| Via | khoan 0.30 mm (12 mil) / pad 0.60 mm (24 mil); 4 via trong pad EPAD của U1 khoan 0.25 mm |
| Lỗ nhỏ nhất | 0.25 mm |
| Mặt nạ hàn | Xanh lá (hoặc tuỳ chọn), **phủ kín (tented) tất cả via** |
| Silkscreen | Trắng, mặt trên |
| Bề mặt | HASL không chì hoặc ENIG (khuyên ENIG vì U1 có EPAD lưới 3x3) |
| Lỗ bắt vít | 4 x Ø3.2 mm có mạ, pad Ø6 mm |

**Yêu cầu riêng:** 4 via nằm trong pad EPAD của U1 (ESP32, toạ độ 48.6/50.0 mm, 59.4/62.2 mm) — yêu cầu **lấp và phủ (filled & capped)** hoặc ít nhất phủ mặt nạ từ mặt dưới, để thiếc không chảy qua lỗ.

## Danh sách file
| File | Nội dung | Chuẩn |
|---|---|---|
| SiPAP_PCB.GTL / .GBL | Đồng mặt trên / dưới | Gerber RS-274X, mm, 4:4 |
| SiPAP_PCB.GTO / .GBO | Silkscreen trên / dưới | RS-274X |
| SiPAP_PCB.GTS / .GBS | Mặt nạ hàn trên / dưới | RS-274X |
| SiPAP_PCB.GTP / .GBP | Kem hàn (stencil) trên / dưới | RS-274X |
| SiPAP_PCB.GM | Viền board (board outline) | RS-274X |
| SiPAP_PCB-RoundHoles-Plated.TXT | Lỗ tròn có mạ | Excellon, mm, 4:3, bỏ số 0 cuối |
| SiPAP_PCB-SlotHoles-Plated.TXT | Lỗ rãnh có mạ (chân jack J1, tai encoder U6) | Excellon G85 |
| SiPAP_PCB.DRR | Bảng mũi khoan | Báo cáo |
| SiPAP_RevC.ipc | Netlist kiểm tra điện | IPC-D-356A (inch 0.0001) |
| SiPAP_RevC_CPL.csv | Toạ độ gắp-đặt (mm, gốc góc dưới-trái) | CSV chung |
| BOM_RevC.csv | Danh sách linh kiện + MPN + mã thay thế | CSV |
| SiPAP_RevC_top.png | Ảnh mặt trên (tham khảo lắp ráp) | — |
| SiPAP_RevC_schematic.png | Ảnh schematic (tham khảo) | — |

Toạ độ Gerber và khoan cùng gốc (relative origin = góc dưới-trái board).

## Lắp ráp
- **SMD (xưởng gắn máy):** U1, U2, U3, U7, U8, U9, Q1, D1, D2, L1, L2, toàn bộ R/C 0805. Có 3 fiducial (FID1–FID3) Ø1 mm.
- **THT (hàn tay hoặc xưởng hàn tay):** J1, J2, J3, U4, U5, U6, F1, BZ1 — đánh dấu "THT" trong CPL.
- **Không lắp:** TP1–TP4 (pad test 12V / 5V / 3V3 / GND), FID1–3.
- Cực tính: D1 vạch (cathode) về phía pad 2 (VIN12V, bên phải); D2 vạch về pad 1 (VIN12V, bên trái); U1/U7/U8/U9/Q1 chấm pin 1 theo silkscreen.
- Góc xoay trong CPL theo quy ước Altium (0° = như footprint trong thư viện). Mỗi xưởng có thể cần chỉnh offset góc cho SOT-23-6 / SO-8 — yêu cầu xưởng gửi ảnh xác nhận (DFM preview) trước khi gắn.

## Nếu xưởng thiếu linh kiện
BOM có cột **Alternate MPN** cho từng dòng chính. Điện trở/tụ 0805 dùng bất kỳ hãng nào đúng giá trị, sai số (1% cho R), điện áp và điện môi (X5R/X7R/C0G) ghi trong BOM. Không thay TPS54202 bằng IC khác chân (chỉ TPS54202H tương thích).

## Kiểm tra sau khi nhận board
1. Đo trở kháng 12V–GND, 5V–GND, 3V3–GND (không được chập) trước khi cấp nguồn.
2. Cấp 12V qua J1 có giới hạn dòng 0.5 A, đo TP2 = 5.0–5.15 V, TP3 = 3.25–3.35 V.
3. Kiểm tra mạch an toàn: tăng áp U3 > ~25 cmH2O → Q1 cắt nguồn blower (J2 pin 1 về 0 V).
