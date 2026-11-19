# SiPAP PCB Rev B — bảng chiều cao cho thiết kế vỏ

Board: 101.60 x 76.20 mm, dày 1.6 mm (FR-4 2 lớp). Gốc toạ độ (0,0) = góc dưới-trái board nhìn từ mặt trên. Mọi linh kiện ở mặt trên, mặt dưới phẳng (chỉ có chân xuyên lỗ ~1.5 mm).

Lỗ bắt vít M3 (Ø3.2 mm, pad Ø6 mm): (3.81, 3.81), (97.79, 3.81), (3.81, 72.39), (97.79, 72.39).

**Chiều cao lớn nhất trên mặt board: 26.5 mm (trục encoder U6)**. Nếu không tính trục encoder (trục xuyên qua mặt vỏ): 22 mm (module buck, giả định) — hoặc 13.21 mm (cảm biến áp) nếu module buck lắp nằm/thấp hơn.

File 3D: `SiPAP_PCB_RevB.step` (cùng thư mục) — import vào CAD để làm vỏ.

| Ký hiệu | Linh kiện | X (mm) | Y (mm) | Kích thước đáy (mm) | Cao từ mặt PCB (mm) | Nguồn số liệu | Ghi chú cho vỏ |
|---|---|---|---|---|---|---|---|
| U6 | Encoder PEC11R-4220F-S0024 (trục đứng) | 57.15 | 14.22 | 12.5 x 13.4 (thân) | 26.5 | Bourns PEC11R datasheet tr.2-3: thân 6.5 + cổ M7 7 + trục L=20 | Tâm trục — cần lỗ Ø7 (ren M7) trên mặt vỏ |
| U2 | Cảm biến áp MPXV5010GC6U (chính) | 90.42 | 46.99 | 10.8 x 10.8 (thân), chân 18.4 | 13.21 | MPX5010 datasheet tr.7 Case 482A: C max 13.21 | Ống áp Ø3.17 dài 6.48 tại tâm — chừa chỗ cắm ống silicon |
| U3 | Cảm biến áp MPXV5010GC6U (dự phòng) | 90.42 | 29.21 | 10.8 x 10.8 (thân), chân 18.4 | 13.21 | như trên | Ống áp Ø3.17 tại tâm |
| U8 | Module buck 12V->5V (header 4 chân) | 25.4 | 59.69 | header 2.54 x 10.2 | 22.0 | GIẢ ĐỊNH: module lắp đứng ~5 x 22 x 22 mm | Chưa chốt module — đo lại khi mua |
| U9 | Module buck 5V->3V3 (header 4 chân) | 33.02 | 59.69 | header 2.54 x 10.2 | 22.0 | GIẢ ĐỊNH: như U8 | Chưa chốt module |
| J1 | Jack DC 5.5x2.1 | 7.62 | 57.15 | 9 x 14 | 11.0 | GIẢ ĐỊNH jack PCB thông dụng | Đầu cắm hướng ra mép trái board — khoét lỗ vỏ |
| BZ1 | Còi 12 mm | 46.99 | 7.62 | Ø12 | 9.5 | GIẢ ĐỊNH còi 12 mm bước 7.6 | Nên có lỗ thoát âm trên vỏ |
| J2 | Header 2x2 blower | 7.62 | 11.43 | 5.1 x 5.1 | 8.5 | Header 2.54 tiêu chuẩn (2.5 nhựa + 6 chân) | Cộng thêm ~9 mm cho đầu cái + dây |
| U4 | Header 1x6 SFM3300 | 82.55 | 10.16 | 2.54 x 15.2 | 8.5 | Header 2.54 tiêu chuẩn | Cáp ra cảm biến lưu lượng |
| U5 | Header 1x4 OLED | 80.01 | 69.85 | 2.54 x 10.2 | 8.5 | Header 2.54 tiêu chuẩn | Cáp ra OLED gắn mặt vỏ |
| J3 | Header 1x6 nạp UART | 96.52 | 54.61 | 2.54 x 15.2 | 8.5 | Header 2.54 tiêu chuẩn | Cần tiếp cận được khi nạp firmware |
| U1 | ESP32-S3-WROOM-1 | 50.8 | 63.25 | 18 x 25.5 | 3.1 | Espressif DS v1.8 Fig 10-1 tr.42: 3.1±0.15 | Anten ở mép trên board: không đặt kim loại/vít trong vùng 15 mm quanh anten |
| D1 | Diode SS54 SMC | 16.51 | 62.23 | 7 x 6 | 2.62 | DO-214AB (giả định theo chuẩn) |  |
| U7 | LM393 SOIC-8 | 36.83 | 22.86 | 4.9 x 6.0 | 1.75 | JEDEC MS-012 |  |
| Q1 | AO3401A SOT-23 | 17.78 | 17.78 | 2.9 x 2.4 | 1.12 | JEDEC TO-236 |  |
| R/C | Điện trở/tụ 0805 (30 con) |  |  | 2.0 x 1.25 | 1.25 | EIA 0805, MLCC dày nhất |  |

Dòng ghi **GIẢ ĐỊNH** là linh kiện chưa chốt mã cụ thể — đo lại khi có hàng thật rồi cập nhật.
