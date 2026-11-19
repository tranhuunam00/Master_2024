# SiPAP PCB Rev C — bảng chiều cao cho thiết kế vỏ

Board: 101.60 x 76.20 mm, dày 1.6 mm (FR-4 2 lớp, đồng 1 oz). Gốc (0,0) = góc dưới-trái nhìn từ mặt trên. Mọi linh kiện ở mặt trên; mặt dưới chỉ có chân xuyên lỗ (~1.5 mm).

Lỗ bắt vít M3 (Ø3.2 mm, pad Ø6 mm): (3.81, 3.81), (97.79, 3.81), (3.81, 72.39), (97.79, 72.39).

**Cao nhất: 26.5 mm (trục encoder U6, xuyên mặt vỏ). Không tính encoder: 24.8 mm (cầu chì PTC F1 MF-R400 đứng)**, rồi 13.21 mm (cảm biến áp), 11.5 mm (tụ C25) và 11.0 mm (jack DC).
F1 phải để đứng: không bẻ nằm được (phía +Y chỉ còn 7.6 mm tới mép board, phía −Y đè lên L2/Q4). Vỏ cần chừa ≥ 26 mm trên mặt PCB ở vùng X 15–30 mm, Y 66–71 mm.

File 3D: `SiPAP_PCB_RevC.step` (cùng thư mục). Rev B (`SiPAP_PCB_RevB.step`) đã cũ — không dùng.

| Ký hiệu | Linh kiện | X (mm) | Y (mm) | Đáy (mm) | Cao (mm) | Nguồn số liệu | Ghi chú cho vỏ |
|---|---|---|---|---|---|---|---|
| U6 | Encoder PEC11R-4220F-S0024 | 57.15 | 14.22 | 12.5 x 13.4 | 26.5 | Bourns PEC11R DS tr.2-3 | Lỗ Ø7 (ren M7) trên mặt vỏ, đúng tâm trục |
| F1 | PTC Bourns MF-R400 (đứng) | 22.86 | 68.58 | 14.4 x 3.0 | 24.8 | Bourns MF-R DS tr.3 (MF-R400: A 14.4, B 24.8, E 3.0 max) | Chi tiết cao nhất sau encoder |
| C25 | Tụ hoá Panasonic EEU-FC1E221 220 µF 25 V | 12.19 | 5.84 | Ø8.0 | 11.5 | Panasonic FC DS (D8 x L11.5) | Ngay cạnh J2; chừa thêm ~1 mm cho đầu tụ |
| U2 | Cảm biến áp MPXV5010GC6U (chính) | 90.42 | 46.99 | 10.8 x 10.8 | 13.21 | NXP MPX5010 DS Case 482A | Ống áp Ø3.17 — chừa chỗ cắm ống silicon |
| U3 | Cảm biến áp MPXV5010GC6U (dự phòng) | 90.42 | 29.21 | 10.8 x 10.8 | 13.21 | như trên | Ống áp Ø3.17 |
| J1 | Jack DC Same Sky PJ-102AH | 0 → 14.4 | 62.99 | 14.4 x 9.0 | 11.0 | PJ-102AH DS tr.2 | Miệng jack đúng mép trái board (X=0) — khoét lỗ Ø≥9 trên vách trái, tâm cao 6.5 mm từ mặt PCB |
| J2 | Header 2x2 blower | 7.62 | 11.43 | 5.1 x 5.1 | 8.5 | Header 2.54 chuẩn | +~9 mm cho đầu cái + dây |
| U4 | Header 1x6 SFM3300 | 82.55 | 10.16 | 15.2 x 2.54 | 8.5 | Header 2.54 chuẩn | Cáp ra cảm biến lưu lượng |
| U5 | Header 1x4 OLED | 80.01 | 69.85 | 10.2 x 2.54 | 8.5 | Header 2.54 chuẩn | Cáp ra OLED |
| J3 | Header 1x6 nạp UART | 96.52 | 54.61 | 2.54 x 15.2 | 8.5 | Header 2.54 chuẩn | Cần tiếp cận khi nạp firmware |
| BZ1 | Còi TDK PS1240P02BT | 43.69 | 7.11 | Ø12.2 | 6.5 | TDK PS series DS tr.2-3 | Lỗ thoát âm trên vỏ ngay trên còi |
| L1, L2 | Cuộn cảm Bourns SRN6045TA | 7.37 / 26.42 | 49.78 | 6.0 x 6.0 | 4.55 | Bourns SRN6045TA DS tr.1 (4.25 ±0.3) |  |
| U1 | ESP32-S3-WROOM-1 | 50.8 | 63.25 | 18 x 25.5 | 3.1 | Espressif DS v1.8 | Anten ở mép trên: không đặt kim loại/vít trong 15 mm quanh anten |
| Q4 | AO4407A (SO-8), chống ngược cực | 24.13 | 61.47 | 4.9 x 6.0 | 1.75 | AOS AO4407A DS | Thay D1 SS54 cũ |
| D2 | SMBJ15A (SMB) | 33.02 | 61.47 | 4.6 x 3.9 | 2.44 | Bourns SMBJ DS |  |
| Q1 | AO4407A (SO-8) | 17.78 | 18.29 | 4.9 x 6.0 | 1.75 | AOS AO4407A DS |  |
| U7 | LM393 (SO-8) | 36.83 | 22.86 | 4.9 x 6.0 | 1.75 | JEDEC MS-012 |  |
| Q2, Q3 | AO3401A / 2N7002 (SOT-23), công tắc nguồn SFM3300 | 85.09 / 77.72 | 6.10 | 2.9 x 2.4 | 1.12 | JEDEC TO-236 |  |
| U8, U9 | TPS54202 (SOT-23-6) | 10.16 / 29.21 | 43.18 | 1.6 x 2.9 | 1.1 | TI SLVSD26C DDC0006A |  |
| R/C | 0805 (50 con) |  |  | 2.0 x 1.25 | ≤1.45 | MLCC 22 µF 0805 dày nhất |  |

Tất cả số liệu lấy từ datasheet đã tải trong `Cpap/tools/datasheets/`; header 2.54 dùng kích thước chuẩn (2.5 mm nhựa + 6 mm chân).
