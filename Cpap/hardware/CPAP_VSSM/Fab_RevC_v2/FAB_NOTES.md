> **ĐÃ CŨ — KHÔNG DÙNG ĐỂ ĐẶT IN (2026-11-21).** Thay bằng `../Fab_RevC_v3/` (layout v3: sửa 22 lỗi clearance, 84 góc gắt, dây dưới thân linh kiện; thêm C27; đổi gán GPIO — firmware v3 không chạy đúng trên board v2 và ngược lại).

# SiPAP Rev C (layout v2) — hồ sơ đặt mạch

> Xuất 2026-11-21 sau khi **đặt lại linh kiện và đi dây lại toàn bộ** theo góp ý review (dây gọn, độ rộng đồng nhất, không đi dây dưới thân linh kiện, khối giống nhau đi dây giống nhau). Schematic và danh sách linh kiện **không đổi** so với Rev C; chỉ đổi vị trí/đi dây. Bộ cũ trong `Fab_RevC/` là layout cũ — **không dùng nữa**.

Gửi xưởng file `SiPAP_RevC_v2_Gerber.zip` (Gerber + khoan). Đặt lắp ráp (PCBA) thì gửi thêm `BOM_RevC_v2.csv` và `SiPAP_RevC_v2_CPL.csv`.

## Thông số board
| Mục | Giá trị |
|---|---|
| Số lớp | 2 (Top / Bottom) |
| Kích thước | 101.60 x 76.20 mm |
| Độ dày | 1.6 mm, FR-4 (Tg ≥ 130) |
| Đồng | 1 oz (35 µm) cả 2 mặt |
| Track nhỏ nhất / khe hở nhỏ nhất | 10 mil / 10 mil (0.254 / 0.254 mm) |
| Track nguồn | 5V/3V3/GND/SW 20 mil; 12V/blower 50 mil (thắt 24 mil vào pad SO-8) |
| Via | khoan 0.30 mm / pad 0.60 mm (94 via); 4 via khoan 0.25 mm / pad 0.50 mm ở khe giữa các ô pad EPAD của U1 |
| Lỗ nhỏ nhất | 0.25 mm |
| Mặt nạ hàn | Xanh lá, **phủ kín (tented) tất cả via** |
| Silkscreen | Trắng, mặt trên |
| Bề mặt | HASL không chì hoặc ENIG (khuyên ENIG vì U1 có EPAD lưới 3x3) |
| Lỗ bắt vít | 4 x Ø3.2 mm có mạ, pad Ø6 mm |

**Via và pad:** không có via nào nằm trong pad SMD (DRC rule "Vias Under SMD = Not Allowed"). 4 via GND của EPAD U1 nằm ở **khe giữa** các ô pad như datasheet ESP32-S3-WROOM-1 v1.8 trang 45 vẽ, nên **không cần lấp/phủ (filled & capped)** — chỉ cần phủ mặt nạ như các via khác.

## Danh sách file
| File | Nội dung | Chuẩn |
|---|---|---|
| SiPAP_PCB.GTL / .GBL | Đồng mặt trên / dưới | Gerber RS-274X, mm, 4:4 |
| SiPAP_PCB.GTO / .GBO | Silkscreen trên / dưới | RS-274X |
| SiPAP_PCB.GTS / .GBS | Mặt nạ hàn trên / dưới | RS-274X |
| SiPAP_PCB.GTP / .GBP | Kem hàn (stencil) trên / dưới | RS-274X |
| SiPAP_PCB.GM | Viền board | RS-274X |
| SiPAP_PCB-RoundHoles.TXT | Lỗ tròn có mạ (133 lỗ) | Excellon, mm, 4:3, bỏ số 0 cuối |
| SiPAP_PCB-SlotHoles.TXT | Lỗ rãnh có mạ (chân jack J1, tai encoder U6; 5 rãnh) | Excellon G85 |
| SiPAP_PCB.DRR | Bảng mũi khoan | Báo cáo |
| SiPAP_RevC_v2.ipc | Netlist kiểm tra điện | IPC-D-356A |
| SiPAP_RevC_v2_CPL.csv | Toạ độ gắp-đặt (mm, gốc góc dưới-trái) | CSV chung |
| BOM_RevC_v2.csv | Danh sách linh kiện + MPN + mã thay thế (giống Rev C) | CSV |
| SiPAP_RevC_v2.step | Mô hình 3D board | STEP |
| SiPAP_v2_TOP.png / SiPAP_v2_BOTTOM.png | Ảnh 3D mặt trên / dưới (mặt dưới nhìn từ dưới lên nên bị lật gương) | — |

Toạ độ Gerber, khoan và CPL cùng gốc (góc dưới-trái board).

## Lắp ráp
- **SMD (xưởng gắn máy, 66 linh kiện):** U1, U2, U3, U7, U8, U9, Q1, Q2, Q3, Q4, D2, L1, L2 và toàn bộ R/C 0805. Có 3 fiducial (FID1–FID3) Ø1 mm.
- **THT (hàn tay, 9 linh kiện):** J1, J2, J3, U4, U5, U6, F1, BZ1, C25 — đánh dấu "THT" trong CPL.
- **Không lắp:** TP1–TP4 (pad test 12V / 5V / 3V3 / GND), FID1–3.
- **Cực tính — đổi so với layout cũ:**
  - D2 xoay 180°: vạch (cathode, pad 1 = VIN12V) nằm **bên phải**.
  - C25 xoay 180°: chân + (pad vuông, BLOWER_V12) **bên phải**, vạch '−' (GND) **bên trái**.
  - U2/U3 (MPXV5010) xoay 180°: chân 1 theo chấm silkscreen; cổng áp suất nằm giữa thân nên vị trí cổng không đổi so với vỏ v0.3.
  - U1/U7/U8/U9/Q1/Q4: chấm pin 1 theo silkscreen.
- Góc xoay trong CPL theo quy ước Altium. Yêu cầu xưởng gửi ảnh DFM preview để xác nhận góc SOT-23-6 / SO-8 / MPXV trước khi gắn.

## Nếu xưởng thiếu linh kiện
BOM có cột **Alternate MPN** cho từng dòng chính. R/C 0805 dùng bất kỳ hãng nào đúng giá trị, sai số (1% cho R), điện áp và điện môi ghi trong BOM. Không thay TPS54202 bằng IC khác chân (chỉ TPS54202H tương thích).

## Kiểm tra sau khi nhận board
1. Đo trở kháng 12V–GND, 5V–GND, 3V3–GND (không được chập) trước khi cấp nguồn.
2. Cấp 12V qua J1, giới hạn dòng 0.5 A; đo TP2 = 5.0–5.15 V, TP3 = 3.25–3.35 V.
3. Mạch an toàn: tăng áp U3 > ~25 cmH2O → Q1 cắt nguồn blower (J2 pin 1 về 0 V).
4. Cấp ngược cực −12 V (giới hạn 0.3 A): dòng ≈ 0 (Q4 chặn).
5. Kéo IO16 xuống 0 → U4 pin 2 (VDD SFM3300) về 0 V; thả ra → 5 V (công tắc Q2/Q3).

## Trạng thái kiểm tra (Altium DRC 2026-11-21)
Clearance, short-circuit, un-routed, width, hole size, hole-to-hole, mask sliver, net antennae: **0 lỗi**. Còn lỗi silkscreen (nét silk của footprint chạm mask pad) — xưởng tự cắt silk trên pad, không ảnh hưởng điện.
