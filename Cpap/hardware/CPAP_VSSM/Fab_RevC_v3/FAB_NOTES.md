# SiPAP Rev C (layout v3) — hồ sơ đặt mạch

> Xuất ngày 2026-11-21. Đây là **bộ duy nhất dùng để đặt in**. Hai bộ `Fab_RevC/` và `Fab_RevC_v2/` là layout cũ, **không dùng nữa**.
>
> So với v2, bản v3 bố trí lại linh kiện và đi dây lại toàn bộ theo góp ý review. Schematic có 3 thay đổi, đều đã đồng bộ sang PCB, BOM và firmware:
> - đổi gán GPIO của ESP32;
> - thêm tụ C27 22 µF ở chân 3V3 của U1;
> - sửa mã encoder U6 trên PCB thành PEC11R-4225F cho khớp BOM.
>
> Chi tiết và số liệu kiểm tra: `REVIEW_v3.md`.

## Gửi xưởng
- **Chỉ in mạch trần:** gửi `SiPAP_RevC_v3_Gerber.zip`, gồm Gerber và file khoan.
- **Có lắp ráp (PCBA):** gửi thêm `BOM_RevC_v3.csv` và `SiPAP_RevC_v3_CPL.csv`.
- Nếu xưởng nhận ODB++ thì có thể gửi `SiPAP_RevC_v3_ODB++.zip` thay cho bộ Gerber. Nội dung hai bộ giống nhau.

Board đã qua kiểm giới hạn công nghệ **JLCPCB bản tiêu chuẩn, 2 lớp, đồng 1 oz**: đạt hết 7/7 mục. Mục đồng–mép board bridge không tự kiểm được nên tôi đo trực tiếp trên Gerber, đạt 0.647 mm ≥ 0.2 mm.

## Thông số board
| Mục | Giá trị |
|---|---|
| Số lớp | 2 (Top / Bottom) |
| Kích thước | 101.60 x 76.20 mm, 4 lỗ bắt vít M3 ở góc |
| Độ dày | 1.6 mm, FR-4 (Tg ≥ 130) |
| Đồng | 1 oz (35 µm) cả 2 mặt |
| Track nhỏ nhất / khe hở nhỏ nhất | 10 mil / 10 mil (0.254 / 0.254 mm) |
| Độ rộng dây | Tín hiệu 10 mil. 5V/3V3 20 mil, riêng đường cảm biến áp ra về điện trở FB 10 mil vì không mang dòng tải. SW 25–30 mil. 12V 50 mil, VIN_RAW 60 mil. Cực MOSFET Q1/Q4 nối bằng vùng đồng |
| Via | 131 via khoan 0.30 / pad 0.60 mm; 4 via khoan 0.25 / pad 0.50 mm ở khe EPAD của U1. Tổng 135 via (71 GND), **phủ mask kín toàn bộ** |
| Lỗ khoan | 170 lỗ tròn (0.254 → 3.2 mm) và 5 lỗ rãnh G85 (3 ở chân jack J1, 2 ở tai encoder U6). Tất cả có mạ |
| Mặt nạ hàn | Xanh lá, via phủ kín |
| Silkscreen | Trắng, chỉ mặt trên. Nét ≥ 0.15 mm, chữ cao 1.0 mm |
| Bề mặt | ENIG khuyên dùng (U1 có EPAD lưới 3x3); HASL không chì cũng được |
| Lỗ bắt vít | 4 x Ø3.2 mm có mạ, pad Ø6 mm, tại (3.81, 3.81), (97.79, 3.81), (3.81, 72.39), (97.79, 72.39) mm |

**Via và pad:** không có via nào nằm trong pad SMD. 4 via GND của EPAD U1 nằm ở **khe giữa** các ô pad, đúng như Hình 9-1 trang 41 datasheet ESP32-S3-WROOM-1 v1.8, nên không cần lấp/phủ (filled & capped).

## Danh sách file
| File | Nội dung | Định dạng |
|---|---|---|
| `SiPAP_RevC_v3_Gerber.zip` | Gửi xưởng. Gồm 9 lớp Gerber, 2 file khoan và bảng mũi khoan | — |
| `gerber/SiPAP_RevC_v3.GTL` / `.GBL` | Đồng mặt trên / dưới | Gerber RS-274X, mm, 4:4 |
| `gerber/….GTO` / `.GBO` | Silkscreen trên / dưới (mặt dưới trống) | RS-274X |
| `gerber/….GTS` / `.GBS` | Mặt nạ hàn trên / dưới | RS-274X |
| `gerber/….GTP` / `.GBP` | Kem hàn trên / dưới (mặt dưới trống) | RS-274X |
| `gerber/….GM` | Viền board 101.6 x 76.2 mm | RS-274X |
| `gerber/…-RoundHoles.TXT`, `…-SlotHoles.TXT`, `….DRR` | Lỗ tròn, lỗ rãnh, bảng mũi khoan | Excellon, mm, 4:3 |
| `SiPAP_RevC_v3.ipc` | Netlist kiểm tra điện, 403 điểm (268 pad + 135 via) | IPC-D-356 |
| `SiPAP_RevC_v3_ODB++.zip` | Bản ODB++ cùng nội dung | ODB++ |
| `SiPAP_RevC_v3_CPL.csv` | Toạ độ gắp-đặt, mm, gốc ở góc dưới-trái | CSV |
| `BOM_RevC_v3.csv` | 42 dòng, có MPN và mã thay thế | CSV |
| `SiPAP_RevC_v3.step` | Mô hình 3D (trục encoder đúng 31.5 mm) | STEP AP214 |
| `SiPAP_RevC_v3_Schematic.pdf` | Schematic 1 trang | PDF |
| `SiPAP_v3_TOP.png`, `SiPAP_v3_BOTTOM.png` | Ảnh 3D mặt trên / dưới (mặt dưới bị lật gương) | PNG |
| `SiPAP_v3_TOP_copper.png`, `SiPAP_v3_BOTTOM_copper.png` | Ảnh đồng dựng lại từ chính file Gerber, để review đường mạch | PNG |
| `REVIEW_v3.md` | Đối chiếu góp ý review với số đo, rà schematic/BOM có trích datasheet | — |

Gerber, file khoan và CPL dùng chung gốc toạ độ (góc dưới-trái board). Đã kiểm bằng cách đọc ngược file Gerber, xem mục "Trạng thái kiểm tra".

## Lắp ráp
- **SMD (xưởng gắn máy, 67 linh kiện):** U1, U2, U3, U7, U8, U9, Q1–Q4, D2, L1, L2 và 54 R/C 0805 (đã có C27 mới). Có 3 fiducial FID1–FID3 Ø1 mm.
- **THT (hàn tay, 9 linh kiện):** J1, J2, J3, U4, U5, U6, F1, BZ1, C25. Trong CPL ghi "THT".
- **Không lắp:** TP1–TP4 (pad test 12V / 5V / 3V3 / GND, có nhãn silk) và FID1–FID3.
- **Dấu cực tính trên silkscreen** (mới ở v3, các footprint cũ thiếu):
  - D2: vạch cathode bên phải, cạnh pad 1 = cathode = VIN12V.
  - C25 (tụ hoá): dấu "+" bên phải, cạnh pad + (BLOWER_V12).
  - BZ1: dấu "+" bên trái.
  - Chấm tròn chân 1 cho Q1, Q4, U7, U8, U9, U2, U3, U1 và các header J2, J3, U4, U5.
- **J3 (cổng nạp UART):** nằm dọc mép phải, chân 1 (GND) ở trên cùng. Tên từng chân in trên silk: GND, 3V3, TX, RX, EN, IO0.
- Góc xoay trong CPL theo quy ước Altium. Nên yêu cầu xưởng gửi ảnh DFM preview để xác nhận góc của SOT-23-6, SO-8 và MPXV trước khi gắn.

## Nếu xưởng thiếu linh kiện
BOM có cột **Alternate MPN** cho từng dòng chính. R/C 0805 dùng hãng nào cũng được, miễn đúng giá trị, sai số (1% cho R), điện áp và điện môi ghi trong BOM. Không thay TPS54202 bằng IC khác chân; chỉ TPS54202H tương thích. Cột mã LCSC mới có 3 dòng. Nếu đặt PCBA ở JLCPCB, xưởng sẽ tự khớp phần còn lại theo MPN; cần duyệt từng dòng trước khi thanh toán.

## Firmware
Gán GPIO v3 nằm trong `Cpap/firmware/sipap_v1/config.h` và đã biên dịch thử cho ESP32-S3: 678 634 byte flash (20%), 31 980 byte RAM (9%). **Không nạp firmware cũ lên board v3** vì chân đã đổi:

| Tín hiệu | GPIO | Tín hiệu | GPIO |
|---|---|---|---|
| PRESS_PRI | IO4 | ENC_A / ENC_B / ENC_SW | IO47 / IO21 / IO48 |
| PRESS_RED | IO5 | BLOWER_PWM / BLOWER_FG | IO10 / IO11 |
| SAFETY_TRIP | IO9 | BUZZER | IO12 |
| I2C SDA / SCL | IO2 / IO1 | SFM_EN | IO16 |

## Kiểm tra sau khi nhận board
1. Trước khi cấp nguồn, đo điện trở 12V–GND, 5V–GND và 3V3–GND (TP1–TP4); không được chập.
2. Cấp 12V qua J1, giới hạn dòng 0.5 A. Đo TP2 = 5.0–5.15 V, TP3 = 3.25–3.35 V.
3. Mạch an toàn: tăng áp U3 > ~25 cmH2O thì Q1 phải cắt nguồn blower (J2 chân 1 về 0 V).
4. Cấp ngược cực −12 V, giới hạn 0.3 A: dòng ≈ 0 (Q4 chặn).
5. Kéo IO16 xuống 0 thì U4 chân 2 (VDD của SFM3300) về 0 V; thả ra thì về 5 V (công tắc Q2/Q3).

## Trạng thái kiểm tra (2026-11-21)
| Kiểm tra | Kết quả |
|---|---|
| Altium DRC, 15 luật | **0 vi phạm**: clearance 10 mil, short, un-routed, width 3 nhóm, hole size, hole-to-hole 10 mil, mask sliver 4 mil, silk-to-mask 6 mil, silk-to-silk 6 mil, net antennae, height, modified polygon, routing topology |
| `pcb_layout_audit` của eda-agent | Đi dây 39/39 net, DRC 0, góc gắt hơn 45° = 0 (trên 197 điểm nối), chồng linh kiện 0, pad-entry mức nặng 0. Ngoại lệ ghi ở bảng dưới |
| Giới hạn JLCPCB tiêu chuẩn | Đạt 7/7 (track 10 ≥ 4 mil, lỗ 0.254 ≥ 0.15 mm, vành khuyên 5 ≥ 2 mil, khe hở 10 ≥ 4 mil, lỗ–lỗ 10 ≥ 7.9 mil, mask sliver 4 ≥ 3.9 mil) |
| Đọc ngược Gerber/khoan | Viền đúng 101.6 x 76.2 mm. **268/268 pad** có đồng và mask đúng lớp; kem hàn đủ, trừ TP và FID (đúng ý). **135/135 via** có đồng 2 mặt và phủ mask kín. **175/175 lỗ** khớp vị trí và cỡ, không thừa lỗ |
| Schematic ↔ PCB | 76/76 linh kiện, `in_sync`. 218 chân có net khớp từng chân. 29 chân để trống (net 1 chân) đều khớp |
| BOM ↔ board | 76 linh kiện, mỗi linh kiện đúng 1 dòng BOM, giá trị khớp comment |
| ERC | 0 lỗi. 7 cảnh báo "no driving source" trên net cầu chia/cổng (COMP_REF, EN, FB3, FB5, GATE, RPP_G, SFM_G), bình thường |
| Dây cáp (`check_harness.py`) | 19 đạt, 0 lỗi, 1 chờ dữ liệu blower |

**Trạng thái DRC:** bridge ghi verdict là `pass_unapproved_rules`, tức DRC đạt nhưng bộ luật chưa được bạn duyệt. Phiên này đã sửa 4 luật:
- silk-to-mask và silk-to-silk đặt 6 mil, theo mức tối thiểu 0.15 mm của JLC;
- tách luật độ rộng thành 3 nhóm: tín hiệu, nguồn 5V/3V3/GND/SW, 12V;
- nới chiều cao tối đa từ 1200 lên 1250 mil, cho trục encoder 31.5 mm.

Duyệt trong Altium bằng `pcb_drc_rules(action='approve')` nếu đồng ý.

**Ngoại lệ đã biết, có chủ đích:**
| Mục | Vì sao chấp nhận |
|---|---|
| J1 chạm keep-out mép board; U1 nằm trên keep-out anten | Miệng jack phải ra đúng mép trái. Anten module phải nằm trên vùng cấm đồng ở mép board (ESP32-S3-WROOM-1 DS v1.8, mục layout anten) |
| 24/64 via tín hiệu không có via GND trong 40 mil | Đều là tín hiệu chậm hoặc DC: EN, ADC, buzzer, gate MOSFET, nguồn 3V3/5V, SW của buck. Board không có tín hiệu tốc độ cao |
| 10 mảng GND rời ở mặt trên | Cả 10 đều nối xuống mặt phẳng GND liền khối ở mặt dưới qua via; 0 mảng chết |
| Silk viền module U1 cách mép trên 0.10 mm | Thân module đặt sát mép theo yêu cầu anten. Xưởng có thể cắt bớt nét silk này, không ảnh hưởng gì |
| 1 đoạn EN ở mặt dưới chạm mép thân R12 (9.9 mil) | Nằm ở mặt dưới, không nằm giữa 2 pad. Mặt trên: 0 đoạn dây lạ dưới thân linh kiện |

File OutJob để xuất lại: `CPAP_VSSM/SiPAP_Fab.OutJob` (Gerber, khoan, IPC-356, ODB++) và `SiPAP_Docs.OutJob` (PDF schematic). Script gom hồ sơ: `tools/layout_v3/fab_v3.py`.
