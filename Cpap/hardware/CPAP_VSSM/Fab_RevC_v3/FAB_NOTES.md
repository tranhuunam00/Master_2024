# SiPAP Rev C (layout v3) — hồ sơ đặt mạch

> Xuất ngày 2026-11-21, xuất lại 2026-11-22 hai lần (xem "Cập nhật 2026-11-22" và "Sửa sau review đối kháng 2026-11-22" ở cuối). Đây là **bộ duy nhất dùng để đặt in**. Hai bộ `Fab_RevC/` và `Fab_RevC_v2/` là layout cũ, **không dùng nữa**.
>
> So với v2, bản v3 bố trí lại linh kiện và đi dây lại toàn bộ theo góp ý review. Schematic có 3 thay đổi, đều đã đồng bộ sang PCB, BOM và firmware:
> - đổi gán GPIO của ESP32;
> - thêm tụ C27 22 µF ở chân 3V3 của U1;
> - sửa mã encoder U6 trên PCB thành PEC11R-4225F cho khớp BOM.
>
> Chi tiết và số liệu kiểm tra: `REVIEW_v3.md`.

## Gửi xưởng
- **Chỉ in mạch trần:** gửi `SiPAP_RevC_v3_Gerber.zip`, gồm Gerber và file khoan.
- **Có lắp ráp (PCBA) ở JLCPCB:** gửi thêm `SiPAP_RevC_v3_BOM_JLC.csv` và `SiPAP_RevC_v3_CPL_JLC.csv`, xem mục "Đặt lắp ráp ở JLCPCB".
- **Có lắp ráp ở xưởng khác:** gửi thêm `BOM_RevC_v3.csv` và `SiPAP_RevC_v3_CPL.csv`. Hai file này đủ cả linh kiện THT.
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
| Via | 135 via khoan 0.30 / pad 0.60 mm; 4 via khoan 0.25 / pad 0.50 mm ở khe EPAD của U1. Tổng 139 via (71 GND), **phủ mask kín toàn bộ** |
| Lỗ khoan | 174 lỗ tròn (0.254 → 3.2 mm) và 5 lỗ rãnh G85 (3 ở chân jack J1, 2 ở tai encoder U6). Tất cả có mạ |
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
| `SiPAP_RevC_v3.ipc` | Netlist kiểm tra điện, 412 điểm (273 pad + 139 via) | IPC-D-356 |
| `SiPAP_RevC_v3_ODB++.zip` | Bản ODB++ cùng nội dung | ODB++ |
| `SiPAP_RevC_v3_CPL.csv` | Toạ độ gắp-đặt, mm, gốc ở góc dưới-trái | CSV |
| `BOM_RevC_v3.csv` | 44 dòng, có MPN, mã thay thế và mã LCSC | CSV |
| `SiPAP_RevC_v3_BOM_JLC.csv` | BOM theo cột của JLC, 29 dòng, chỉ SMD | CSV |
| `SiPAP_RevC_v3_CPL_JLC.csv` | Toạ độ gắp-đặt theo cột của JLC, 69 linh kiện SMD | CSV |
| `SiPAP_RevC_v3_pin1_check.csv` | Toạ độ chân 1 của 12 linh kiện có cực tính hoặc có hướng (thêm RV1), để so với ảnh preview của JLC | CSV |
| `review_evidence/` | Bằng chứng review: ảnh các trang datasheet được trích, SHA-256 của datasheet, kết quả kiểm Gerber/netlist, `SOURCES.md` | PNG / TXT |
| `SiPAP_RevC_v3.step` | Mô hình 3D (trục encoder đúng 31.5 mm) | STEP AP214 |
| `SiPAP_RevC_v3_Schematic.pdf` | Schematic 1 trang | PDF |
| `SiPAP_v3_TOP.png`, `SiPAP_v3_BOTTOM.png` | Ảnh 3D mặt trên / dưới (mặt dưới bị lật gương) | PNG |
| `SiPAP_v3_TOP_copper.png`, `SiPAP_v3_BOTTOM_copper.png` | Ảnh đồng dựng lại từ chính file Gerber, để review đường mạch | PNG |
| `REVIEW_v3.md` | Đối chiếu góp ý review với số đo, rà schematic/BOM có trích datasheet | — |

Gerber, file khoan và CPL dùng chung gốc toạ độ (góc dưới-trái board). Đã kiểm bằng cách đọc ngược file Gerber, xem mục "Trạng thái kiểm tra".

## Lắp ráp
- **SMD (xưởng gắn máy, 69 linh kiện):** U1, U2, U3, U7, U8, U9, Q1–Q4, D2, L1, L2, biến trở RV1 và 55 R/C 0805 (đã có C27, R30 mới). Có 3 fiducial FID1–FID3 Ø1 mm.
- **THT (hàn tay, 9 linh kiện):** J1, J2, J3, U4, U5, U6, F1, BZ1, C25. Trong CPL ghi "THT".
- **Không lắp:** TP1–TP4 (pad test 12V / 5V / 3V3 / GND, có nhãn silk) và FID1–FID3.
- **Dấu cực tính trên silkscreen** (mới ở v3, các footprint cũ thiếu):
  - D2: vạch cathode bên phải, cạnh pad 1 = cathode = VIN12V.
  - C25 (tụ hoá): dấu "+" bên phải, cạnh pad + (BLOWER_V12).
  - BZ1: dấu "+" bên trái.
  - Chấm tròn chân 1 cho Q1, Q4, U7, U8, U9, U2, U3, U1 và các header J2, J3, U4, U5.
- **J3 (cổng nạp UART):** nằm dọc mép phải, chân 1 (GND) ở trên cùng. Tên từng chân in trên silk: GND, 3V3, TX, RX, EN, IO0.
- Góc xoay trong CPL theo quy ước Altium. Nên yêu cầu xưởng gửi ảnh DFM preview để xác nhận góc của SOT-23-6, SO-8 và MPXV trước khi gắn.

## Đặt lắp ráp ở JLCPCB
Upload `SiPAP_RevC_v3_BOM_JLC.csv` và `SiPAP_RevC_v3_CPL_JLC.csv`.
- **Phạm vi:** 69 linh kiện SMD mặt trên, 29 dòng. 9 linh kiện THT không có trong 2 file này, hàn tay sau.
- **Cột đúng mẫu JLC.** BOM: Comment, Designator (ngăn bằng dấu phẩy), Footprint (tên gói như `0805`, `SOT-23-6`, không dùng tên footprint thư viện), LCSC Part #. CPL: Designator, Mid X, Mid Y, Layer, Rotation. Script đã kiểm: designator trong CPL và trong BOM trùng nhau đúng 67/67, số lượng mỗi dòng khớp.
- **Mã LCSC** tra ngày 2026-11-22, sau đó đối chiếu từng mã trên trang sản phẩm lcsc.com (`review_evidence/SOURCES.md`). 27/29 dòng có mã: 14 dòng basic (50 linh kiện) và 13 dòng extended (17 linh kiện). C4360 (R10 36k) và C48484 (RV1) là extended theo trang jlcpcb.com/partdetail. JLC tính thêm phí cho mỗi mã extended.
  - R/C dùng mã basic cùng giá trị, cùng sai số, điện áp bằng hoặc cao hơn và cùng điện môi như BOM, đúng luật thay thế ở mục dưới. Các tụ 10 µF thay bằng loại 50 V (BOM ghi 35 V), cũng X5R.
  - Dòng 22 µF 16 V và 22 µF 25 V gộp chung 1 mã loại 25 V (C45783). Hai dòng 100k cũng gộp làm một.
- **2 dòng chưa có mã: L1 (SRN6045TA-150M), L2 (SRN6045TA-100M).** Để JLC tự khớp theo MPN lúc đặt. Nếu JLC không có thì bỏ 2 dòng này ra và hàn tay. Nếu thay bằng cuộn khác, điều kiện ghi ở cột Notes, lấy theo datasheet SRN6045TA trang 1:
  - L1: 15 µH, Isat ≥ 3.8 A, Irms ≥ 2.8 A, DCR ≤ 71 mΩ;
  - L2: 10 µH, Isat ≥ 4.6 A, Irms ≥ 3.2 A, DCR ≤ 52 mΩ.
- **3 dòng cần để ý:**
  - **RV1 = C48484 (Bourns 3224W-1-503E, 50 kΩ ±10 %, 12 vòng).** Biến trở chỉnh ngưỡng cắt áp phần cứng; phải chỉnh sau khi lắp, xem "Kiểm tra sau khi nhận board" bước 3. Footprint và thứ tự chân theo datasheet Bourns 3224 trang 1. Nếu JLC hết hàng thì bỏ RV1 khỏi đơn và hàn tay (gói SMD 4.8 × 3.5 mm, pad 1.6 mm, dễ hàn).
  - **U2/U3 = C3238865 (MPXV5010GC6T1).** Đây là cùng cảm biến với GC6U, chỉ khác đóng gói: cuộn băng thay vì ống (bảng Ordering Information, datasheet MPX5010 của Motorola/Freescale, nay là NXP, trang 5). Lúc tra, LCSC chỉ còn 3 con. Nếu hết hàng thì bỏ U2/U3 khỏi đơn và hàn tay.
    - Gắn máy được: datasheet trang 6 nói gói SOP tự căn chỉnh khi hàn reflow.
    - Cảm biến được kiểm với khí khô (trang 3), nên giữ lỗ cổng áp không dính flux hay nước rửa.
  - **D2 = C1973187 (SMBJ13CA, 2 chiều)** thay cho SMBJ13A (1 chiều).
    - Hai mã nằm cùng một dòng thông số trong datasheet Bourns SMBJ trang 2: VRWM 13 V, VBR 14.4–15.9 V, VC 21.5 V @ 28 A. Datasheet của chính linh kiện được gắn (Vishay SMBJ5.0A thru SMBJ188A, trang 2, đã lưu ở `Cpap/tools/datasheets/Vishay_SMBJ.pdf`) cho cùng giá trị: VWM 13 V, VBR 14.4–15.9 V, VC 21.5 V @ 27.9 A.
    - D2 đứng sau Q4 trên VIN12V (J1 → F1 → Q4 → VIN12V → D2), nên không bao giờ bị cấp ngược cực. Vì vậy xoay chiều nào cũng được, và vạch cathode trên silk không còn ý nghĩa.
    - Muốn giữ đúng SMBJ13A thì xoá mã này đi để JLC tự khớp.
- **Góc xoay.** CPL dùng góc của Altium, trong khi thư viện JLC có thể đặt gốc khác cho một số gói như SOT-23 và SOIC-8.
  - Ở màn preview của JLC, so chấm chân 1 của từng linh kiện với `SiPAP_RevC_v3_pin1_check.csv`. File này có toạ độ chân 1 (mm) và dấu silk tương ứng của 11 linh kiện có cực tính.
  - U1: điểm CPL là tâm thân module, cao hơn tâm cụm pad 3.62 mm, đầu anten quay ra mép trên.
  - Q2/Q3 là SOT-23 ba chân, không có chấm, nên nhìn hướng chân đơn.

## Nếu xưởng thiếu linh kiện
BOM có cột **Alternate MPN** cho từng dòng chính. R/C 0805 dùng hãng nào cũng được, miễn đúng giá trị, sai số (1% cho R), điện áp và điện môi ghi trong BOM. Không thay TPS54202 bằng IC khác chân; chỉ TPS54202H tương thích. Mã LCSC đã chọn cho từng dòng nằm trong cột `LCSC #` và cột Notes của `BOM_RevC_v3.csv`. Vẫn cần duyệt từng dòng trên trang JLC trước khi thanh toán, vì tồn kho thay đổi hằng ngày.

## Firmware
Gán GPIO v3 nằm trong `Cpap/firmware/sipap_v1/config.h`. Bản sửa ngày 2026-11-22 (sau review) biên dịch đạt với arduino-esp32 3.3.8 (`esp32s3`, PSRAM OPI, flash 8 MB, bảng phân vùng mặc định): 679 626 byte chương trình, 31 980 byte RAM. **Không nạp firmware cũ lên board v3** vì chân đã đổi. Từ bản sửa này, BLE phải ghép đôi bằng mã PIN 6 số; mã hiện trên OLED lúc khởi động và in ra cổng Serial (mỗi board một mã, tính từ địa chỉ MAC). Lệnh `PWM:n` chỉ có khi biên dịch với `SIPAP_BENCH_MODE`.

| Tín hiệu | GPIO | Tín hiệu | GPIO |
|---|---|---|---|
| PRESS_PRI | IO4 | ENC_A / ENC_B / ENC_SW | IO47 / IO21 / IO48 |
| PRESS_RED | IO5 | BLOWER_PWM / BLOWER_FG | IO10 / IO11 |
| SAFETY_TRIP | IO9 | BUZZER | IO12 |
| I2C SDA / SCL | IO2 / IO1 | SFM_EN | IO16 |

## Kiểm tra sau khi nhận board
1. Trước khi cấp nguồn, đo điện trở 12V–GND, 5V–GND và 3V3–GND (TP1–TP4); không được chập.
2. Cấp 12V qua J1, giới hạn dòng 0.5 A. Đo TP2 = 5.0–5.15 V, TP3 = 3.25–3.35 V.
3. **Chỉnh ngưỡng cắt áp phần cứng bằng RV1 (bắt buộc với từng board).** Điện áp ra ở áp 0 của U3 có thể nằm bất kỳ đâu trong 0–0.425 V (datasheet MPX5010 trang 2). Nếu không chỉnh, ngưỡng cắt có thể rơi vào khoảng ~20–30 cmH2O.
   - Chạy self-test của firmware: giữ núm khi cấp nguồn, hoặc gửi `TEST` qua Serial. Mục 9 in ra điện áp COMP_REF cần đặt, tính từ điểm 0 đo được của U3.
   - Vặn RV1 cho tới khi U7 chân 2 (so với GND) đúng điện áp đó.
   - Tăng áp ở **ống chung của U2 và U3** (cút chữ T, dùng xi-lanh hoặc cột nước). Q1 phải cắt nguồn blower (J2 chân 1 về 0 V) ở 24–26 cmH2O, đọc bằng U2. Self-test báo PASS/FAIL và hướng vặn: chiều kim đồng hồ là tăng điện trở, tức là cắt ở áp thấp hơn (sơ đồ chân Bourns 3224 trang 1).
   - Khi mạch cắt, R18 đẩy số đọc của U3 lên khoảng 2.4 cmH2O. Đây là cơ chế tạo trễ, không phải lỗi.
4. Cấp ngược cực −12 V, giới hạn 0.3 A: dòng ≈ 0 (Q4 chặn).
5. Kéo IO16 xuống 0 thì U4 chân 2 (VDD của SFM3300) về 0 V; thả ra thì về 5 V (công tắc Q2/Q3).

## Trạng thái kiểm tra (2026-11-21, cập nhật số liệu sau lần sửa 2026-11-22)
| Kiểm tra | Kết quả |
|---|---|
| Altium DRC, 15 luật | **0 vi phạm**: clearance 10 mil, short, un-routed, width 3 nhóm, hole size, hole-to-hole 10 mil, mask sliver 4 mil, silk-to-mask 6 mil, silk-to-silk 6 mil, net antennae, height, modified polygon, routing topology |
| `pcb_layout_audit` của eda-agent | Đi dây 40/40 net, DRC 0, chồng linh kiện 0, mảng đồng chết 0. Góc gắt: 2/210 điểm nối, cả hai là chỗ rẽ nhánh 5V (3150, 1880 mil) và GND (3040, 1812.5 mil) có sẵn từ trước: hình học y hệt trong board ở commit 6f5d0b1, đọc bằng `review_evidence/scripts/tracks6.py`. Lần audit trước báo 0/197. Ngoại lệ ghi ở bảng dưới |
| Giới hạn JLCPCB tiêu chuẩn | Đạt (track 10 ≥ 4 mil, lỗ 0.254 ≥ 0.15 mm, vành khuyên 5 ≥ 2 mil, khe hở 10 ≥ 4 mil, lỗ–lỗ 10 ≥ 7.9 mil, mask sliver 4 ≥ 3.9 mil; đồng cách mép đo trên Gerber 0.647 mm ≥ 0.2 mm) |
| Đọc ngược Gerber/khoan | Viền đúng 101.6 x 76.2 mm. **273/273 pad** có đồng và mask đúng lớp; kem hàn đủ, trừ TP và FID (đúng ý). **139/139 via** có đồng 2 mặt và phủ mask kín. **179 lỗ** (bảng mũi khoan) |
| Schematic ↔ PCB | 78/78 linh kiện, `in_sync`. Netlist IPC-356 khớp từng pad với dump board (`tools/layout_v3/board_dump_v3_rv1.json`) |
| BOM ↔ board | 78 linh kiện, mỗi linh kiện đúng 1 dòng BOM, giá trị khớp comment |
| ERC | 0 vi phạm (2026-11-22). 7 cảnh báo "Nets with no driving source" trên COMP_REF, EN, FB3, FB5, GATE, RPP_G, SFM_G được tắt bằng dấu No ERC đặt đúng nhãn net, và chỉ tắt riêng loại cảnh báo này. Các net này chỉ được cấp qua điện trở (cầu chia hoặc kéo lên) |
| Chạm keep-out | 0 (2026-11-22). Vùng cấm anten và đoạn keep-out mép board chỗ U1 và J1 đã chuyển vào footprint của chính linh kiện đó, giữ nguyên hình học |
| Dây cáp (`check_harness.py`) | 19 đạt, 0 lỗi, 1 chờ dữ liệu blower |

**Trạng thái DRC:** bridge ghi verdict là `pass_unapproved_rules`, tức DRC đạt nhưng bộ luật chưa được bạn duyệt. Phiên này đã sửa 4 luật:
- silk-to-mask và silk-to-silk đặt 6 mil, theo mức tối thiểu 0.15 mm của JLC;
- tách luật độ rộng thành 3 nhóm: tín hiệu, nguồn 5V/3V3/GND/SW, 12V;
- nới chiều cao tối đa từ 1200 lên 1250 mil, cho trục encoder 31.5 mm.

Duyệt trong Altium bằng `pcb_drc_rules(action='approve')` nếu đồng ý.

**Ngoại lệ đã biết, có chủ đích:**
| Mục | Vì sao chấp nhận |
|---|---|
| 28/68 via tín hiệu không có via GND trong 40 mil (24 cũ + 4 via mới của REF_TOP và 5V cạnh RV1) | Đều là tín hiệu chậm hoặc DC: EN, ADC, buzzer, gate MOSFET, nguồn 3V3/5V, SW của buck, nút chia áp REF_TOP. Board không có tín hiệu tốc độ cao |
| 10 mảng GND rời ở mặt trên | Cả 10 đều nối xuống mặt phẳng GND liền khối ở mặt dưới qua via; 0 mảng chết |
| Silk viền module U1 cách mép trên 0.10 mm | Thân module đặt sát mép theo yêu cầu anten. Xưởng có thể cắt bớt nét silk này, không ảnh hưởng gì |
| 1 đoạn EN ở mặt dưới chạm mép thân R12 (9.9 mil) | Nằm ở mặt dưới, không nằm giữa 2 pad. Mặt trên: 0 đoạn dây lạ dưới thân linh kiện |

File OutJob để xuất lại: `CPAP_VSSM/SiPAP_Fab.OutJob` (Gerber, khoan, IPC-356, ODB++) và `SiPAP_Docs.OutJob` (PDF schematic). Script gom hồ sơ: `tools/layout_v3/fab_v3.py`.

Lưu ý: `SiPAP_Fab.OutJob` còn một mục in ra máy in (`Print To \\CLDFS01\Supply Room B&W`), sót lại từ mẫu của Altium. Máy in này không có trên máy đang dùng; lần xuất ngày 2026-11-22 không có lệnh in nào vào hàng đợi. Nên tắt mục này trong OutJob.

## Cập nhật 2026-11-22
**Thư viện và board:**
- Vùng cấm anten của U1 và đoạn keep-out mép board ngay chỗ U1 (mép trên) và J1 (mép trái) đã chuyển vào footprint `ESP32-S3-WROOM-1_DS` và `CUI_PJ-102AH`, giữ đúng toạ độ và bề rộng 30 mil như cũ. Đoạn keep-out mép board được cắt ra để nối tiếp hai đoạn này.
- Footprint `PEC11R-4xxxF-S_DS` trong thư viện sửa cao 31.5 mm cho khớp board: thân 6.5 mm cộng trục L = 25 mm (Bourns PEC11R DS tr.2-3).
- U1, J1, U6 được nạp lại từ thư viện. Cả 268/268 pad giữ nguyên vị trí, cỡ và net.
- Silk viền J1 trước đó đã xoá trên board để nhường chỗ cho chữ "12V IN"; nay xoá lại như cũ.

**Kết quả:** DRC 0, chạm keep-out 0, ERC 0.

**Xuất lại hồ sơ** rồi so từng lớp với bản ngày 2026-11-21. Cách so: vẽ cả hai bản thành ảnh 20 µm/điểm, rồi đếm điểm ảnh khác nhau.
- Silk, mask, kem hàn và đồng mặt dưới giống hệt.
- Đồng mặt trên lệch 0.0056 mm², quanh toạ độ (19.5, 30.5) mm. Đây là sai khác khi đổ đồng lại (repour), nằm xa chỗ sửa keep-out và nhỏ hơn mức xưởng làm được.
- Lỗ khoan giống hệt.
- Đọc ngược Gerber vẫn đạt: 268/268 pad, 135/135 via phủ kín, 175/175 lỗ, đồng cách mép 0.647 mm.

**Bước 6 (model STEP của hãng) chưa làm.** Bourns PEC11R, Same Sky PJ-102AH và Espressif ESP32-S3-WROOM-1 không có link tải STEP trực tiếp, chỉ tải được qua SnapEDA/Octopart (cần tài khoản) hoặc trang tài nguyên riêng của hãng. Thân 3D hiện tại là khối đùn, kích thước lấy từ datasheet (xem `../3D/height_map.md`). Nếu bạn tải file STEP về, có thể gắn vào thư viện bằng `lib_link_3d_model`.

## Sửa sau review đối kháng 2026-11-22
Review đối kháng (một lượt, chỉ đọc) tìm ra 2 lỗi lớn (M1, M2) và 3 lỗi nhỏ (m3–m5). Báo cáo, ảnh các trang datasheet được trích, SHA-256 của datasheet và kết quả kiểm tra nằm trong `review_evidence/` (bắt đầu từ `SOURCES.md`). Đã sửa:

**M1. Ngưỡng cắt áp phần cứng chỉnh được.**
- **Lý do:** datasheet MPX5010 trang 2 cho Voff = 0 / 0.2 / 0.425 V và VFSS = 4.275–4.725 V. Với cầu chia cố định cũ (R10 47k / R11 10k), ngưỡng cắt nằm đâu đó trong khoảng ~20.4–30.0 cmH2O (`review_evidence/checks/trip_calc.txt`).
- **Sửa:** R10 47k → 36k (C4360), cộng RV1 50 kΩ Bourns 3224W-1-503E (C48484) mắc nối tiếp phía 5V. COMP_REF chỉnh được trong 0.52–1.09 V (0.55 V nếu RV1 lệch −10 %), phủ khoảng cần có là 0.69–1.05 V.
- **An toàn khi hỏng:** wiper RV1 nối với đầu CW. Nếu phần tử điện trở bị hở hoặc wiper mất tiếp xúc, điện trở tăng, COMP_REF giảm và mạch cắt blower.
- **Layout:**
  - RV1 đặt dưới R10, tâm (33.27, 18.29) mm, xoay 90°.
  - Pad 1 của R10 trước đây là điểm nối của đường 5V, nên đường 5V giờ đi vòng bên dưới R10.
  - REF_TOP và đường 5V vào RV1 chui dưới BLOWER_FG bằng 2 cặp via (4 via 0.30/0.60 mm, phủ mask kín).
  - Dây nối chân 3 sang chân 2 của RV1 đi vòng ngoài thân linh kiện (giữ quy tắc không đi dây dưới thân).
- **Thư viện:**
  - Footprint `BOURNS_3224W` theo land pattern ở datasheet Bourns 3224 trang 1: pad 1.6 × 2.0 mm và 1.6 × 1.3 mm, cách nhau 2.9 mm và 2.5 mm.
  - Thân 3D là khối đùn 3.5 × 4.8 mm, cao 5.3 mm, cũng theo trang 1. Máy chưa có CadQuery/FreeCAD nên chưa xuất được STEP.
  - Ký hiệu schematic `TRIMPOT_3224W`: chân 1 = CCW, 2 = wiper, 3 = CW.

**M2. Firmware an toàn.**
- **Cửa sổ điểm 0** lấy đúng giới hạn datasheet: 0 < V ≤ 0.446 V (= 0.425 V × 5.25/5).
- **AL_ZERO** chỉ được xoá khi hiệu chuẩn điểm 0 lại thành công: blower dừng và không có xung FG trong 3 s. Nhấn núm hoặc gửi CLEAR qua BLE không xoá được nữa.
- **BLE** bắt buộc ghép đôi bằng passkey, có MITM và bonding (làm theo ví dụ `Server_secure_static_passkey` của arduino-esp32 3.3.8).
- **Lệnh `PWM:n`** chỉ có trong bản biên dịch với `SIPAP_BENCH_MODE`. Ở chế độ tay vẫn báo áp cao khi vượt SET_MAX + 5 cmH2O.
- **PRESS_VS** = 5.077 V, tức 0.596 V × (1 + 100k/13.3k), theo TI SLVSD26C trang 5.
- **Self-test mục 9** tính điện áp COMP_REF cần đặt và chỉ hướng vặn RV1.

**m3. Thêm R30 100k kéo BLOWER_VSR xuống GND.** Đặt dưới R28, tâm (18.03, 11.94) mm. Pad GND nối vào via GND có sẵn ở (760, 530) mil.

**m4. Datasheet đúng linh kiện được gắn.**
- Thêm `Vishay_SMBJ.pdf` (D2, C1973187), `onsemi_LM393.pdf` (U7, C7955), `JSCJ_2N7002.pdf` (Q3, C8545) và `Bourns_3224.pdf` (RV1).
- Xoá 2 file hỏng: `MPX5010.pdf` (một trang lỗi HTML) và `EC11.pdf` (một ảnh JPEG).
- MPX5010 vẫn dùng bản Motorola 2001 (`MPX5010_b.pdf`), vì link nxp.com/docs/en/data-sheet/MPX5010.pdf trả lỗi 404 ngày 2026-11-22.

**m5. Mô tả J1.** BOM giờ ghi chân giữa Ø2.0 mm, 24 V / 5.0 A (PJ-102AH DS trang 1–2).

**Kiểm tra lại sau khi sửa:**
- **ERC:** 0 (13 mục bị chặn, giống trước).
- **DRC:** 0 vi phạm (kết luận `pass_unapproved_rules`).
- **Audit:** 40/40 net đã đi dây; 0 chồng linh kiện; 0 mảng đồng chết.
- **Schematic ↔ PCB:** 78/78 linh kiện.
- **Gerber đọc ngược:** 273/273 pad, 139/139 via phủ kín, 179 lỗ.
- **IPC-356:** REF_TOP = R10.1, RV1.2, RV1.3; net 5V có RV1.1; BLOWER_VSR có R30.1.
- **XOR với bản ở commit 6f5d0b1:**
  - GBS và GM giống hệt.
  - GTL, GBL, GTO, GTS, GTP chỉ khác trong vùng x 14.8–37.2 mm, y 11.1–32.2 mm (quanh RV1, R10, R30).
  - Đồng ở y > 27.5 mm chỉ khác 0.06 mm², do đổ đồng lại.
  - Mọi thay đổi cách mép board ≥ 11 mm, nên khoảng cách đồng–mép 0.647 mm vẫn đúng.
- **Firmware:** biên dịch đạt.

**Còn mở:**
- Khi chạy ECO, Altium chưa tìm thấy footprint BOURNS_3224W, vì bộ đệm thư viện được nạp lúc mở Altium. RV1 đã được đặt lên board kèm UniqueId của schematic, nên schematic và PCB vẫn khớp. Nên khởi động lại Altium trước lần ECO tiếp theo.
- Ảnh `SiPAP_v3_TOP.png` và `SiPAP_v3_BOTTOM.png` (render 3D) vẽ từ trước lần sửa này.
- Datasheet Bourns ghi rằng sản phẩm Bourns không được khuyến nghị cho ứng dụng y tế sống còn (life-critical medical) nếu không có phê duyệt bằng văn bản của Bourns (Bourns 3224 DS trang 3). Board có các linh kiện Bourns: RV1, F1, L1/L2, U6. Đây là giới hạn của bản thiết kế nghiên cứu này.
- Bộ luật DRC vẫn chờ bạn duyệt.
