# SIPAP — Requirements & Output tracking

Ghi lại để đọc lại sau. Cập nhật mỗi khi kế hoạch/tiến độ đổi.

## Thay đổi lớn so với plan gốc

Plan ban đầu định làm PCB bằng **KiCad** (vì tưởng không có Altium). Sau đó phát hiện
**Altium Designer đã cài sẵn và đang chạy** trên máy → chuyển sang điều khiển Altium
**trực tiếp qua MCP server `altium-mcp`**. Vì vậy:
- File schematic/PCB bây giờ là **file Altium gốc thật** (`.SchDoc`/`.PcbDoc`/`.PcbLib`),
  KHÔNG phải KiCad. Phần "không tạo được Altium gốc" trong plan cũ đã **không còn đúng nữa**.
- Vị trí file cũng đổi: `Cpap/hardware/CPAP_VSSM/` (project Altium), không phải
  `Cpap/hardware/sipap_v1.kicad_*` như dự định ban đầu.

## Giai đoạn A — Điện tử & Firmware

| Output | Vị trí dự kiến ban đầu | Vị trí thực tế | Trạng thái |
|---|---|---|---|
| Firmware ESP32-S3 tích hợp (áp suất + dự phòng + flow sensor + PID blower + BLE + OLED + encoder + còi báo) | `Cpap/firmware/sipap_v1/*.ino` | (giữ nguyên dự kiến) | **Chưa làm** — mới đọc xong `dull.ino` + `LEDCFade.ino` (driver SFM3300), chưa viết file thật |
| Schematic PCB | `sipap_v1.kicad_sch` | `Cpap/hardware/CPAP_VSSM/SiPAP_Main_RevB.SchDoc` (Rev B, 2026-11-18) | **Xong, ERC 0 lỗi** — vẽ lại 45 linh kiện. Sheet cũ `SiPAP_Schematic.SchDoc` (gỡ khỏi project) có lỗi: J1 VIN chập GND, buck U8 không có đầu vào, pull-up I2C hở, 12V lọt vào GPIO6 ESP32, không có cổng nạp |
| Thư viện symbol | — | `Cpap/hardware/CPAP_VSSM/SiPAP_Parts_v2.SchLib` | **Xong** — tạo lại từ schematic (file `SiPAP_Parts.SchLib` gần như rỗng), sửa hướng chân trái (trước đây đầu nối nằm trong thân), thêm `CONN_UART6` |
| Thư viện footprint | — | `Cpap/hardware/CPAP_VSSM/SiPAP_Footprints.PcbLib` | **Xong (Rev B)** — 3 footprint cũ SAI so với datasheet đã dựng lại: `ESP32-S3-WROOM-1_DS` (DS v1.8 Fig 11-1; cũ bước 0.9mm, EPAD 15.8x17.6mm), `MPXV5010GC6U_482A` (DS Fig 5; cũ 2 hàng cách 7.62mm + lật gương, đúng 16.76mm), `PEC11R-4xxxF-S_DS` (DS tr.2; cũ bước 5mm, thiếu 2 tai). Thêm `DIOM_SMC_DO-214AB`. Footprint cũ vẫn nằm trong lib nhưng không dùng |
| PCB layout (đặt vị trí + đi dây thật) | — | `Cpap/hardware/CPAP_VSSM/SiPAP_PCB.PcbDoc` | **Xong (Rev B, 2026-11-18)** — 2 lớp, 101.6x76.2mm, 4 lỗ M3, đổ GND 2 mặt + via khâu. DRC: 0 lỗi điện/va chạm 3D; còn 308 cảnh báo silkscreen (silk footprint cũ đè pad, xưởng tự cắt) |
| Mô hình 3D + bảng chiều cao cho vỏ | — | `Cpap/hardware/CPAP_VSSM/3D/` (`SiPAP_PCB_RevB.step`, `height_map.md`) | **Xong** — khối 3D theo datasheet cho ESP32/MPXV5010/PEC11R; module buck, jack DC, còi là GIẢ ĐỊNH. Cao nhất 26.5mm (trục encoder) |
| BOM.csv | `Cpap/hardware/BOM.csv` | `Cpap/hardware/BOM.csv` | **Xong (bản nháp)** — 45 linh kiện, MPN gợi ý cần xác nhận trước khi đặt hàng |
| Gerber + drill + CPL | `Cpap/hardware/gerber/` | (giữ nguyên dự kiến, xuất từ Altium thay vì KiCad) | **Chưa làm** |
| Sơ đồ dây nối ngoài PCB | `Cpap/hardware/wiring_harness.md` | (giữ nguyên dự kiến) | **Chưa làm** |

## Giai đoạn B — Cơ khí & Khí động học

| Output | Vị trí | Trạng thái |
|---|---|---|
| `enclosure.py` (CadQuery) + `.step`/`.stl` | `Cpap/mechanical/` | **Chưa làm** — chờ xong PCB để lấy đúng toạ độ lỗ bắt vít (không đoán) |
| Ốc vít | M3, lỗ clearance Ø3.2mm, boss Ø~6mm, khớp đúng 4 lỗ mounting trên PCB | Ghi nhận yêu cầu, chưa làm |
| `layout_diagram.png` — vị trí từng khối (PCB, blower, buồng mica+humidifier, OLED, encoder, cổng khí) bên trong vỏ | `Cpap/mechanical/` | **Chưa làm** |
| CFD đường khí (OpenFOAM) | `Cpap/cfd/` | **Chưa làm** — đã kiểm tra: máy chưa cài OpenFOAM/ANSYS, cần cài thêm (vài GB, có thể mất nhiều thời gian) |

## Giai đoạn C — An toàn & Hồ sơ

| Output | Vị trí | Trạng thái |
|---|---|---|
| Hồ sơ rủi ro (ISO14971-style) | `Cpap/document/risk_analysis_v1.md` | **Chưa làm** |
| Mẫu báo cáo test/calibration từng máy | `Cpap/document/test_report_template.md` | **Chưa làm** |
| Hướng dẫn sử dụng sơ bộ (IFU) | `Cpap/document/ifu_draft.md` | **Chưa làm** |

## Quyết định kỹ thuật đã chốt (áp dụng xuyên suốt)

- MCU: **ESP32-S3** (thay Nano33+ESP32 rời) — BLE+WiFi, hỗ trợ AI on-device, 1 chip thay 2.
- An toàn phần cứng độc lập firmware: cảm biến áp suất dự phòng (MPXV5010GC6U #2) + mạch
  comparator (LM393) so với ngưỡng quá áp + MOSFET (AO3401A) cắt nguồn blower khi vượt ngưỡng,
  độc lập hoàn toàn với code chạy trên ESP32.
- Hàn: PCBA SMD tại xưởng + tự hàn tay connector/module lớn.
- PCB: thiết kế bằng Altium trực tiếp (không qua KiCad nữa).
- Vỏ: M3, in 3D.
- CFD: mức đầy đủ (không chỉ tính giải tích), nhưng phụ thuộc cài được solver trên máy.

## Việc tiếp theo (đang làm dở)

1. Dọn lại ERC project (đang gỡ vài sheet rác còn sót trong project Altium gây nhiễu kết quả kiểm tra).
2. PCB layout thật (đặt vị trí + đi dây đồng) trong `PCB2.PcbDoc`.
3. BOM.csv đầy đủ.
4. Firmware `.ino` thật.
5. Gerber + wiring harness.
6. Cơ khí (enclosure) + CFD.
7. Hồ sơ an toàn/IFU.
