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
| Schematic PCB | `sipap_v1.kicad_sch` | `Cpap/hardware/CPAP_VSSM/SiPAP_Schematic.SchDoc` (Altium thật) | **Xong + ERC sạch (0 lỗi)** — 30 linh kiện, đã verify từng pinout theo datasheet thật |
| Thư viện symbol | — | `Cpap/hardware/CPAP_VSSM/SiPAP_Parts.SchLib` | **Xong** — 7 symbol tạo tay theo datasheet thật (ESP32-S3-WROOM-1, MPXV5010GC6U, SFM3300-D, PEC11R, SSD1306, LM393, buck module) |
| Thư viện footprint | — | `Cpap/hardware/CPAP_VSSM/SiPAP_Footprints.PcbLib` | **Xong** — 12 footprint, đã gán cho cả 30 linh kiện. ESP32-S3 footprint (41 pad castellated) đã verify số đo khớp datasheet 100% |
| PCB layout (đặt vị trí + đi dây thật) | — | `Cpap/hardware/CPAP_VSSM/PCB2.PcbDoc` (cần đổi tên) | **Chưa làm** — bước tiếp theo |
| BOM.csv | `Cpap/hardware/BOM.csv` | (giữ nguyên dự kiến) | **Chưa làm** — có sẵn bảng giá 15 dòng cũ trong `generate_docx.py` làm nền, cần bổ sung part number/nhà phân phối thật |
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
