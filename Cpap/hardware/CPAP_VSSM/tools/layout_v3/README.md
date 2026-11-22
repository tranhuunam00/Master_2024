# Layout v3: quy tắc và công cụ đi dây

**Bộ script đã tạo ra board gửi xưởng nằm trong `final/`** (2026-11-21, hồ sơ `Fab_RevC_v3/`). Các file `.py` ở thư mục gốc này là lần thử trước đó. Chúng được giữ lại để tham khảo, nhưng không tạo ra board hiện tại.

Lần sau sửa board thì sửa script trong `final/` rồi chạy lại, không vá tay từng đoạn dây trong Altium.

## Quy tắc (bắt buộc)

Các quy tắc dưới đây lấy từ góp ý của người review bản v1/v2 và datasheet của nhà sản xuất.

1. **Dưới thân linh kiện** chỉ được có đoạn dây ngắn nhất đi từ chân của chính linh kiện đó ra mép thân. Dây của net khác không được chạy dưới thân. Router chặn cứng vùng thân và chỉ chừa hành lang thoát cho chân của chính linh kiện; không dùng cách cộng chi phí phạt.
2. **Độ rộng dây theo nhóm net:**
   - tín hiệu 10 mil;
   - 5V/3V3/GND 20 mil (riêng dây cảm biến áp ra về FB 10 mil vì không mang dòng);
   - SW 25–30 mil;
   - 12V 50 mil, VIN_RAW/VIN_FUSED 60 mil.

   Chân MOSFET công suất (Q1, Q4, gói SO-8) nối bằng vùng đồng. Luật width trong Altium tách thành 3 nhóm tương ứng.
3. **Buck TPS54202** theo TI SLVSD26C, Hình 7-16 "Board Layout", trang 21:
   - đồng VIN và đồng GND tách nhau, tụ ngõ vào bắc ngang ranh giới;
   - nút SW xuống mặt dưới qua 2 via rồi lên cuộn cảm;
   - FB lấy áp ngay tại tụ ngõ ra (Kelvin), đặt xa nút SW;
   - hai khối 5V và 3V3 giống hệt nhau (`hand.buck()` / `place_v3.buck_rel()`).
4. **Chỉ dùng góc 45°**, mỗi đoạn tối thiểu 2 ô lưới. Router:
   - lưới 10 mil, raster an toàn 2.5 mil;
   - đi chéo chi phí ×1.65, mỗi lần rẽ +25, mỗi via +250;
   - mặt dưới chi phí ×2.2 để giữ mặt GND liền.
5. **Thứ tự:** đặt linh kiện → đồng nguồn vẽ tay (`hand.py`) → fan-out U1 và các bus vẽ tay (UART/BOOT, I2C) → router cho phần còn lại, sắp thứ tự theo lỗi (`iterate.py`) → via GND kèm (dog-bone, ≤ 40 mil) → kiểm offline (`verify.py`, `check_place.py`) → đưa vào Altium.
6. **Không via trong pad SMD.** Khe hở 10 mil, cách mép board 20 mil, chừa vùng anten (x 1560–2440, y > 2745 mil) và lỗ bắt vít (r 150 mil).
7. **Chân, footprint, chiều cao** luôn đối chiếu datasheet trong `Cpap/tools/datasheets/`, không suy đoán.
8. **Sau mỗi lần đưa vào Altium**, chạy:
   - `pcb_layout_audit` (góc, pad-entry, via hồi GND, đảo đồng);
   - DRC Altium;
   - `proj_get_differences` và so net từng chân;
   - đọc ngược Gerber (`gerber_check.py`).

## `final/` — file

| File | Việc |
|---|---|
| `edacall.py` | Gọi tool eda-agent từ Python (`call(name, args)`). Ghim session tag để mọi tiến trình cùng quyền sửa tài liệu |
| `keepalive2.py` | Ping bridge mỗi 3 phút (bridge tự tắt sau 10 phút không có lệnh) và giữ máy không ngủ. Xoá `keepalive2.run` để dừng |
| `v2/*.json` | Dump hình học pad/thân linh kiện, dùng làm nguồn footprint cục bộ |
| `v3/model.py`, `v3/netmap.py` | Mô hình board; gán lại GPIO của U1 (`U1_REMAP`) |
| `v3/place_v3.py` | Vị trí từng linh kiện: khối buck, bảo vệ ngõ vào, mạch blower, so sánh an toàn, cảm biến, I2C, SFM |
| `v3/hand.py` | Đồng vẽ tay: ô buck, trục 12V, vùng đồng MOSFET, EPAD U1, góc 3V3/EN của U1, bus UART/BOOT, I2C |
| `v3/astar.py`, `v3/route.py`, `v3/iterate.py` | Router A* (numba) và vòng sắp thứ tự theo lỗi, ra `route_best.json` |
| `route_final.json` | Kết quả cuối đã đưa vào board: 16 vùng đồng, 373 dây, 135 via |
| `v3/verify.py`, `v3/check_place.py`, `v3/render.py` | Kiểm offline (clearance, mép, anten, via-trong-pad, dưới thân, góc; chồng courtyard; pad với đồng tay) và vẽ PNG |
| `apply_pcb1.py` / `apply_pcb2.py` / `apply_pcb3.py` | Đưa vào Altium: gỡ dây, dời linh kiện, thêm C27; đặt vùng đồng, dây, via |
| `sch_edit1.py` | Sửa schematic: nhãn GPIO của U1, thông số U6 |
| `silk_labels.py`, `silk_labels2.py`, `fix_designators.py`, `polarity_marks.py` | Silkscreen: nhãn, tên linh kiện đọc được (0°/90°), dấu cực tính và chân 1 |
| `gerber_check.py` | Đọc ngược Gerber + Excellon, so với dump board: pad, via, lỗ khoan, khoảng cách tới mép |
| `SiPAP_Fab_src.OutJob`, `SiPAP_Docs_src.OutJob` | Nguồn của 2 OutJob trong project (Gerber/khoan/IPC/ODB và PDF schematic) |
| `../fab_v3.py` | Gom hồ sơ `Fab_RevC_v3/`: đổi tên, zip, CPL, BOM, kiểm BOM ↔ board |

## Lưu ý đã gặp
- Net của pad U1 đặt bằng `pcb_bind_pad_nets` không giữ được sau khi lưu và mở lại; dùng `obj_modify` (ePadObject, `Net=`) mới giữ được.
- `proj_run_output` (gerber/drill) không ghi file nếu không có người bấm hộp thoại. Phải dùng OutJob (`proj_generate_fab_package` với `fab="JLCPCB"`).
- Chạy DRC có thể bật hộp thoại "Repour Polygons and Run DRC"; trả lời bằng `app_press_dialog_button`.
