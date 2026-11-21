# Layout v3: quy tắc và công cụ đi dây

Thư mục này chứa đúng bộ script đã dùng để bố trí và đi dây lại SiPAP Rev C (layout v3, 2026-11). Lần sau sửa board thì sửa ở đây rồi chạy lại, không đi dây tay từng đoạn trong Altium.

## Quy tắc (bắt buộc)

Các quy tắc dưới đây lấy từ góp ý của người review bản v1 và datasheet của nhà sản xuất.

1. **Dưới thân linh kiện** chỉ được có đoạn dây ngắn nhất đi từ chân của chính linh kiện đó ra mép thân. Dây của net khác không được chạy dưới thân, kể cả khi net đó có chân trên linh kiện (ví dụ U1). Router chặn cứng điều này (`masks_for` → hành lang thoát chân), không dùng chi phí phạt.
2. **Độ rộng dây:** tín hiệu 10 mil, 5V/3V3/SW 20 mil, 12V 50 mil. Chân MOSFET công suất (Q1, Q4, gói SO‑8) nối bằng vùng đồng (region), không nối bằng dây mảnh.
3. **Buck TPS54202:** bố trí theo TI SLVSD26C, Hình 7‑16:
   - Đồng GND (phía dưới IC) và đồng VIN (phía trên IC) tách nhau; tụ ngõ vào bắc ngang ranh giới giữa hai vùng.
   - Nút SW xuống via sang mảng đồng ở lớp dưới, rồi lên lại cuộn cảm.
   - Cầu phân áp FB đặt xa nút SW và lấy áp phản hồi ngay tại tụ ngõ ra (Kelvin).
   - Hai khối 5V và 3V3 giống hệt nhau (hàm `tile()` / `buck()`).
4. **Chỉ dùng góc 45°.** Góc 90° bị phạt nặng, góc gắt hơn bị cấm. Lớp dưới có chi phí ×2.2 để giữ mặt GND ít bị cắt. Mỗi via cộng chi phí tương đương 350 mil dây.
5. **Thứ tự đi dây:**
   - Fan‑out của U1 vẽ tay: chân trên cùng đi làn ngoài cùng, các làn cách nhau 25 mil.
   - Sau đó đi các net của U1 theo thứ tự chân, tiếp theo là 5V và 3V3, cuối cùng là các net còn lại.
6. **Không via trong pad SMD.** Khe hở 10 mil (JLCPCB 2 lớp, 1 oz).
7. **Chân và footprint** luôn đối chiếu với datasheet của nhà sản xuất (file trong `Cpap/tools/datasheets`), không suy đoán.
8. **Sau mỗi lần chạy:** chạy `pcb_layout_audit` (góc dây, cách vào pad, via hồi GND, đảo đồng) và DRC của Altium.

## File

| File | Việc |
|---|---|
| `eda.py` | Gọi tool của eda-agent từ dòng lệnh: `python eda.py TOOL '{json}' [out.txt]`. Viết `%PRJ%` để thay cho thư mục project. |
| `model.py` | Đọc toạ độ pad và thân linh kiện từ dump `c_pads.txt` / `c_comps.txt` (lấy bằng `obj_query` / `pcb_get_components`). |
| `placement.py` | Vị trí mới của từng linh kiện (khối buck, khối bảo vệ ngõ vào, mạch gate blower, …). |
| `power.py` | Đồng nguồn vẽ tay: vùng đồng và dây của buck, trục 12V, vùng đồng MOSFET, fan‑out U1. |
| `router.py` | Router lưới 5 mil, 2 lớp, 45°, A* có trọng số. Ghi kết quả ra `route.json`. |
| `preview.py` | Vẽ PNG để kiểm tra trước khi đưa vào Altium. |
| `apply.py` | Tạo các file tham số cho eda-agent (dời linh kiện, đặt vùng đồng, dây, via). |

## Chạy

```
python eda.py obj_query '{"object_type":"ePadObject","properties":"Component,Name,Net,X,Y,Layer,TopXSize,TopYSize,Rotation,HoleSize,TopShape"}' c_pads.txt
python eda.py pcb_get_components '{}' c_comps.txt
python router.py            # -> route.json
python preview.py check.png route.json
python apply.py             # -> a_*.json, rồi gọi eda.py với từng file
```
