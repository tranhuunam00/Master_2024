# SiPAP — vỏ hộp v0.1 (FreeCAD, tham số hoá)

Dựng lại: `"D:\Program Files\FreeCAD 1.1\bin\freecadcmd.exe" sipap_enclosure.py` (đường dẫn có dấu tiếng Việt thì đặt `SIPAP_OUT` ra thư mục ASCII), rồi `python render_preview.py out`.
Mọi kích thước là tham số đầu file `sipap_enclosure.py` — đổi blower / dung tích bình chỉ cần sửa số và chạy lại.

![](out/preview_open.png)

## Bố cục (trái → phải), ngoài 308.6 × 125 × 81.5 mm
| Khoang | Nội dung | Ghi chú |
|---|---|---|
| Điện tử | PCB Rev C trên 4 trụ M3 cao (board cách nắp 18 mm) | Board **cao hơn mực nước** — tràn nước không chạm mạch. Jack 12 V qua vách trái; trục encoder nhô 6 mm khỏi nắp; lỗ còi + cửa sổ OLED trên nắp |
| Blower | Blower giả định cỡ WM7040 (70 × 70 × 40) trên 4 trụ cao su (grommet), lưới hút gió + khe lọc ở vách sau, SFM3300 nằm ngang phía trên | Vách ngăn kín với khoang điện tử (chỉ 1 khe cáp + 1 lỗ ống áp) → giảm ồn/rung lan sang |
| Làm ẩm | Bình nước tháo rời trượt ra bên phải, khí vào phía trên mực nước (có gờ chắn bắn), ra côn 22 mm ISO ở mặt phải — ống thở cắm thẳng vào bình | **Rev C: làm ẩm thường (không sưởi)**, 315 mL tại vạch MAX 35 mm. Chừa 6 mm dưới đáy bình cho đĩa sưởi ở Rev D |

Nắp: 6 vít M3 vào insert nhiệt (heat-set) — mở ra là thấy toàn bộ board và cổng nạp J3.

## Kiểm tra tự động (`out/report.json`)
- 4 chi tiết in đều là khối đặc hợp lệ.
- Va chạm (mm³): PCB–đáy 0, PCB–nắp 0, bình–đáy/nắp 0, blower–đáy 0, SFM–blower/đáy/nắp 0.
- PCB ghép từ `SiPAP_PCB_RevC.step` (căn theo mặt trên board).

## Còn mở / cần quyết
1. **Blower thật**: khối blower là giả định. WM7040-12V ghi 4.8 A / 57.6 W và 45 dB(A) — **vượt F1 4 A và adapter 3.5 A**, lại ồn. Cần chọn blower ≤ 3 A ở 12 V, hoặc chuyển nguồn 24 V ở Rev D.
2. **Bump trên nắp** che cầu chì F1 cao 24.8 mm — bỏ được nếu Rev D đổi sang PTC dán (SMD).
3. **STEP từ Altium**: board chỉ dày 0.39 mm (thiết lập layer stack) thay vì 1.6 mm — không ảnh hưởng Gerber (xưởng làm theo 1.6 mm đặt hàng), nhưng nên sửa layer stack cho file 3D đúng.
4. Chưa có: gioăng bình, đệm chống rung, tiêu âm đường hút, chân đế cao su, khắc chữ — làm ở v0.2 khi có blower.
