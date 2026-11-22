# Model SPICE dùng trong mô phỏng

File model không được commit vào repo, vì đây là tài liệu của hãng. Hãy tải về từ link gốc, đặt vào `C:\Users\ADMIN\EDA Agent\workspace\spice\` (đường dẫn chỉ có ký tự ASCII, để Altium Mixed Sim đọc được), rồi so SHA-256.

| Linh kiện | File | Nguồn | Trạng thái | SHA-256 |
|---|---|---|---|---|
| U7 LM393 | `lm393.lib` (subckt `LM2903B`) | TI, SLCJ016B, trang sản phẩm LM393 → Design & development → Simulation model | Đọc được, đã chạy trong Altium Mixed Sim | `bb640f9b040d51a5ebaec7cb9a8a37bfcf2ae53a6c7e4a28405226b67ca43ff3` |
| Q3 2N7002 | `2N7002_Diodes.spice.txt` (subckt `N7002`, D G S) | Diodes Inc., https://www.diodes.com/spice/download/1941/2N7002.spice.txt (rev 2017-02-09) | SPICE3 dạng chữ thường (MOSFET LEVEL 3 + diode). Đã chạy trong Altium Mixed Sim ngày 2026-11-22 | `ca0fa03b0ff97da1ba91235cf8882ff82a1fe936dfa91d384ae168ec2a35f241` |
| U8/U9 TPS54202 | `TPS54202_TRANS.LIB` | TI SLVMBJ5C (tên là "Unencrypted PSpice Transient Model", rev C) | **Không dùng được.** File bắt đầu bằng `**$ENCRYPTED_LIB`, chỉ Cadence PSpice mở được. Trong mô phỏng thay bằng nguồn ổn áp theo hành vi | — |
| Q1/Q4 AO4407A, Q2 AO3401A | — | AOS: trang sản phẩm chỉ có datasheet, marking, reliability, package (kiểm 2026-11-22) | **Hãng không công bố model.** Dùng mô hình theo thông số datasheet | — |

Kiểm ngày 2026-11-22.

Chạy lại: copy `../sim_normal_operation.cir` sang thư mục spice ở trên, rồi gọi `sim_run_deck(deck=<nội dung file>, file_path=...\sim_normal_v3.nsx)`. Kết quả ghi trong `../README.md`.
