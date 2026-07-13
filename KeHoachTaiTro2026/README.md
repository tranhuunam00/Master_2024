# Hướng dẫn tạo tài liệu Thông báo Kế hoạch tài trợ năm 2026

Thư mục này được tạo tự động để phục vụ việc soạn thảo và xuất bản tài liệu **Thông báo Kế hoạch tài trợ nhiệm vụ nghiên cứu phát triển công nghệ năm 2026**.

## Các tệp tin trong thư mục
1. **[thong_bao.md](file:///c:/Users/ADMIN/OneDrive/M%C3%A1y%20t%C3%ADnh/Master_2024/KeHoachTaiTro2026/thong_bao.md)**: Nội dung chi tiết của thông báo dưới định dạng Markdown để dễ dàng xem và sao chép nhanh.
2. **[generate_docx.py](file:///c:/Users/ADMIN/OneDrive/M%C3%A1y%20t%C3%ADnh/Master_2024/KeHoachTaiTro2026/generate_docx.py)**: Mã nguồn Python sử dụng thư viện `python-docx` để tự động tạo file Word (.docx) được định dạng chuẩn quy chuẩn hành chính Nhà nước (Nghị định 30/2020/NĐ-CP).

## Hướng dẫn tạo file Word (.docx) chuẩn
Để tự động tạo ra file Word định dạng chuẩn (Font chữ Times New Roman cỡ 13pt, lề trái 3cm, các lề còn lại 2cm, dãn dòng 1.15, căn lề 2 bên, tiêu đề và khối chữ ký căn giữa thẳng hàng), bạn làm theo các bước sau:

1. Mở terminal tại thư mục này hoặc chạy lệnh cmd/powershell.
2. Đảm bảo bạn đã cài đặt thư viện `python-docx`:
   ```bash
   pip install python-docx
   ```
3. Chạy tập lệnh python để sinh file Word:
   ```bash
   python generate_docx.py
   ```
Tệp tin **`Thong_Bao_Ke_Hoach_Tai_Tro_2026.docx`** sẽ được tạo ngay trong thư mục này với định dạng chuyên nghiệp.
