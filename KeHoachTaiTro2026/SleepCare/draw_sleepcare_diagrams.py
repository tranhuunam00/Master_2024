import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def set_style():
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'

def draw_arrow(ax, x1, y1, x2, y2, label="", color="gray", style="-", label_pos=0.5, text_color="black"):
    arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=15, 
                                    color=color, lw=1.8, linestyle=style)
    ax.add_patch(arrow)
    if label:
        x_text = x1 + (x2 - x1) * label_pos
        y_text = y1 + (y2 - y1) * label_pos
        ax.text(x_text, y_text, label, ha='center', va='center', 
                fontsize=8, fontweight='bold', color=text_color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

def draw_architecture():
    set_style()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette
    color_ui = "#E3F2FD"      # Soft Blue
    color_logic = "#E8F5E9"   # Soft Green
    color_hw = "#FFF3E0"      # Soft Orange
    color_server = "#F3E5F5"  # Soft Purple

    border_ui = "#1E88E5"
    border_logic = "#43A047"
    border_hw = "#FB8C00"
    border_server = "#8E24AA"

    box_ui = dict(boxstyle="round,pad=0.4", fc=color_ui, ec=border_ui, lw=2)
    box_logic = dict(boxstyle="round,pad=0.4", fc=color_logic, ec=border_logic, lw=2)
    box_hw = dict(boxstyle="round,pad=0.4", fc=color_hw, ec=border_hw, lw=2)
    box_server = dict(boxstyle="round,pad=0.5", fc=color_server, ec=border_server, lw=2.5)

    # Place Nodes
    # Hardware/Sensors (Left)
    ax.text(2.0, 7.5, "MICROPHONE\n(Thu âm thanh ngủ)", ha='center', va='center', bbox=box_hw, fontsize=9.5, fontweight='bold', color="#E65100")
    ax.text(2.0, 5.0, "CẢM BIẾN ÁNH SÁNG\n/ CAMERA TRƯỚC\n(Đo cường độ sáng)", ha='center', va='center', bbox=box_hw, fontsize=9.5, fontweight='bold', color="#E65100")
    ax.text(2.0, 2.5, "MOBILE BLUETOOTH\n(Nhận dữ liệu từ SIPAP)", ha='center', va='center', bbox=box_hw, fontsize=9.5, fontweight='bold', color="#E65100")

    # UI Layer (Center Top)
    ax.text(6.0, 8.5, "GIAO DIỆN NGƯỜI DÙNG (UI Layer)\n(KiemTraBenhGiacNgu, SnoreDetectionPage,\nRoomQualityPage, PhongKhamList)", ha='center', va='center', bbox=box_ui, fontsize=10, fontweight='bold', color="#0D47A1")

    # Logic Layer / Cubits (Center)
    ax.text(6.0, 5.0, "TẦNG XỬ LÝ LOGIC (Cubit/BLoC)\n\nSnoreDetectionCubit (Nhận dạng ngáy AI)\nRoomQualityCubit (Tính độ ồn, độ sáng)\nLocalStorage (Shared Preferences)", ha='center', va='center', bbox=box_logic, fontsize=10, fontweight='bold', color="#1B5E20")

    # External Server (Right)
    ax.text(10.0, 7.0, "SLEEPCARE CLOUD SYSTEM\n(WebSocket/Socket.IO Server,\nCơ sở dữ liệu y tế lâm sàng,\nKết nối bác sĩ và bệnh nhân)", ha='center', va='center', bbox=box_server, fontsize=9.5, fontweight='bold', color="#4A148C")
    ax.text(10.0, 3.0, "THIẾT BỊ HỖ TRỢ THỞ\nSIPAP\n(BLE Peripheral Device)", ha='center', va='center', bbox=box_hw, fontsize=9.5, fontweight='bold', color="#E65100")

    # Connections
    # HW to Logic
    draw_arrow(ax, 3.8, 7.5, 4.4, 5.8, "Audio Stream", color=border_hw, text_color="#E65100")
    draw_arrow(ax, 3.8, 5.0, 4.2, 5.0, "Lux / Image Stream", color=border_hw, text_color="#E65100")
    draw_arrow(ax, 3.8, 2.5, 4.4, 4.2, "BLE Data Stream", color=border_hw, text_color="#E65100")

    # UI and Logic
    draw_arrow(ax, 6.0, 7.7, 6.0, 6.2, "User Events", color=border_ui, text_color="#0D47A1")
    draw_arrow(ax, 6.0, 6.2, 6.0, 7.7, "States Update", color=border_logic, text_color="#1B5E20")

    # Logic to Server
    draw_arrow(ax, 7.8, 5.5, 8.4, 6.2, "JSON / WebSocket", color=border_logic, text_color="#1B5E20")
    # BLE to SIPAP
    draw_arrow(ax, 7.8, 4.5, 8.4, 3.5, "BLE Control (Write)", color=border_hw, text_color="#E65100")
    draw_arrow(ax, 8.4, 3.5, 7.8, 4.5, "Sensor Data (Notify)", color=border_hw, text_color="#E65100")

    plt.title("SƠ ĐỒ CẤU TRÚC PHẦN MỀM HỆ THỐNG SLEEPCARE", fontsize=13, fontweight='bold', color="#0D47A1", pad=15)
    
    out_dir = r"c:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\document"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "sleepcare_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_snore_algorithm():
    set_style()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Styles
    color_step = "#E8F5E9"
    border_step = "#43A047"
    box_step = dict(boxstyle="round,pad=0.4", fc=color_step, ec=border_step, lw=2)

    # Nodes
    ax.text(5.0, 9.0, "BẮT ĐẦU\n(Khởi tạo AudioRecorder - 16kHz PCM 16-bit Mono)", ha='center', va='center', bbox=box_step, fontsize=10, fontweight='bold', color="#1B5E20")
    ax.text(5.0, 7.5, "TRÍCH XUẤT AUDIO STREAM\n(Ghép 2 byte thô thành mẫu 16-bit bù hai)", ha='center', va='center', bbox=box_step, fontsize=10, fontweight='bold', color="#1B5E20")
    ax.text(5.0, 6.0, "CHUẨN HÓA MẢNG MẪU\n(Chia cho 32768.0 để thu về khoảng [-1.0, 1.0])", ha='center', va='center', bbox=box_step, fontsize=10, fontweight='bold', color="#1B5E20")
    
    # Decisions / Splits
    ax.text(2.5, 4.5, "MÔ HÌNH AI (TFLite)\n(Xác nhận có tiếng ngáy)", ha='center', va='center', bbox=dict(boxstyle="round4,pad=0.3", fc="#E3F2FD", ec="#1E88E5", lw=2), fontsize=9, fontweight='bold', color="#0D47A1")
    ax.text(7.5, 4.5, "QUY ĐỔI BIÊN ĐỘ\n(Tính decibel dB thực tế\ntừ âm lượng dBFS)", ha='center', va='center', bbox=box_step, fontsize=9, fontweight='bold', color="#1B5E20")

    ax.text(5.0, 2.8, "KIỂM TRA TIẾNG NGÁY BỆNH LÝ\n(Có tiếng ngáy & Cường độ > 56 dB\nliên tục trong hơn 10 giây?)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3E0", ec="#FB8C00", lw=2), fontsize=10, fontweight='bold', color="#E65100")
    ax.text(5.0, 1.0, "KẾT QUẢ\n(Tăng biến đếm, cập nhật biểu đồ,\ntuyền dữ liệu lên Cloud SleepCare)", ha='center', va='center', bbox=box_step, fontsize=10, fontweight='bold', color="#1B5E20")

    # Arrows
    draw_arrow(ax, 5.0, 8.5, 5.0, 8.0, color=border_step)
    draw_arrow(ax, 5.0, 7.0, 5.0, 6.5, color=border_step)
    
    draw_arrow(ax, 5.0, 5.5, 3.2, 4.8, color=border_step)
    draw_arrow(ax, 5.0, 5.5, 6.8, 4.8, color=border_step)

    draw_arrow(ax, 2.5, 4.0, 4.4, 3.2, color="#1E88E5")
    draw_arrow(ax, 7.5, 4.0, 5.6, 3.2, color=border_step)

    draw_arrow(ax, 5.0, 2.2, 5.0, 1.5, color="#FB8C00")

    plt.title("LƯU ĐỒ THUẬT TOÁN PHÁT HIỆN TIẾNG NGÁY BỆNH LÝ TRÊN SLEEPCARE", fontsize=12, fontweight='bold', color="#0D47A1", pad=15)
    plt.savefig(os.path.join(out_dir, "snore_detection_algorithm.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_room_quality():
    set_style()
    fig, ax = plt.subplots(figsize=(10, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Styles
    color_box = "#FFF3E0"
    border_box = "#FB8C00"
    box_style = dict(boxstyle="round,pad=0.4", fc=color_box, ec=border_box, lw=2)

    # Nodes
    ax.text(5.0, 9.0, "ĐÁNH GIÁ CHẤT LƯỢNG PHÒNG NGỦ (Room Quality)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#E3F2FD", ec="#1E88E5", lw=2.5), fontsize=11, fontweight='bold', color="#0D47A1")

    # Light branch
    ax.text(2.5, 7.0, "THEO DÕI ÁNH SÁNG", ha='center', va='center', bbox=box_style, fontsize=9.5, fontweight='bold', color="#E65100")
    ax.text(2.5, 5.2, "Android: Lux Stream từ cảm biến ánh sáng\niOS: Phân tích Camera Image Stream\n(Lux = (B / 255.0) * 200)", ha='center', va='center', bbox=box_style, fontsize=8.5, color="#E65100")
    ax.text(2.5, 3.0, "Mức tối ưu: < 5 Lux (Phòng tối)\n(Đảm bảo Melatonin tổng hợp sinh lý)", ha='center', va='center', bbox=box_style, fontsize=8.5, color="#E65100")

    # Noise branch
    ax.text(7.5, 7.0, "THEO DÕI ĐỘ ỒN NỀN", ha='center', va='center', bbox=box_style, fontsize=9.5, fontweight='bold', color="#E65100")
    ax.text(7.5, 5.2, "Microphone ghi nhận decibel (dB)\nmỗi 1 giây từ Audio Amplitude", ha='center', va='center', bbox=box_style, fontsize=8.5, color="#E65100")
    ax.text(7.5, 3.0, "Mức tối ưu: < 35 dB (Phòng yên tĩnh)\n(Tránh kích thích vỏ não gây thức giấc)", ha='center', va='center', bbox=box_style, fontsize=8.5, color="#E65100")

    ax.text(5.0, 1.2, "ĐÁNH GIÁ CHUNG VỀ SỰ PHÙ HỢP CỦA PHÒNG NGỦ", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#43A047", lw=2), fontsize=10, fontweight='bold', color="#1B5E20")

    # Arrows
    draw_arrow(ax, 3.8, 8.5, 2.5, 7.5, color="#1E88E5")
    draw_arrow(ax, 6.2, 8.5, 7.5, 7.5, color="#1E88E5")

    draw_arrow(ax, 2.5, 6.5, 2.5, 6.0, color=border_box)
    draw_arrow(ax, 2.5, 4.4, 2.5, 3.8, color=border_box)

    draw_arrow(ax, 7.5, 6.5, 7.5, 6.0, color=border_box)
    draw_arrow(ax, 7.5, 4.4, 7.5, 3.8, color=border_box)

    draw_arrow(ax, 2.5, 2.2, 4.2, 1.5, color="#43A047")
    draw_arrow(ax, 7.5, 2.2, 5.8, 1.5, color="#43A047")

    plt.title("SƠ ĐỒ QUY TRÌNH GIÁM SÁT CHẤT LƯỢNG MÔI TRƯỜNG PHÒNG NGỦ", fontsize=12, fontweight='bold', color="#0D47A1", pad=15)
    plt.savefig(os.path.join(out_dir, "room_quality_algorithm.png"), dpi=300, bbox_inches='tight')
    plt.close()

def draw_screening_model():
    set_style()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle="round,pad=0.4", fc="#F3E5F5", ec="#8E24AA", lw=2)

    # Nodes
    ax.text(5.0, 9.0, "QUY TRÌNH SÀNG LỌC VÀ ĐÁNH GIÁ RỦI RO NGƯNG THỞ KHI NGỦ", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", fc="#E3F2FD", ec="#1E88E5", lw=2.5), fontsize=11, fontweight='bold', color="#0D47A1")

    # Inputs
    ax.text(1.8, 7.0, "Bước 1:\nBảng hỏi chung,\nHuyết áp & Chỉ số BMI", ha='center', va='center', bbox=box_style, fontsize=8.5, fontweight='bold', color="#4A148C")
    ax.text(5.0, 7.0, "Bước 2:\nKhảo sát buồn ngủ ban ngày\nThang đo Epworth (ESS)", ha='center', va='center', bbox=box_style, fontsize=8.5, fontweight='bold', color="#4A148C")
    ax.text(8.2, 7.0, "Bước 3:\nĐánh giá rủi ro ngưng thở\nBộ câu hỏi STOP-BANG", ha='center', va='center', bbox=box_style, fontsize=8.5, fontweight='bold', color="#4A148C")

    # Mid Calculation
    ax.text(5.0, 5.0, "TÍNH TOÁN TỔNG HỢP\n(Tính tổng điểm kết hợp & Phân loại nguy cơ)", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#43A047", lw=2), fontsize=10, fontweight='bold', color="#1B5E20")

    # Risk Levels
    ax.text(2.0, 3.0, "RỦI RO THẤP\n(Tổng điểm < 15)\n-> Khuyên duy trì thói quen\nvệ sinh giấc ngủ tốt", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#43A047", lw=1.5), fontsize=8.5, color="#1B5E20")
    ax.text(5.0, 3.0, "RỦI RO TRUNG BÌNH\n(Tổng điểm 15 - 20)\n-> Khuyên dùng Snore Detection\ntrên app để tự đo tiếng ngáy", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FB8C00", lw=1.5), fontsize=8.5, color="#E65100")
    ax.text(8.0, 3.0, "RỦI RO CAO\n(Tổng điểm > 20)\n-> Khuyên khám chuyên khoa\nvà đăng ký đo PSG tại nhà", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#E53935", lw=1.5), fontsize=8.5, color="#C62828")

    # Clinic Finder linkage at bottom
    ax.text(5.0, 1.0, "KẾT NỐI VỚI HỆ THỐNG PHÒNG KHÁM CHUYÊN KHOA VSSM & ĐO ĐA KÝ GIẤC NGỦ (PSG) TẠI NHÀ", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1E88E5", lw=2), fontsize=9.5, fontweight='bold', color="#0D47A1")

    # Arrows
    draw_arrow(ax, 1.8, 6.2, 4.0, 5.3, color="#8E24AA")
    draw_arrow(ax, 5.0, 6.2, 5.0, 5.6, color="#8E24AA")
    draw_arrow(ax, 8.2, 6.2, 6.0, 5.3, color="#8E24AA")

    draw_arrow(ax, 4.0, 4.4, 2.5, 3.8, color="#43A047")
    draw_arrow(ax, 5.0, 4.4, 5.0, 3.8, color="#43A047")
    draw_arrow(ax, 6.0, 4.4, 7.5, 3.8, color="#43A047")

    draw_arrow(ax, 2.0, 2.0, 4.0, 1.3, color="#1E88E5")
    draw_arrow(ax, 5.0, 2.2, 5.0, 1.5, color="#1E88E5")
    draw_arrow(ax, 8.0, 2.0, 6.0, 1.3, color="#1E88E5")

    plt.title("SƠ ĐỒ HỆ THỐNG TẦM SOÁT LÂM SÀNG VÀ CHỈ DẪN Y KHOA", fontsize=12, fontweight='bold', color="#0D47A1", pad=15)
    plt.savefig(os.path.join(out_dir, "screening_risk_model.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    out_dir = r"c:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\document"
    print("Generating diagrams in:", out_dir)
    draw_architecture()
    draw_snore_algorithm()
    draw_room_quality()
    draw_screening_model()
    print("All diagrams generated successfully!")
