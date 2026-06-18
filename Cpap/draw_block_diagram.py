import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_block_diagram():
    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette
    color_mcu = "#E3F2FD"      # Soft Blue
    color_input = "#E8F5E9"    # Soft Green
    color_output = "#FFF3E0"   # Soft Orange
    color_comm = "#F3E5F5"     # Soft Purple
    color_cloud = "#ECEFF1"    # Soft Gray
    
    border_mcu = "#1E88E5"
    border_input = "#43A047"
    border_output = "#FB8C00"
    border_comm = "#8E24AA"
    border_cloud = "#607D8B"

    # Define Box Styles
    box_mcu = dict(boxstyle="round,pad=0.5", fc=color_mcu, ec=border_mcu, lw=2.5)
    box_input = dict(boxstyle="round,pad=0.4", fc=color_input, ec=border_input, lw=2)
    box_output = dict(boxstyle="round,pad=0.4", fc=color_output, ec=border_output, lw=2)
    box_comm = dict(boxstyle="round,pad=0.4", fc=color_comm, ec=border_comm, lw=2)
    box_cloud = dict(boxstyle="round,pad=0.5", fc=color_cloud, ec=border_cloud, lw=2.5)

    # 1. Place Nodes
    # MCU (Center)
    ax.text(6.0, 5.0, "BỘ XỬ LÝ TRUNG TÂM\n\nArduino Nano 33 BLE Sense\n(ARM Cortex-M4 32-bit)", 
            ha='center', va='center', bbox=box_mcu, fontsize=11, fontweight='bold', color="#0D47A1")

    # Inputs (Left Side)
    ax.text(2.0, 7.5, "CẢM BIẾN LƯU LƯỢNG\nSensirion SFM3300-D\n(Đo dòng khí thở tức thời)", 
            ha='center', va='center', bbox=box_input, fontsize=9.5, fontweight='bold', color="#1B5E20")
    
    ax.text(2.0, 5.0, "CẢM BIẾN ÁP SUẤT\nMPXV5010G\n(Đo áp suất mặt nạ)", 
            ha='center', va='center', bbox=box_input, fontsize=9.5, fontweight='bold', color="#1B5E20")
    
    ax.text(2.0, 2.5, "ROTARY ENCODER\n& Nút nhấn\n(Cài đặt thông số tại chỗ)", 
            ha='center', va='center', bbox=box_input, fontsize=9.5, fontweight='bold', color="#1B5E20")

    # Outputs (Right Side)
    ax.text(10.0, 7.5, "MÀN HÌNH OLED 0.96\"\n(Hiển thị áp suất, lưu lượng,\nchế độ thở và nhịp thở)", 
            ha='center', va='center', bbox=box_output, fontsize=9.5, fontweight='bold', color="#E65100")
    
    ax.text(10.0, 5.0, "ĐỘNG CƠ QUẠT THỔI\nBrushless Blower WS4540\n+ Mạch lái Driver PWM", 
            ha='center', va='center', bbox=box_output, fontsize=9.5, fontweight='bold', color="#E65100")
    
    ax.text(10.0, 2.5, "LED CHỈ THỊ trạng thái\n& Còi báo động (Buzzer)", 
            ha='center', va='center', bbox=box_output, fontsize=9.5, fontweight='bold', color="#E65100")

    # Connectivity & Cloud (Bottom / Top)
    ax.text(6.0, 2.2, "KẾT NỐI KHÔNG DÂY\nBluetooth Low Energy (BLE)\n(Gói dữ liệu JSON mỗi 200ms)", 
            ha='center', va='center', bbox=box_comm, fontsize=9.5, fontweight='bold', color="#4A148C")
    
    ax.text(6.0, 0.6, "ỨNG DỤNG DI ĐỘNG (Smartphone App)\n+ HỆ THỐNG GIÁM SÁT CLOUD (SleepCare)", 
            ha='center', va='center', bbox=box_cloud, fontsize=10.5, fontweight='bold', color="#263238")

    # 2. Draw Connections (Arrows)
    def draw_arrow(x1, y1, x2, y2, label="", color="gray", style="-", label_pos=0.5, text_color="black"):
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=15, 
                                        color=color, lw=1.8, linestyle=style)
        ax.add_patch(arrow)
        if label:
            x_text = x1 + (x2 - x1) * label_pos
            y_text = y1 + (y2 - y1) * label_pos
            ax.text(x_text, y_text, label, ha='center', va='center', 
                    fontsize=8, fontweight='bold', color=text_color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # Inputs to MCU
    draw_arrow(3.8, 7.5, 4.4, 5.8, "I2C Bus (SDA/SCL)", color=border_input, text_color="#1B5E20")
    draw_arrow(3.8, 5.0, 4.2, 5.0, "Analog (0-3.3V)\n(Qua cầu phân áp)", color=border_input, text_color="#1B5E20")
    draw_arrow(3.8, 2.5, 4.4, 4.2, "Digital GPIOs", color=border_input, text_color="#1B5E20")

    # MCU to Outputs
    draw_arrow(7.6, 5.8, 8.2, 7.5, "I2C Bus", color=border_output, text_color="#E65100")
    
    # Bi-directional / dual arrows for Blower Control & Feedback
    # PWM Control output
    draw_arrow(7.8, 5.1, 8.2, 5.1, "PWM Control (D3)", color=border_output, text_color="#E65100")
    # Feedback RPM input (use dashed arrow or reverse arrow)
    draw_arrow(8.2, 4.9, 7.8, 4.9, "RPM FG (D2 - Ngắt)", color="#D84315", text_color="#D84315")
    
    draw_arrow(7.6, 4.2, 8.2, 2.5, "Digital GPIOs", color=border_output, text_color="#E65100")

    # MCU to BLE
    draw_arrow(6.0, 3.8, 6.0, 3.0, "Tích hợp sẵn\ntrên Board", color=border_comm, text_color="#4A148C")
    
    # BLE to App & Cloud
    draw_arrow(6.0, 1.4, 6.0, 1.0, "Truyền dữ liệu không dây", color=border_cloud, text_color="#263238")

    # Title
    plt.title("SƠ ĐỒ KHỐI KẾT NỐI VÀ GIAO TIẾP HỆ THỐNG SIPAP", fontsize=14, fontweight='bold', color="#0D47A1", pad=20)
    
    # Save the plot
    out_dir = r"c:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\document"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "system_block_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"System block diagram saved successfully at {out_path}!")

if __name__ == "__main__":
    draw_block_diagram()
