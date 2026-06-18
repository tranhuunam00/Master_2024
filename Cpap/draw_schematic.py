import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_schematic():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    
    # Define styles
    box_style_mcu = dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#1E88E5", lw=2)      # Light Blue
    box_style_sensor = dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#43A047", lw=2)   # Light Green
    box_style_power = dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#E53935", lw=2)    # Light Red
    box_style_blower = dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#FB8C00", lw=2)   # Light Orange
    box_style_divider = dict(boxstyle="round,pad=0.3", fc="#F3E5F5", ec="#8E24AA", lw=1.5) # Light Purple

    # 1. Draw Boxes
    # Power Input
    ax.text(1.5, 7.5, "NGUỒN CẤP\n12VDC", ha='center', va='center', bbox=box_style_power, fontsize=10, fontweight='bold')
    
    # Buck Converter
    ax.text(1.5, 5.5, "MẠCH HẠ ÁP BUCK\n(12V -> 5V)", ha='center', va='center', bbox=box_style_power, fontsize=10, fontweight='bold')
    
    # Arduino MCU
    ax.text(5.5, 4.5, "VI ĐIỀU KHIỂN TRUNG TÂM\nArduino Nano 33 BLE Sense\n(Điện áp hoạt động: 3.3V)", ha='center', va='center', bbox=box_style_mcu, fontsize=11, fontweight='bold')
    
    # MPXV5010G Pressure Sensor
    ax.text(1.5, 3.2, "CẢM BIẾN ÁP SUẤT\nMPXV5010G\n(Nguồn cấp: 5V)", ha='center', va='center', bbox=box_style_sensor, fontsize=10, fontweight='bold')
    
    # Voltage Divider Resistors
    ax.text(1.5, 1.2, "CẦU PHÂN ÁP\nR1 = 10k, R2 = 20k\n(Giảm áp 1.5 lần)", ha='center', va='center', bbox=box_style_divider, fontsize=9)
    
    # SFM3300-D Flow Sensor
    ax.text(5.5, 7.5, "CẢM BIẾN LƯU LƯỢNG\nSensirion SFM3300-D\n(Giao tiếp I2C - Nguồn: 5V)", ha='center', va='center', bbox=box_style_sensor, fontsize=10, fontweight='bold')
    
    # Blower Driver
    ax.text(8.5, 3.2, "MẠCH LÁI ĐỘNG CƠ\n(Driver Blower PWM)", ha='center', va='center', bbox=box_style_blower, fontsize=10, fontweight='bold')
    
    # WS4540 Blower Motor
    ax.text(8.5, 1.2, "ĐỘNG CƠ QUẠT THỔI\nBrushless Blower\nWS4540-12-NZ03", ha='center', va='center', bbox=box_style_blower, fontsize=10, fontweight='bold')

    # 2. Draw Connections (Arrows)
    def draw_arrow(x1, y1, x2, y2, text="", color="black", linestyle="-", text_offset_y=0.1, text_offset_x=0.0):
        # Draw line with arrow
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15, color=color, lw=1.5, linestyle=linestyle)
        ax.add_patch(arrow)
        # Add text
        if text:
            ax.text((x1 + x2)/2 + text_offset_x, (y1 + y2)/2 + text_offset_y, text, ha='center', va='center', color=color, fontsize=8, fontweight='bold')

    # Power lines
    draw_arrow(1.5, 7.1, 1.5, 5.9, "12VDC", "red")
    draw_arrow(2.2, 7.5, 8.5, 3.6, "12VDC", "red") # 12V to Blower Driver
    draw_arrow(1.5, 5.1, 1.5, 3.7, "5VDC", "red")  # 5V to Pressure Sensor
    # 5V to Arduino Vin
    draw_arrow(2.2, 5.5, 3.7, 4.8, "5VDC (Vin)", "red")
    # 5V to Flow Sensor
    draw_arrow(5.5, 5.2, 5.5, 7.1, "5VDC", "red")

    # Signal lines
    # Pressure Sensor output to Divider
    draw_arrow(1.5, 2.7, 1.5, 1.7, "Vout (0-5V)", "blue")
    # Divider to Arduino A0
    draw_arrow(2.4, 1.2, 4.2, 4.0, "V_adc (0-3.3V) -> Pin A0", "blue")
    
    # Flow Sensor I2C to Arduino
    # Bidirectional I2C
    draw_arrow(5.2, 7.1, 5.2, 5.2, "I2C SDA (A4)", "green")
    draw_arrow(5.8, 7.1, 5.8, 5.2, "I2C SCL (A5)", "green")

    # Blower Driver to Motor
    draw_arrow(8.5, 2.8, 8.5, 1.7, "3-Phase U/V/W", "orange")
    
    # Arduino D3 to Blower Driver (PWM Control)
    draw_arrow(6.7, 4.2, 7.8, 3.5, "PWM Ctrl -> D3", "purple")
    # Blower Driver to Arduino D2 (RPM FG Feedback)
    draw_arrow(7.8, 3.1, 6.7, 3.8, "RPM FG -> D2 (Ngắt)", "brown")

    # GND Line
    ax.text(5.0, 0.2, "* LƯU Ý: TẤT CẢ CÁC THÀNH PHẦN PHẢI NỐI CHUNG ĐẤT (COMMON GND) VỚI ARDUINO", ha='center', va='center', color='black', fontsize=9, style='italic', fontweight='bold')

    # Set Title
    plt.title("SƠ ĐỒ KẾT NỐI HỆ THỐNG PHẦN CỨNG SIPAP (VERSION 1)", fontsize=13, fontweight='bold', color='#1B365D', pad=15)
    
    # Save figure
    out_dir = r"c:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\document"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "schematic.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Schematic saved successfully at {out_path}!")

if __name__ == "__main__":
    draw_schematic()
