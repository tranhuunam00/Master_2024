import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_schematic():
    """
    Sơ đồ kết nối phần cứng SIPAP v1.
    Layout rộng hơn, tránh chồng chéo nhãn và mũi tên.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # ===== COLOR PALETTE =====
    box_mcu     = dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1E88E5", lw=2.5)
    box_sensor  = dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#43A047", lw=2)
    box_power   = dict(boxstyle="round,pad=0.4", fc="#FFEBEE", ec="#E53935", lw=2)
    box_blower  = dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#FB8C00", lw=2)
    box_divider = dict(boxstyle="round,pad=0.3", fc="#F3E5F5", ec="#8E24AA", lw=1.5)
    box_oled    = dict(boxstyle="round,pad=0.3", fc="#E0F7FA", ec="#00838F", lw=2)

    # ===== NODE POSITIONS (x, y) =====
    # Row 1 (top):    Power 12V,                    Flow Sensor
    # Row 2:          Buck Converter,                (I2C lines down)
    # Row 3 (center): Pressure Sensor,  MCU,         Blower Driver
    # Row 4:          Voltage Divider,               Blower Motor
    # Row 5 (bottom): OLED display

    pos_power   = (2.0, 9.0)
    pos_buck    = (2.0, 7.0)
    pos_flow    = (7.0, 9.0)
    pos_mcu     = (7.0, 5.5)
    pos_press   = (2.0, 5.0)
    pos_divider = (2.0, 3.0)
    pos_driver  = (12.0, 5.5)
    pos_blower  = (12.0, 3.0)
    pos_oled    = (7.0, 2.0)

    # ===== DRAW COMPONENT BOXES =====
    ax.text(*pos_power, "NGUỒN CẤP\n12VDC Adapter",
            ha='center', va='center', bbox=box_power, fontsize=10, fontweight='bold')

    ax.text(*pos_buck, "MẠCH HẠ ÁP BUCK\n(12V → 5V / 3.3V)",
            ha='center', va='center', bbox=box_power, fontsize=10, fontweight='bold')

    ax.text(*pos_flow, "CẢM BIẾN LƯU LƯỢNG\nSensirion SFM3300-D\n(Giao tiếp I2C · Nguồn: 5V)",
            ha='center', va='center', bbox=box_sensor, fontsize=10, fontweight='bold')

    ax.text(*pos_mcu, "VI ĐIỀU KHIỂN TRUNG TÂM\nArduino Nano 33 BLE Sense\n(ARM Cortex-M4 · 3.3V logic)",
            ha='center', va='center', bbox=box_mcu, fontsize=11, fontweight='bold')

    ax.text(*pos_press, "CẢM BIẾN ÁP SUẤT\nMPXV5010G\n(Nguồn cấp: 5V)",
            ha='center', va='center', bbox=box_sensor, fontsize=10, fontweight='bold')

    ax.text(*pos_divider, "CẦU PHÂN ÁP\nR1=10kΩ · R2=20kΩ\n(5V → 3.3V cho ADC)",
            ha='center', va='center', bbox=box_divider, fontsize=9, fontweight='bold')

    ax.text(*pos_driver, "MẠCH LÁI ĐỘNG CƠ\n(Driver Blower PWM)",
            ha='center', va='center', bbox=box_blower, fontsize=10, fontweight='bold')

    ax.text(*pos_blower, "ĐỘNG CƠ QUẠT THỔI\nBrushless Blower\nWS4540-12-NZ03",
            ha='center', va='center', bbox=box_blower, fontsize=10, fontweight='bold')

    ax.text(*pos_oled, "MÀN HÌNH OLED 0.96\"\n& Rotary Encoder",
            ha='center', va='center', bbox=box_oled, fontsize=10, fontweight='bold')

    # ===== DRAW ARROWS =====
    def draw_arrow(x1, y1, x2, y2, color="gray", lw=1.5, style="-", 
                   conn_style="arc3,rad=0.0"):
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=16,
            color=color, lw=lw, linestyle=style,
            connectionstyle=conn_style
        )
        ax.add_patch(arrow)

    def add_label(x, y, text, color="black", fontsize=8, ha='center', va='center',
                  bg="white", alpha=0.85):
        ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, fontweight='bold',
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec="none", alpha=alpha))

    # ──────────────────────────────────────────────
    # POWER LINES (Red)
    # ──────────────────────────────────────────────

    # 12V → Buck Converter (straight down)
    draw_arrow(2.0, 8.5, 2.0, 7.5, color="red", lw=2)
    add_label(1.5, 8.0, "12VDC", color="red")

    # 12V → Blower Driver (routed along the top then down-right)
    draw_arrow(3.2, 9.0, 12.0, 6.2, color="red", lw=2, conn_style="arc3,rad=-0.15")
    add_label(8.5, 8.2, "12VDC", color="red")

    # Buck 5V → Pressure Sensor (straight down)
    draw_arrow(2.0, 6.5, 2.0, 5.6, color="red", lw=1.5)
    add_label(1.4, 6.0, "5VDC", color="red")

    # Buck 5V → Arduino Vin (right)
    draw_arrow(3.2, 7.0, 5.5, 5.9, color="red", lw=1.5, conn_style="arc3,rad=0.1")
    add_label(4.0, 6.8, "5VDC (Vin)", color="red")

    # Buck 5V → Flow Sensor (up along left side of flow sensor)
    draw_arrow(3.2, 7.2, 5.5, 8.8, color="red", lw=1.5, conn_style="arc3,rad=0.15")
    add_label(3.8, 8.2, "5VDC", color="red")

    # ──────────────────────────────────────────────
    # SIGNAL LINES - Pressure path (Blue)
    # ──────────────────────────────────────────────

    # Pressure Sensor Vout → Voltage Divider
    draw_arrow(2.0, 4.4, 2.0, 3.6, color="#1565C0", lw=1.5)
    add_label(1.2, 3.9, "Vout\n(0~5V)", color="#1565C0", fontsize=7)

    # Voltage Divider → Arduino A0
    draw_arrow(3.2, 3.0, 5.5, 5.0, color="#1565C0", lw=1.5, conn_style="arc3,rad=0.1")
    add_label(3.8, 3.8, "V_adc (0~3.3V)\n→ Pin A0", color="#1565C0", fontsize=7)

    # ──────────────────────────────────────────────
    # SIGNAL LINES - Flow Sensor I2C (Green)
    # ──────────────────────────────────────────────

    # Single I2C bus line (combined SDA+SCL for clarity)
    draw_arrow(7.0, 8.3, 7.0, 6.2, color="#2E7D32", lw=2)
    add_label(7.8, 7.3, "I2C Bus\nSDA (A4) + SCL (A5)", color="#2E7D32", fontsize=8)

    # ──────────────────────────────────────────────
    # SIGNAL LINES - Blower control (Purple / Brown)
    # ──────────────────────────────────────────────

    # Arduino D3 → Blower Driver (PWM Control)
    draw_arrow(8.5, 5.7, 10.8, 5.7, color="#7B1FA2", lw=1.5)
    add_label(9.6, 6.1, "PWM Control\n(Pin D3)", color="#7B1FA2", fontsize=8)

    # Blower Driver → Arduino D2 (RPM FG Feedback) - return arrow slightly below
    draw_arrow(10.8, 5.3, 8.5, 5.3, color="#BF360C", lw=1.5)
    add_label(9.6, 4.9, "RPM FG Feedback\n(Pin D2 · Ngắt ngoài)", color="#BF360C", fontsize=8)

    # Blower Driver → Blower Motor
    draw_arrow(12.0, 4.9, 12.0, 3.6, color="#E65100", lw=2)
    add_label(12.8, 4.2, "3-Phase\nU/V/W", color="#E65100", fontsize=8)

    # ──────────────────────────────────────────────
    # SIGNAL LINES - OLED & Encoder (Teal)
    # ──────────────────────────────────────────────

    # Arduino → OLED (I2C or SPI)
    draw_arrow(7.0, 4.8, 7.0, 2.6, color="#00695C", lw=1.5)
    add_label(7.8, 3.5, "I2C / GPIO\n(Hiển thị & Cài đặt)", color="#00695C", fontsize=8)

    # ──────────────────────────────────────────────
    # GND NOTE
    # ──────────────────────────────────────────────
    ax.text(7.0, 0.5,
            "⚡ LƯU Ý: TẤT CẢ CÁC THÀNH PHẦN PHẢI NỐI CHUNG ĐẤT (COMMON GND) VỚI ARDUINO",
            ha='center', va='center', color='#B71C1C', fontsize=10,
            style='italic', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#E53935", lw=1.5, alpha=0.9))

    # ===== TITLE =====
    plt.title("SƠ ĐỒ KẾT NỐI HỆ THỐNG PHẦN CỨNG SIPAP (VERSION 1)",
              fontsize=14, fontweight='bold', color='#1B365D', pad=18)

    # ===== SAVE =====
    out_dir = r"c:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\document"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "schematic.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Schematic saved successfully at {out_path}!")

if __name__ == "__main__":
    draw_schematic()
