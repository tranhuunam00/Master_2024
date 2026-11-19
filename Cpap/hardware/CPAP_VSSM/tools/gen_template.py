"""Generate the sch_draw_template JSON for the SiPAP main sheet (rev B).

Every pin gets a short stub wire ending in a net label or a power port,
so the sheet reads block by block and the netlist is explicit.
"""
import json, os

LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SiPAP_Parts_v2.SchLib")

# symbol pin tables: name -> {pin: (x, y, side, length)} side L/R/D
def two(len_=100):
    return {"1": (0, 2900, "L", len_), "2": (400, 2900, "R", len_)}

SYM = {
    "RESISTOR": two(), "CAPACITOR": two(), "DIODE": two(), "BUZZER": two(200),
    "CONN_DCJACK": {"1": (0, 2900, "L", 200), "2": (400, 2900, "R", 200)},
    "AO3401A": {"1": (0, 2900, "L", 200), "2": (200, 2600, "D", 200), "3": (400, 2900, "R", 200)},
    "CONN_BLOWER4": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "3": (400, 2900, "R", 200), "4": (400, 2800, "R", 200)},
    "BUCK_MODULE_12to5": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "3": (600, 2900, "R", 200), "4": (600, 2800, "R", 200)},
    "MPXV5010GC6U": {"1": (0, 3000, "L", 200), "5": (0, 2900, "L", 200), "6": (0, 2800, "L", 200), "7": (0, 2700, "L", 200), "8": (0, 2600, "L", 200),
                     "2": (600, 3000, "R", 200), "3": (600, 2900, "R", 200), "4": (600, 2800, "R", 200)},
    "SFM3300-D": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "4": (0, 2700, "L", 200), "3": (600, 2900, "R", 200), "5": (600, 2800, "R", 200), "6": (600, 2700, "R", 200)},
    "CONN_UART6": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "4": (0, 2700, "L", 200), "3": (600, 2900, "R", 200), "5": (600, 2800, "R", 200), "6": (600, 2700, "R", 200)},
    "SSD1306_OLED_I2C": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "3": (600, 2900, "R", 200), "4": (600, 2800, "R", 200)},
    "PEC11R-4220F-S0024": {"1": (0, 2900, "L", 200), "3": (0, 2800, "L", 200), "2": (0, 2700, "L", 200), "4": (600, 2900, "R", 200), "5": (600, 2800, "R", 200)},
    "LM393": {"2": (0, 3300, "L", 200), "3": (0, 3200, "L", 200), "4": (0, 3100, "L", 200), "5": (0, 3000, "L", 200), "6": (0, 2900, "L", 200),
              "1": (600, 3400, "R", 200), "7": (600, 3300, "R", 200), "8": (600, 3200, "R", 200)},
}
esp = {}
for i in range(1, 22):
    esp[str(i)] = (0, 4100 - (i - 1) * 100, "L", 200)
for i in range(22, 42):
    esp[str(i)] = (1300, 4100 - (i - 22) * 100, "R", 200)
SYM["ESP32-S3-WROOM-1"] = esp

POWER = {"GND", "3V3", "5V", "VIN12V"}

parts, wires, ports, labels = [], [], [], []
nets = {}


def place(des, sym, x, y, value, conns, footprint=None):
    """x, y: where the symbol body row y=2900 should land (body origin offset removed)."""
    ox, oy = x, y - 2900
    parts.append({"designator": des, "lib_reference": sym, "library_path": LIB, "x": ox, "y": oy, "rotation": 0, "value": value,
                  **({"footprint": footprint} if footprint else {})})
    for pin, net in conns.items():
        if net is None:
            continue
        px, py, side, ln = SYM[sym][pin]
        ax, ay = ox + px, oy + py
        if net in POWER:
            stub = 200 + 300 * ((py // 100) % 2)   # stagger stacked power glyphs
        else:
            stub = 800 if side == "L" else 200     # room for the label text left of the body
        if side == "L":
            cx, cy = ax - ln, ay; ex, ey = cx - stub, cy
        elif side == "R":
            cx, cy = ax + ln, ay; ex, ey = cx + stub, cy
        else:
            cx, cy = ax, ay - ln; ex, ey = cx, cy - stub
        wires.append({"net": net, "path": [f"{des}.{pin}", [ex, ey]]})
        if net in POWER:
            if net == "GND":
                ports.append({"net": net, "at": [ex, ey], "style": "gnd_power"})
            else:
                ports.append({"net": net, "at": [ex, ey], "style": "bar"})
        else:
            labels.append({"net": net, "at": [ex, ey]})
        nets.setdefault(net, []).append(f"{des}.{pin}")


R, C = "RESISTOR", "CAPACITOR"
two_p = lambda a, b: {"1": a, "2": b}

# ---------------- POWER (left column) ----------------
X0 = 1500
place("J1", "CONN_DCJACK", X0, 10600, "DC Jack 12V", {"1": "VIN_RAW", "2": "GND"})
place("D1", "DIODE", X0, 10000, "SS54 (5A 40V Schottky)", {"1": "VIN_RAW", "2": "VIN12V"})
place("C10", C, X0, 9400, "10uF 25V", two_p("VIN12V", "GND"))
place("U8", "BUCK_MODULE_12to5", X0, 8700, "Buck 12V->5V", {"1": "VIN12V", "2": "GND", "3": "5V", "4": "GND"})
place("C2", C, X0, 7900, "10uF 10V", two_p("5V", "GND"))
place("U9", "BUCK_MODULE_12to5", X0, 7200, "Buck 5V->3V3", {"1": "5V", "2": "GND", "3": "3V3", "4": "GND"})
place("C1", C, X0, 6400, "10uF 10V", two_p("3V3", "GND"))

# ---------------- BLOWER + SAFETY CUTOFF (left, lower) ----------------
place("Q1", "AO3401A", X0, 5300, "AO3401A", {"1": "GATE", "2": "VIN12V", "3": "BLOWER_V12"})
place("R13", R, X0, 4400, "10k", two_p("GATE", "VIN12V"))
place("R17", R, X0, 3800, "10k", two_p("GATE", "COMP_OUT"))
place("J2", "CONN_BLOWER4", X0, 3100, "Blower cable", {"1": "BLOWER_V12", "2": "GND", "3": "BLOWER_PWM", "4": "BLOWER_FG"})
place("R12", R, X0, 2300, "10k", two_p("3V3", "BLOWER_FG"))

# ---------------- SAFETY COMPARATOR (column 2) ----------------
X1 = 4600
place("U7", "LM393", X1, 10500, "LM393", {"2": "COMP_REF", "3": "PRESS_RED_ADC", "4": "GND", "5": "GND", "6": "GND",
                                           "1": "COMP_OUT", "7": None, "8": "5V"})
place("C4", C, X1, 9500, "100nF", two_p("5V", "GND"))
place("R10", R, X1, 8900, "47k", two_p("5V", "COMP_REF"))
place("R11", R, X1, 8300, "10k", two_p("COMP_REF", "GND"))
place("R18", R, X1, 7700, "1M", two_p("COMP_OUT", "PRESS_RED_ADC"))
place("R15", R, X1, 7100, "1M", two_p("COMP_OUT", "SAFETY_TRIP"))
place("R16", R, X1, 6500, "330k", two_p("SAFETY_TRIP", "GND"))

# ---------------- PRESSURE SENSORS (column 2, lower) ----------------
place("U2", "MPXV5010GC6U", X1, 5500, "MPXV5010GC6U (primary)", {"2": "5V", "3": "GND", "4": "PRESS_PRI_RAW"})
place("C6", C, X1, 4600, "1uF", two_p("5V", "GND"))
place("C7", C, X1, 4100, "10nF", two_p("5V", "GND"))
place("C8", C, X1, 3600, "470pF", two_p("PRESS_PRI_RAW", "GND"))
place("R1", R, X1, 3100, "10k", two_p("PRESS_PRI_RAW", "PRESS_PRI_ADC"))
place("R2", R, X1, 2600, "20k", two_p("PRESS_PRI_ADC", "GND"))

X2 = 7600
place("U3", "MPXV5010GC6U", X2, 5500, "MPXV5010GC6U (redundant)", {"2": "5V", "3": "GND", "4": "PRESS_RED_RAW"})
place("C9", C, X2, 4600, "1uF", two_p("5V", "GND"))
place("C11", C, X2, 4100, "10nF", two_p("5V", "GND"))
place("C12", C, X2, 3600, "470pF", two_p("PRESS_RED_RAW", "GND"))
place("R3", R, X2, 3100, "10k", two_p("PRESS_RED_RAW", "PRESS_RED_ADC"))
place("R4", R, X2, 2600, "20k", two_p("PRESS_RED_ADC", "GND"))

# ---------------- MCU (centre) ----------------
esp_nets = {"1": "GND", "2": "3V3", "3": "EN", "4": "PRESS_PRI_ADC", "5": "PRESS_RED_ADC", "6": "SAFETY_TRIP", "7": "BUZZER_CTRL",
            "10": "ENC_A", "11": "ENC_B", "12": "I2C_SDA", "15": "BLOWER_PWM", "17": "I2C_SCL",
            "23": "ENC_SW", "27": "BOOT", "36": "UART_RX", "37": "UART_TX", "38": "BLOWER_FG", "40": "GND", "41": "GND"}
place("U1", "ESP32-S3-WROOM-1", 8200, 9100, "ESP32-S3-WROOM-1-N8R8", esp_nets)
place("C3", C, X2, 7300, "100nF", two_p("3V3", "GND"))
place("R9", R, X2, 6800, "10k", two_p("3V3", "EN"))
place("C5", C, X2, 6300, "1uF", two_p("EN", "GND"))

# ---------------- UI + I/O (right column) ----------------
X3 = 11600
place("J3", "CONN_UART6", X3, 10500, "Prog UART", {"1": "GND", "2": "3V3", "4": "UART_RX", "3": "UART_TX", "5": "EN", "6": "BOOT"})
place("U4", "SFM3300-D", X3, 9600, "SFM3300-D", {"1": "GND", "2": "5V", "4": "GND", "3": "I2C_SCL", "5": "I2C_SDA", "6": None})
place("U5", "SSD1306_OLED_I2C", X3, 8700, "OLED 0.96in I2C", {"1": "GND", "2": "3V3", "3": "I2C_SCL", "4": "I2C_SDA"})
place("R5", R, X3, 8000, "4.7k", two_p("3V3", "I2C_SCL"))
place("R6", R, X3, 7500, "4.7k", two_p("3V3", "I2C_SDA"))
place("U6", "PEC11R-4220F-S0024", X3, 6700, "PEC11R-4220F-S0024", {"1": "ENC_A", "3": "GND", "2": "ENC_B", "4": "ENC_SW", "5": "GND"})
place("R7", R, X3, 5800, "10k", two_p("3V3", "ENC_A"))
place("R8", R, X3, 5300, "10k", two_p("3V3", "ENC_B"))
place("R14", R, X3, 4800, "10k", two_p("3V3", "ENC_SW"))
place("BZ1", "BUZZER", X3, 4100, "Piezo buzzer", two_p("BUZZER_CTRL", "GND"))

tpl = {"parts": parts, "wires": wires, "ports": ports, "labels": labels, "nets": nets}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sipap_template.json")
json.dump(tpl, open(out, "w"), indent=1)
print(out, len(parts), "parts", len(wires), "wires", len(nets), "nets")
