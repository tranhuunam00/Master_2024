"""SiPAP main sheet Rev C template: on-board TPS54202 bucks, input fuse + TVS,
real DC jack (PJ-102AH), AO4407A blower switch. Same stub+label style as Rev B.
Existing parts keep their UniqueId so the board stays linked."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "SiPAP_Parts_v2.SchLib")

def two(len_=100):
    return {"1": (0, 2900, "L", len_), "2": (400, 2900, "R", len_)}

# pin -> (x, y, side, length) in library coordinates; BASE is the row the symbol's pins sit on
SYM = {
    "RESISTOR": two(), "CAPACITOR": two(), "DIODE": two(), "BUZZER": two(200),
    "CONN_BLOWER4": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "3": (400, 2900, "R", 200), "4": (400, 2800, "R", 200)},
    "MPXV5010GC6U": {"1": (0, 3000, "L", 200), "5": (0, 2900, "L", 200), "6": (0, 2800, "L", 200), "7": (0, 2700, "L", 200), "8": (0, 2600, "L", 200),
                     "2": (600, 3000, "R", 200), "3": (600, 2900, "R", 200), "4": (600, 2800, "R", 200)},
    "SFM3300-D": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "4": (0, 2700, "L", 200), "3": (600, 2900, "R", 200), "5": (600, 2800, "R", 200), "6": (600, 2700, "R", 200)},
    "CONN_UART6": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "4": (0, 2700, "L", 200), "3": (600, 2900, "R", 200), "5": (600, 2800, "R", 200), "6": (600, 2700, "R", 200)},
    "SSD1306_OLED_I2C": {"1": (0, 2900, "L", 200), "2": (0, 2800, "L", 200), "3": (600, 2900, "R", 200), "4": (600, 2800, "R", 200)},
    "PEC11R-4220F-S0024": {"1": (0, 2900, "L", 200), "3": (0, 2800, "L", 200), "2": (0, 2700, "L", 200), "4": (600, 2900, "R", 200), "5": (600, 2800, "R", 200)},
    "LM393": {"2": (0, 3300, "L", 200), "3": (0, 3200, "L", 200), "4": (0, 3100, "L", 200), "5": (0, 3000, "L", 200), "6": (0, 2900, "L", 200),
              "1": (600, 3400, "R", 200), "7": (600, 3300, "R", 200), "8": (600, 3200, "R", 200)},
    # new Rev C symbols (pins on row 0)
    "TPS54202": {"3": (200, 0, "L", 200), "5": (200, -100, "L", 200), "1": (200, -300, "L", 200),
                 "6": (1000, 0, "R", 200), "2": (1000, -100, "R", 200), "4": (1000, -300, "R", 200)},
    "INDUCTOR": {"1": (100, 0, "L", 100), "2": (300, 0, "R", 100)},
    "FUSE_PTC": {"1": (100, 0, "L", 100), "2": (300, 0, "R", 100)},
    "TVS_UNI": {"2": (100, 0, "L", 100), "1": (300, 0, "R", 100)},
    "CONN_DCJACK3": {"1": (800, 0, "R", 200), "3": (800, -100, "R", 200), "2": (800, -200, "R", 200)},
    "2N7002": {"1": (200, 0, "L", 200), "3": (300, 100, "U", 200), "2": (300, -100, "D", 200)},
    "CAP_POL": {"1": (100, 0, "L", 100), "2": (300, 0, "R", 100)},
    "AO3401A": {"1": (0, 2900, "L", 200), "2": (200, 2600, "D", 200), "3": (400, 2900, "R", 200)},
    "AO4407A": {"1": (200, 0, "L", 200), "2": (200, -100, "L", 200), "3": (200, -200, "L", 200), "4": (200, -400, "L", 200),
                "5": (800, 0, "R", 200), "6": (800, -100, "R", 200), "7": (800, -200, "R", 200), "8": (800, -300, "R", 200)},
}
esp = {}
for i in range(1, 22):
    esp[str(i)] = (0, 4100 - (i - 1) * 100, "L", 200)
for i in range(22, 42):
    esp[str(i)] = (1300, 4100 - (i - 22) * 100, "R", 200)
SYM["ESP32-S3-WROOM-1"] = esp
BASE = {k: 0 for k in ("TPS54202", "INDUCTOR", "FUSE_PTC", "TVS_UNI", "CONN_DCJACK3", "AO4407A", "2N7002", "CAP_POL")}

UID = {'J1': 'LARRENTU', 'F1': 'AXBSQBDD', 'D2': 'MPOVDULB', 'C10': 'OQHIGRIE', 'C23': 'GEQHNXMH', 'Q1': 'SIJKHFGH', 'R13': 'PDUGZXWC', 'R17': 'CJQMKMKS', 'J2': 'IPXHUCUC', 'R12': 'HQXQVQOT', 'U8': 'UHZIGGNU', 'C13': 'JCWSNYIZ', 'C15': 'NHFRQBIB', 'C17': 'AWVJUHTA', 'L1': 'EXJPZLTZ', 'C2': 'PKZXTIPV', 'C18': 'RONSZRXQ', 'R19': 'QZVWRDXT', 'C19': 'FGONDDBB', 'R20': 'RHNZYECN', 'U9': 'WISSGBUX', 'C14': 'ZBUGWAGN', 'C16': 'DSXPYTQJ', 'C20': 'ROFATXIB', 'L2': 'QHVUEGTG', 'C1': 'JSYMXWSU', 'C21': 'GFAPOTZX', 'R21': 'WLDSHLHM', 'C22': 'HIACGPIE', 'R22': 'UFNIUQPH', 'U7': 'DMOWYEOF', 'C4': 'HWQKVUUL', 'R10': 'XWJFOFPJ', 'R11': 'PZHIMIYX', 'R18': 'WCNFAMSQ', 'R15': 'VBIGISGA', 'R16': 'TLKRYWTD', 'U2': 'EVBTLQEC', 'C6': 'MWOTXQNY', 'C7': 'JTIIAVYQ', 'C8': 'FTHUFLZV', 'R1': 'UQEZVUJD', 'R2': 'JGUWTYTX', 'U3': 'JONZAHCO', 'C9': 'MXWIGYVF', 'C11': 'JVKYLKFW', 'C12': 'HBBTYRSM', 'R3': 'JJDACZSN', 'R4': 'NVYXBZNO', 'U1': 'GKWJANVW', 'C3': 'UCEJSYHD', 'R9': 'VWAEVGAH', 'C5': 'CDEDXQBY', 'J3': 'KZQYUFBB', 'U4': 'ZMZCBDUM', 'U5': 'EBIDXNXH', 'R5': 'AVYMKMST', 'R6': 'GDMRIXTS', 'U6': 'YXBUJGMN', 'R7': 'KPQUZJCR', 'R8': 'MNKEZZHV', 'R14': 'MGEPHOKT', 'BZ1': 'KQPTAQNN'}

POWER = {"GND", "3V3", "5V", "VIN12V"}
parts, wires, ports, labels, nets = [], [], [], [], {}


OY = 1500   # whole sheet lifted so the bottom block sits inside the frame


def place(des, sym, x, y, value, conns):
    y += OY
    ox, oy = x, y - BASE.get(sym, 2900)
    p = {"designator": des, "lib_reference": sym, "library_path": LIB, "x": ox, "y": oy, "rotation": 0, "value": value}
    if des in UID:
        p["unique_id"] = UID[des]
    parts.append(p)
    for pin, net in conns.items():
        if net is None:
            continue
        px, py, side, ln = SYM[sym][pin]
        ax, ay = ox + px, oy + py
        if net in POWER:
            stub = 200 + 300 * ((py // 100) % 2)
        else:
            stub = 800 if side == "L" else 200
        if side == "L":
            cx, cy = ax - ln, ay; ex, ey = cx - stub, cy
        elif side == "R":
            cx, cy = ax + ln, ay; ex, ey = cx + stub, cy
        elif side == "U":
            cx, cy = ax, ay + ln; ex, ey = cx, cy + stub
        else:
            cx, cy = ax, ay - ln; ex, ey = cx, cy - stub
        wires.append({"net": net, "path": [f"{des}.{pin}", [ex, ey]]})
        if net in POWER:
            ports.append({"net": net, "at": [ex, ey], "style": "gnd_power" if net == "GND" else "bar"})
        else:
            labels.append({"net": net, "at": [ex, ey]})
        nets.setdefault(net, []).append(f"{des}.{pin}")


R, C = "RESISTOR", "CAPACITOR"
tp = lambda a, b: {"1": a, "2": b}

# ---------- column A: input protection + blower switch ----------
place("J1", "CONN_DCJACK3", 300, 13800, "PJ-102AH 12V IN", {"1": "VIN_RAW", "3": "GND", "2": "GND"})
place("F1", "FUSE_PTC", 1000, 13000, "MF-R400 4A PTC", tp("VIN_RAW", "VIN_FUSED"))
place("D2", "TVS_UNI", 1000, 12000, "SMBJ13A TVS", {"1": "VIN12V", "2": "GND"})
place("C10", C, 1000, 11500, "10uF 35V", tp("VIN12V", "GND"))
place("C23", C, 1000, 11000, "22uF 25V", tp("VIN12V", "GND"))
place("Q1", "AO4407A", 1000, 10300, "AO4407A", {"1": "VIN12V", "2": "VIN12V", "3": "VIN12V", "4": "GATE",
                                                "5": "BLOWER_V12", "6": "BLOWER_V12", "7": "BLOWER_V12", "8": "BLOWER_V12"})
place("R13", R, 1000, 9300, "10k", tp("GATE", "VIN12V"))
place("R17", R, 1000, 8800, "10k", tp("GATE", "COMP_OUT"))
place("J2", "CONN_BLOWER4", 1000, 8100, "Blower cable", {"1": "BLOWER_V12", "2": "GND", "3": "BLOWER_PWM", "4": "BLOWER_FG"})
place("R12", R, 1000, 7300, "10k", tp("3V3", "BLOWER_FG"))
# reverse-polarity P-MOSFET (replaces SS54): D=input, S=VIN12V, gate divider gives Vgs = -VIN/2
place("Q4", "AO4407A", 1000, 6500, "AO4407A", {"1": "VIN12V", "2": "VIN12V", "3": "VIN12V", "4": "RPP_G",
                                               "5": "VIN_FUSED", "6": "VIN_FUSED", "7": "VIN_FUSED", "8": "VIN_FUSED"})
place("R26", R, 1000, 5600, "10k", tp("RPP_G", "GND"))
place("R27", R, 1000, 5100, "10k", tp("RPP_G", "VIN12V"))
# blower bulk capacitor at J2 (Panasonic EEU-FC1E221, 220uF 25V, 555 mA ripple)
place("C25", "CAP_POL", 1000, 4400, "220uF 25V", tp("BLOWER_V12", "GND"))

# ---------- column B: TPS54202 bucks (TI SLVSD26C Fig 7-1 / Table 7-2) ----------
XB = 4000
place("U8", "TPS54202", XB, 13800, "TPS54202 (5V)", {"3": "VIN12V", "5": None, "1": "GND", "6": "BST5", "2": "SW5", "4": "FB5"})
place("C13", C, XB, 13100, "10uF 35V", tp("VIN12V", "GND"))
place("C15", C, XB, 12700, "100nF", tp("VIN12V", "GND"))
place("C17", C, XB, 12300, "100nF", tp("BST5", "SW5"))
place("L1", "INDUCTOR", XB, 11900, "15uH SRN6045TA-150M", tp("SW5", "5V"))
place("C2", C, XB, 11500, "22uF 16V", tp("5V", "GND"))
place("C18", C, XB, 11100, "22uF 16V", tp("5V", "GND"))
place("R19", R, XB, 10700, "100k 1%", tp("5V", "FB5"))
place("C19", C, XB, 10300, "75pF C0G", tp("5V", "FB5"))
place("R20", R, XB, 9900, "13.3k 1%", tp("FB5", "GND"))
place("U9", "TPS54202", XB, 9200, "TPS54202 (3V3)", {"3": "VIN12V", "5": None, "1": "GND", "6": "BST3", "2": "SW3", "4": "FB3"})
place("C14", C, XB, 8500, "10uF 35V", tp("VIN12V", "GND"))
place("C16", C, XB, 8100, "100nF", tp("VIN12V", "GND"))
place("C20", C, XB, 7700, "100nF", tp("BST3", "SW3"))
place("L2", "INDUCTOR", XB, 7300, "10uH SRN6045TA-100M", tp("SW3", "3V3"))
place("C1", C, XB, 6900, "22uF 16V", tp("3V3", "GND"))
place("C21", C, XB, 6500, "22uF 16V", tp("3V3", "GND"))
place("R21", R, XB, 6100, "100k 1%", tp("3V3", "FB3"))
place("C22", C, XB, 5700, "56pF C0G", tp("3V3", "FB3"))
place("R22", R, XB, 5300, "22.1k 1%", tp("FB3", "GND"))

# ---------- column X1: safety comparator + primary sensor ----------
X1, DY = 7600, 1000
place("U7", "LM393", X1, 10500 + DY, "LM393", {"2": "COMP_REF", "3": "PRESS_RED_ADC", "4": "GND", "5": "GND", "6": "GND",
                                                "1": "COMP_OUT", "7": None, "8": "5V"})
place("C4", C, X1, 9500 + DY, "100nF", tp("5V", "GND"))
place("R10", R, X1, 8900 + DY, "47k", tp("5V", "COMP_REF"))
place("R11", R, X1, 8300 + DY, "10k", tp("COMP_REF", "GND"))
place("R18", R, X1, 7700 + DY, "1M", tp("COMP_OUT", "PRESS_RED_ADC"))
place("R15", R, X1, 7100 + DY, "1M", tp("COMP_OUT", "SAFETY_TRIP"))
place("R16", R, X1, 6500 + DY, "330k", tp("SAFETY_TRIP", "GND"))
place("U2", "MPXV5010GC6U", X1, 5500 + DY, "MPXV5010GC6U (primary)", {"2": "5V", "3": "GND", "4": "PRESS_PRI_RAW"})
place("C6", C, X1, 4600 + DY, "1uF", tp("5V", "GND"))
place("C7", C, X1, 4100 + DY, "10nF", tp("5V", "GND"))
place("C8", C, X1, 3600 + DY, "470pF", tp("PRESS_PRI_RAW", "GND"))
place("R1", R, X1, 3100 + DY, "10k", tp("PRESS_PRI_RAW", "PRESS_PRI_ADC"))
place("R2", R, X1, 2600 + DY, "20k", tp("PRESS_PRI_ADC", "GND"))

# ---------- column X2: redundant sensor + ESP32 support ----------
X2 = 10600
place("U3", "MPXV5010GC6U", X2, 5500 + DY, "MPXV5010GC6U (redundant)", {"2": "5V", "3": "GND", "4": "PRESS_RED_RAW"})
place("C9", C, X2, 4600 + DY, "1uF", tp("5V", "GND"))
place("C11", C, X2, 4100 + DY, "10nF", tp("5V", "GND"))
place("C12", C, X2, 3600 + DY, "470pF", tp("PRESS_RED_RAW", "GND"))
place("R3", R, X2, 3100 + DY, "10k", tp("PRESS_RED_RAW", "PRESS_RED_ADC"))
place("R4", R, X2, 2600 + DY, "20k", tp("PRESS_RED_ADC", "GND"))
esp_nets = {"1": "GND", "2": "3V3", "3": "EN", "4": "PRESS_PRI_ADC", "5": "PRESS_RED_ADC", "6": "SAFETY_TRIP", "7": "BUZZER_CTRL",
            "10": "ENC_A", "11": "ENC_B", "12": "I2C_SDA", "15": "BLOWER_PWM", "17": "I2C_SCL",
            "23": "ENC_SW", "27": "BOOT", "36": "UART_RX", "37": "UART_TX", "38": "BLOWER_FG", "40": "GND", "41": "GND", "9": "SFM_EN"}
place("U1", "ESP32-S3-WROOM-1", 11200, 9100 + DY, "ESP32-S3-WROOM-1-N8R8", esp_nets)
place("C3", C, X2, 7300 + DY, "100nF", tp("3V3", "GND"))
place("R9", R, X2, 6800 + DY, "10k", tp("3V3", "EN"))
place("C5", C, X2, 6300 + DY, "1uF", tp("EN", "GND"))

# ---------- column X3: UI + I/O ----------
X3 = 14600
place("J3", "CONN_UART6", X3, 10500 + DY, "Prog UART", {"1": "GND", "2": "3V3", "4": "UART_RX", "3": "UART_TX", "5": "EN", "6": "BOOT"})
place("U4", "SFM3300-D", X3, 9600 + DY, "SFM3300-D", {"1": "GND", "2": "SFM_5V", "4": "GND", "3": "I2C_SCL", "5": "I2C_SDA", "6": None})
place("U5", "SSD1306_OLED_I2C", X3, 8700 + DY, "OLED 0.96in I2C", {"1": "GND", "2": "3V3", "3": "I2C_SCL", "4": "I2C_SDA"})
place("R5", R, X3, 8000 + DY, "4.7k", tp("3V3", "I2C_SCL"))
place("R6", R, X3, 7500 + DY, "4.7k", tp("3V3", "I2C_SDA"))
place("U6", "PEC11R-4220F-S0024", X3, 6700 + DY, "PEC11R-4220F-S0024", {"1": "ENC_A", "3": "GND", "2": "ENC_B", "4": "ENC_SW", "5": "GND"})
place("R7", R, X3, 5800 + DY, "10k", tp("3V3", "ENC_A"))
place("R8", R, X3, 5300 + DY, "10k", tp("3V3", "ENC_B"))
place("R14", R, X3, 4800 + DY, "10k", tp("3V3", "ENC_SW"))
place("BZ1", "BUZZER", X3, 4100 + DY, "PS1240P02BT", tp("BUZZER_CTRL", "GND"))
# SFM3300 power switch (datasheet 4.8: sensor must be resettable by power cycling). IO16 high = on (default via R24).
place("Q2", "AO3401A", X3, 4200, "AO3401A", {"1": "SFM_G", "2": "5V", "3": "SFM_5V"})
place("R23", R, X3 + 1400, 3700, "100k", tp("5V", "SFM_G"))
place("Q3", "2N7002", X3 + 800, 2700, "2N7002", {"1": "SFM_EN", "2": "GND", "3": "SFM_G"})
place("R24", R, X3, 2000, "10k", tp("3V3", "SFM_EN"))
place("C24", C, X3, 1500, "100nF", tp("SFM_5V", "GND"))

tpl = {"parts": parts, "wires": wires, "ports": ports, "labels": labels, "nets": nets}
out = os.path.join(HERE, "sipap_template_revc.json")
json.dump(tpl, open(out, "w"), indent=1)
print(len(parts), "parts", len(wires), "wires", len(nets), "nets")
