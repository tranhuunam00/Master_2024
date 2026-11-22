"""SiPAP Rev C layout v3 placement. Units mil, origin = board bottom-left. Rotations in degrees (Altium)."""
from model import CUR, rot

P = dict(CUR)

def put(d, x, y, r):
    P[d] = (round(x, 3), round(y, 3), r % 360)

def block(origin, rel, R=0):
    X, Y = origin
    for d, (dx, dy, r) in rel.items():
        ox, oy = rot(dx, dy, R)
        put(d, X + ox, Y + oy, r + R)

# ---- fixed: enclosure v0.3 cut-outs / module antenna edge
put("U1", 2000, 2490, 0); put("J1", 0, 2480, 0); put("F1", 900, 2700, 0); put("J2", 300, 450, 0)
put("U2", 3560, 1850, 180); put("U3", 3560, 1150, 180); put("U4", 3250, 400, 0); put("U5", 3150, 2750, 0)
put("U6", 2250, 560, 180)      # shaft stays at (57.15, 14.22) mm; A/C/B now face U1
put("BZ1", 1720, 280, 0)
put("J3", 3800, 2650, 270)      # pin 1 (GND) at top: TXD 2450 / RXD 2350 run parallel to U1 pins 37/36

# ---- power input J1 -> F1 -> Q4
put("Q4", 1000, 2440, 90)      # D up to F1.2, S down (VIN12V), G = pin 4 at (1075, 2334)
put("R27", 1170, 2333.7, 0)    # RPP_G left (on G) / VIN12V right
put("R26", 1132.6, 2440, 90)   # RPP_G bottom / GND top
put("D2", 720, 2330, 180)      # K = VIN12V right, A = GND left
put("C10", 470, 2150, 180)       # VIN12V left / GND right
put("C23", 470, 2075, 180)

# ---- TPS54202 tiles (TI SLVSD26C Fig 7-16), identical, turned 90: VIN top-left, output right
def buck_rel(ic, cin_hf, cin, bst, ind, co_a, co_b, rtop, cff, rbot):
    return {ic: (0, 0, 180), cin_hf: (200, 0, 270), cin: (275, 0, 270), bst: (-53.15, -145, 270),
            ind: (165, -274.9, 0), co_a: (127.5, -485, 270), co_b: (202.5, -485, 270),
            rtop: (-160, -90, 90), cff: (-235, -90, 90), rbot: (-160, 60, 90)}
T3 = (800, 1910); T5 = (800, 1290)
block(T3, buck_rel("U9", "C16", "C14", "C20", "L2", "C21", "C1", "R21", "C22", "R22"), 90)
block(T5, buck_rel("U8", "C15", "C13", "C17", "L1", "C2", "C18", "R19", "C19", "R20"), 90)

# ---- U1 helpers: 3V3 bar on the pin-2 row (y 2647.1), decoupling pads on it, EN RC below (hand copper)
put("C3", 1572, 2684.5, 90)     # 3V3 bottom on the bar / GND top
put("C27", 1505, 2684.5, 90)    # 22 uF, same
put("R9", 1440, 2609.7, 270)    # 3V3 top on the bar / EN bottom
put("C5", 1440, 2480, 270)      # EN top / GND bottom

# ---- blower block (bottom-left)
put("C25", 300, 870, 180)      # + (BLOWER_V12) right, - left
put("Q1", 640, 760, 270)       # S (VIN12V) + G on top, D (BLOWER_V12) at the bottom
put("R13", 540, 985, 90)       # GATE bottom / VIN12V top
put("R17", 860, 960, 180)      # COMP_OUT right / GATE left
put("R28", 620, 550, 180)      # VSR left / PWM right
put("R29", 560, 400, 0)        # FG_IN left / FG right
put("C26", 490, 512.6, 270)    # VSR top (on J2.3-R28 line) / GND bottom
put("R12", 1925, 1760, 90)     # 3V3 bottom / FG top, at U1 pin 19

# ---- comparator block below U1's bottom-left
put("U7", 1500, 1300, 90)
put("R10", 1400, 1040, 90); put("R11", 1500, 1040, 90); put("R18", 1720, 1250, 90)
put("R15", 1295, 1160, 0); put("R16", 1295, 1060, 0); put("C4", 1480, 1530, 0)

# ---- right side: sensor tiles (VS pin2 y+50, GND pin3, VOUT pin4 y-150 on the left at x = 3230)
def sensor(cvs1, cvs2, cflt, rser, rsh, Y):
    """MPXV5010 decoupling per NXP datasheet Fig. (1 uF + 10 nF on VS, 470 pF on VOUT) + 10k/20k divider."""
    put(cvs1, 3110, Y + 0, 270)          # 5V top (on VS pin y+50) / GND bottom (on GND pin y-50)
    put(cvs2, 3040, Y + 0, 270)
    put(cflt, 3110, Y - 187.4, 270)      # RAW top (on VOUT y-150) / GND bottom
    put(rser, 2980, Y - 150, 180)        # RAW right / ADC left
    put(rsh, 2942.6, Y - 250, 270)       # ADC top / GND bottom
sensor("C6", "C7", "C8", "R1", "R2", 1850)
sensor("C9", "C11", "C12", "R3", "R4", 1150)
put("R5", 2760, 2775.4, 270)   # SCL pull-up above the SCL line, short stub down to it
put("R6", 2860, 2528, 90)      # SDA pull-up between the SDA and TX lanes, short stub up to SDA
# SFM3300 power switch at U4 (pins: 1 GND 3250, 2 SFM_5V 3350, 3 SCL 3450, 4 GND 3550, 5 SDA 3650)
put("Q2", 3350, 280, 0)          # D up to U4.2, G bottom-left, S bottom-right
put("C24", 3225, 300, 180)       # SFM_5V right / GND left (to U4.1)
put("R23", 3350, 160, 180)       # 5V right (under S) / SFM_G left (under G)
put("Q3", 3200, 200, 270)        # D right -> SFM_G, G top-left, S bottom-left
put("R24", 3060, 237.4, 0)       # 3V3 left / SFM_EN right (to Q3.G)
# pull-ups in one row right under U1's bottom row (signal pad top, 3V3 pad bottom)
put("R12", 1925, 1690, 90); put("R8", 2110, 1690, 90); put("R7", 2190, 1690, 90); put("R14", 2270, 1690, 90)

# ---- test points / fiducials (free pads): net -> new position
TP_MOVE = {"VIN12V": (1272, 2270), "5V": (1170, 1260), "3V3": (1400, 2190), "GND": (450, 1330)}
FID_MOVE = {"FID1": (1250, 2950), "FID2": (3600, 2930), "FID3": (1000, 80)}

# apply free-pad moves to the shared model (test points by net, fiducials by name)
from model import FREEPADS
for _f in FREEPADS:
    _f.setdefault("ox", _f["x"]); _f.setdefault("oy", _f["y"])
    if _f["name"].startswith("TP") and _f["net"] in TP_MOVE:
        _f["x"], _f["y"] = TP_MOVE[_f["net"]]
    elif _f["name"] in FID_MOVE:
        _f["x"], _f["y"] = FID_MOVE[_f["name"]]

# bottom-layer bus corridors (router: bottom cost ~ top cost inside, x3 elsewhere)
BOTTOM_CORRIDORS = [(1420, 1730, 2720, 1930),     # H1: under U1's bottom edge, left -> right
                    (1420, 1730, 1625, 2660),     # H1: along U1's left pad column
                    (2540, 280, 2680, 2720)]      # V1: right of U1, down to U4
ROUTE_PLAN = ["VIN12V", "VIN_RAW", "VIN_FUSED", "BLOWER_V12",
              "3V3", "5V",
              "EN", "PRESS_PRI_ADC", "PRESS_RED_ADC", "SFM_EN", "I2C_SCL", "I2C_SDA",
              "UART_TX", "UART_RX", "BOOT",
              "SAFETY_TRIP", "BLOWER_PWM", "BLOWER_FG", "BUZZER_CTRL", "ENC_B", "ENC_A", "ENC_SW",
              "COMP_OUT", "GATE", "COMP_REF", "PRESS_PRI_RAW", "PRESS_RED_RAW", "RPP_G",
              "BLOWER_VSR", "BLOWER_FG_IN", "SFM_G", "SFM_5V"]
