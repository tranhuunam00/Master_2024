"""Hand-drawn power copper (TI SLVSD26C Fig 7-16 tiles, MOSFET copper, 12 V trunk)."""
from placement import P, T5, T3, tile

R, T, V = [], [], []          # regions, tracks, vias
TOP, BOT = "TopLayer", "BottomLayer"
VIA = dict(size=24, hole=12)

def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def reg(pts, net, layer=TOP):
    R.append(dict(pts=[(round(x, 2), round(y, 2)) for x, y in pts], net=net, layer=layer))

def trk(pts, w, net, layer=TOP):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        T.append(dict(x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2), width=w, net=net, layer=layer))

def via(x, y, net):
    V.append(dict(x=round(x, 2), y=round(y, 2), net=net, **VIA))

TRUNK_Y = 2190

def buck(X, Y, sw, vout, fb, bst_dummy=None):
    a = lambda dx, dy: (X + dx, Y + dy)
    A = lambda pts: [a(*p) for p in pts]
    reg(A(rect(31, 25, 300, 100)), "VIN12V")                 # VIN copper (pin 3 + input caps)
    reg(A(rect(100, 95, 150, TRUNK_Y - 25 - Y + 10)), "VIN12V")  # stub up into the 12 V trunk
    reg(A(rect(31, -100, 300, -25)), "GND")                  # GND copper (pin 1 + input caps)
    for dx in (110, 190, 265):
        via(*a(dx, -80), "GND")
    reg(A(rect(31, -12, 160, 12)), sw)                       # SW pin strip
    via(*a(100, 0), sw); via(*a(135, 0), sw)                 # vias to bottom SW copper
    trk(A([(117.5, 0), (117.5, -99.5), (35, -182.4)]), 30, sw, BOT)
    via(*a(35, -182.4), sw)
    trk(A([(-53.15, -182.4), (90, -182.4)]), 20, sw)          # BOOT cap -> via -> inductor pad 1
    trk(A([(-53.15, -49.2), (-53.15, -87.9)]), 10, bst_dummy)  # VBST -> BOOT cap
    reg(A(rect(64.6, -422.3, 265.4, -302)), vout)            # VOUT copper (L pad 2 + output caps)
    trk(A([(132, -492.4), (257, -492.4)]), 20, "GND")
    via(*a(157, -540), "GND"); via(*a(232, -540), "GND")
    trk(A([(157, -512), (157, -540)]), 20, "GND"); trk(A([(232, -512), (232, -540)]), 20, "GND")
    # feedback: FB pin -> divider, sensed at the output caps (Kelvin), away from SW
    trk(A([(-53.15, 37.4), (-160, 37.4)]), 10, fb)
    trk(A([(-160, 22.6), (-160, -52.6)]), 10, fb)
    trk(A([(-235, -52.6), (-160, -52.6)]), 10, fb)
    trk(A([(-160, -127.4), (-235, -127.4), (-235, -400), (64.6, -400)]), 10, vout)
    via(*a(-215, 97.4), "GND"); trk(A([(-160, 97.4), (-215, 97.4)]), 10, "GND")   # FB divider ground

buck(*T5, "SW5", "5V", "FB5", "BST5")
buck(*T3, "SW3", "3V3", "FB3", "BST3")

# ---- 12 V trunk -------------------------------------------------------------
trk([(310 + 125, TRUNK_Y), (1225, TRUNK_Y)], 50, "VIN12V")
trk([(1150, 2300), (1150, TRUNK_Y)], 50, "VIN12V")
trk([(1225, 2232), (1225, TRUNK_Y)], 20, "VIN12V")          # R27 (gate pull-up) to trunk
trk([(665, TRUNK_Y), (665, 1150), (935, 1150), (935, 700)], 50, "VIN12V")

# ---- input: J1 -> F1 -> Q4 -> VIN12V ------------------------------------------
trk([(539.37, 2480), (539.37, 2665), (634.6, 2760)], 50, "VIN_RAW")
trk([(835.4, 2760), (835.4, 2680)], 60, "VIN_FUSED")
reg([(1050, 2596), (1240, 2596), (1240, 2680), (835.4 - 30, 2680), (835.4 - 30, 2610), (1050, 2610)], "VIN_FUSED")
reg([(899, 2300), (1187, 2300), (1187, 2444), (1050, 2444), (1050, 2420), (899, 2420)], "VIN12V")
trk([(1225, 2383), (1225, 2347)], 10, "RPP_G")               # Q4 gate -> R27
trk([(1225, 2327.4), (1300, 2327.4)], 10, "RPP_G")            # -> R26
via(1300, 2205, "GND"); trk([(1300, 2252.6), (1300, 2205)], 15, "GND")
via(1000, 2252.6, "GND"); trk([(950, 2252.6), (1050, 2252.6)], 15, "GND")   # bulk caps ground

# ---- blower switch Q1 (rot 180 at 750,720) ------------------------------------
reg(rect(826, 633, 960, 757), "VIN12V")                       # source pins 1-3 + trunk
reg([(613, 633), (674, 633), (674, 900), (380, 900), (380, 840), (613, 840)], "BLOWER_V12")
trk([(398.9, 880), (398.9, 690), (310, 601), (300, 550)], 50, "BLOWER_V12")

# ---- ESP32 U1 fan-out (hand): left column, top pin on the outermost lane ----------
LEFT = [("EN", 2597.1), ("PRESS_PRI_ADC", 2547.1), ("PRESS_RED_ADC", 2497.1), ("SAFETY_TRIP", 2447.1),
        ("BUZZER_CTRL", 2397.1), ("SFM_EN", 2297.1), ("ENC_A", 2247.1), ("ENC_B", 2197.1), ("I2C_SDA", 2147.1)]
LANE0, PITCH, PADX = 1390, 25, 1655.5
for k, (net, y) in enumerate(LEFT):
    x = LANE0 + k * PITCH
    trk([(PADX, y), (x + 15, y), (x, y - 15), (x, 1990 - k * 0)], 10, net)
# 3V3 into pin 2 from the left, through C3 (decoupling right at the pin)
trk([(1345, 2647.1), (PADX, 2647.1)], 20, "3V3")
trk([(1480, 2722.4), (1600, 2722.4), (1625, 2697.1), (PADX, 2697.1)], 15, "GND")
via(1440, 2722.4, "GND"); trk([(1480, 2722.4), (1440, 2722.4)], 15, "GND")
# bottom row: straight down out of the module
for net, x in (("BLOWER_PWM", 1725), ("I2C_SCL", 1825), ("ENC_SW", 2125)):
    trk([(x, 1997.9), (x, 1925)], 10, net)
# right column: straight out to the right
for net, y in (("BOOT", 2047.1), ("UART_RX", 2497.1), ("UART_TX", 2547.1), ("BLOWER_FG", 2597.1)):
    trk([(2344.5, y), (2440, y)], 10, net)

# U1 EPAD: 4 GND vias in the gaps of the 3x3 land grid (Espressif DS v1.8 Fig 11-1), 0.25/0.5 mm
for x, y in ((1968.5, 2420.71), (1913.39, 2420.71), (1968.5, 2365.59), (1913.39, 2365.59)):
    V.append(dict(x=x, y=y, net="GND", size=20, hole=10))

# blower speed input: J2.3 -> R28.2 -> C26.1 are neighbours, tie them by hand
trk([(400, 550), (482.6, 550), (482.6, 505), (490, 497.6), (490, 475.4)], 10, "BLOWER_VSR")
trk([(400, 450), (400, 360), (435, 325), (462.6, 325)], 10, "BLOWER_FG_IN")   # J2.4 -> R29.1
