"""Hand-drawn copper for v3: buck tiles (TI SLVSD26C Fig 7-16), MOSFET copper, 12 V input/trunk, U1 EPAD vias.
Everything as dicts: regions {pts, net, layer}, tracks {x1,y1,x2,y2,width,net,layer}, vias {x,y,size,hole,net}."""
from model import rot
import place_v3 as PL

TOP, BOT = "TopLayer", "BottomLayer"
R, T, V = [], [], []

def reg(pts, net, layer=TOP):
    R.append(dict(pts=[(round(x, 2), round(y, 2)) for x, y in pts], net=net, layer=layer))

def rect(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def trk(pts, w, net, layer=TOP):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        T.append(dict(x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2), width=w, net=net, layer=layer))

def via(x, y, net, size=24, hole=12):
    V.append(dict(x=round(x, 2), y=round(y, 2), size=size, hole=hole, net=net))

# ------------------------------------------------------------------ buck tile, drawn in the tile frame
def buck(origin, Rdeg, vin, sw, vout, fb, bst):
    X, Y = origin
    f = lambda x, y: (X + rot(x, y, Rdeg)[0], Y + rot(x, y, Rdeg)[1])
    F = lambda pts: [f(*p) for p in pts]
    reg(F(rect(31.5, 25, 310, 100)), vin)                    # VIN copper: IC pin 3 + input caps pad 1
    reg(F(rect(31.5, -100, 310, -25)), "GND")                # GND copper: IC pin 1 + input caps pad 2
    for dx in (190, 265):          # clear of the bottom SW track
        via(*f(dx, -80), "GND")
    reg(F(rect(31.5, -12, 165, 12)), sw)                     # SW pin strip between VIN and GND copper
    via(*f(100, 0), sw); via(*f(135, 0), sw)                 # down to the bottom-layer SW node
    trk(F([(100, 0), (135, 0)]), 24, sw, BOT)
    trk(F([(117.5, 0), (117.5, -65), (0, -182.5)]), 30, sw, BOT)
    via(*f(0, -182.4), sw)
    trk(F([(-53.15, -182.4), (165, -182.4)]), 25, sw)        # BOOT cap pad 2 - via - inductor pad 1
    trk(F([(-53.15, -37.4), (-53.15, -107.6)]), 15, bst)     # VBST -> BOOT cap pad 1
    reg(F(rect(64.5, -467, 265.5, -332)), vout)              # VOUT copper: inductor pad 2 + output caps pad 1
    reg(F(rect(103, -541.9, 227, -502.9)), "GND")            # output caps pad 2
    for dx in (127.5, 202.5):
        trk(F([(dx, -522.4), (dx, -575)]), 20, "GND"); via(*f(dx, -575), "GND")
    # feedback: FB pin -> divider (away from SW), sensed at the output caps (Kelvin)
    trk(F([(-53.15, 37.4), (-90, 37.4), (-104.8, 22.6), (-146, 22.6)]), 10, fb)
    trk(F([(-160, 12.6), (-160, -42.6)]), 10, fb)
    trk(F([(-235, -52.6), (-174, -52.6)]), 10, fb)
    trk(F([(-221, -127.4), (-174, -127.4)]), 10, vout)
    trk(F([(-235, -141.4), (-235, -395), (-215, -415), (64.5, -415)]), 10, vout)
    trk(F([(-160, 97.4), (-215, 97.4)]), 15, "GND"); via(*f(-215, 97.4), "GND")

buck(PL.T3, 90, "VIN12V", "SW3", "3V3", "FB3", "BST3")
buck(PL.T5, 90, "VIN12V", "SW5", "5V", "FB5", "BST5")

# ------------------------------------------------------------------ 12 V input: J1 -> F1 -> Q4 -> VIN12V node
trk([(539.4, 2480), (689.4, 2480), (799.6, 2590.2), (799.6, 2700)], 60, "VIN_RAW")
reg([(963, 2700), (1038, 2700), (1038, 2600), (1088, 2600), (1088, 2516), (912, 2516), (912, 2600), (963, 2600)], "VIN_FUSED")
reg(rect(813, 2254, 1037, 2364), "VIN12V")             # Q4 source pins + TVS cathode
reg(rect(700, 2232, 1037, 2254), "VIN12V")             # link to the 3V3 tile VIN copper (above its GND copper)
reg(rect(700, 2210, 775, 2240), "VIN12V")
trk([(1207.4, 2333.7), (1207.4, 2290), (1187.4, 2270), (1037, 2270)], 20, "VIN12V")   # R27 (gate pull-up) to the node
# trunk down the left: both tiles' VIN, bulk caps, Q1 source
trk([(700, 2225), (655, 2225), (630, 2200), (630, 1000), (665, 965), (665, 930)], 50, "VIN12V")
trk([(630, 1560), (700, 1560)], 50, "VIN12V")          # 5V tile VIN copper
reg(rect(603, 836, 727, 930), "VIN12V")                # Q1 source pins 1-3
trk([(507.4, 2150), (630, 2150)], 25, "VIN12V"); trk([(507.4, 2075), (630, 2075)], 25, "VIN12V")
trk([(540, 1022.4), (630, 1022.4)], 20, "VIN12V")      # R13 (Q1 gate pull-up)
# blower switch output: Q1 drain -> J2.1 + C25 (+)
reg(rect(515, 598, 727, 684), "BLOWER_V12")           # Q1 drain pins 5-8 as one copper area
trk([(530, 630), (369, 630)], 50, "BLOWER_V12")         # from the copper area, not over the pads
trk([(368.9, 630), (368.9, 870)], 40, "BLOWER_V12")    # C25 (+)
trk([(369, 630), (300, 561), (300, 550)], 50, "BLOWER_V12")   # J2.1

# ------------------------------------------------------------------ U1 EPAD: 4 GND vias in the gaps of the 3x3 land grid
for x, y in ((1913.4, 2365.6), (1968.5, 2365.6), (1913.4, 2420.7), (1968.5, 2420.7)):
    via(x, y, "GND", 20, 10)

# ------------------------------------------------------------------ U1 3V3 / EN corner (pins 1-3)
for a, b in (((1655.5, 2647.1), (1572, 2647.1)), ((1572, 2647.1), (1505, 2647.1)), ((1505, 2647.1), (1440, 2647.1))):
    trk([a, b], 20, "3V3")                                                     # bar split at each pad centre
trk([(1440, 2647.1), (1350, 2647.1), (1330, 2627.1), (1330, 2200), (1300, 2170), (1267, 2170)], 20, "3V3")   # bar + riser from the 3V3 buck
trk([(1626, 2697.1), (1610, 2697.1), (1585.2, 2721.9)], 15, "GND")          # pin 1 -> C3 GND pad
trk([(1572, 2721.9), (1505, 2721.9)], 15, "GND")
trk([(1538.5, 2721.9), (1538.5, 2762)], 15, "GND"); via(1538.5, 2762, "GND")   # between C27 and C3
trk([(1626, 2597.1), (1490, 2597.1), (1465.2, 2572.3), (1440, 2572.3)], 10, "EN")   # pin 3 -> R9 (pull-up)
trk([(1440, 2562.3), (1440, 2527.4)], 10, "EN")                              # R9 -> C5
trk([(1440, 2452.6), (1440, 2400)], 15, "GND"); via(1440, 2400, "GND")       # C5 GND

# ------------------------------------------------------------------ UART + BOOT to J3 (pin 1 at top): parallel lanes
trk([(2344.5, 2547.1), (2400, 2547.1), (2497.1, 2450), (3800, 2450)], 10, "UART_TX")
trk([(2344.5, 2497.1), (2370, 2497.1), (2517.1, 2350), (3800, 2350)], 10, "UART_RX")
trk([(2344.5, 2047.1), (2380, 2047.1), (2482.9, 2150), (3800, 2150)], 10, "BOOT")
# ------------------------------------------------------------------ I2C: top to the OLED header U5, bottom bus (V1) to the SFM3300 header U4
trk([(2344.5, 2647.1), (2400, 2647.1), (2452.9, 2700), (3330, 2700), (3350, 2720), (3350, 2750)], 10, "I2C_SCL")
trk([(2344.5, 2597.1), (3420, 2597.1), (3450, 2627.1), (3450, 2750)], 10, "I2C_SDA")
trk([(2760, 2700), (2760, 2738)], 10, "I2C_SCL")           # R5 (pull-up) stub
trk([(2860, 2597.1), (2860, 2565.4)], 10, "I2C_SDA")       # R6 (pull-up) stub
via(2620, 2700, "I2C_SCL"); via(2650, 2597.1, "I2C_SDA")
trk([(2620, 2700), (2620, 500), (2640, 480), (3430, 480), (3450, 460), (3450, 400)], 10, "I2C_SCL", BOT)
trk([(2650, 2597.1), (2650, 530), (2670, 510), (3630, 510), (3650, 490), (3650, 400)], 10, "I2C_SDA", BOT)
