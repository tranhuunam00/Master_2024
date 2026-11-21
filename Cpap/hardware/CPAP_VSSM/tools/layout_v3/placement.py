"""New placement for SiPAP Rev C layout v3. Units mil, origin board bottom-left."""
from model import CUR

P = dict(CUR)  # start from current, override movable parts

def tile(ic, cin_s, cin_b, bst, ind, cout_a, cout_b, rtop, cff, rbot, X, Y):
    """TPS54202 tile per TI SLVSD26C Fig 7-16, IC rotated 180 (VIN on top, VOUT at bottom)."""
    rel = {ic: (0, 0, 180), cin_s: (200, 0, 270), cin_b: (275, 0, 270), bst: (-53.15, -145, 270),
           ind: (165, -245, 0), cout_a: (232, -455, 270), cout_b: (157, -455, 270),
           rtop: (-160, -90, 90), cff: (-235, -90, 90), rbot: (-160, 60, 90)}
    for d, (dx, dy, r) in rel.items():
        P[d] = (X + dx, Y + dy, r)

T5 = (310, 2025)
T3 = (1000, 2025)
tile("U8", "C15", "C13", "C17", "L1", "C18", "C2", "R19", "C19", "R20", *T5)
tile("U9", "C16", "C14", "C20", "L2", "C21", "C1", "R21", "C22", "R22", *T3)

# input protection: J1 -> F1 -> Q4 (reverse-polarity P-FET) -> VIN12V
P["Q4"] = (1150, 2520, 90)
P["D2"] = (940, 2480, 90)
P["C10"] = (950, 2290, 270)
P["C23"] = (1050, 2290, 270)
P["R27"] = (1225, 2290, 270)
P["R26"] = (1300, 2290, 270)

# ESP32 helpers
P["C3"] = (1480, 2685, 90)
P["R9"] = (2560, 1880, 0)
P["C5"] = (2560, 1790, 0)

# blower switch gate network (Q1 rot 180 at 750,720: gate pad 856.3,795)
P["R13"] = (870, 900, 0)      # GATE (pin 1) -> VIN12V trunk (pin 2)
P["R17"] = (740, 975, 180)    # GATE (pin 1, right) -> COMP_OUT (pin 2, left)
# blower connector signals around J2
P["R28"] = (520, 550, 180)    # PWM (pin 1) -> VSR (pin 2) next to J2.3
P["C26"] = (490, 438, 270)  # VSR filter
P["R29"] = (500, 325, 0)      # FG_IN -> FG
P["R12"] = (640, 350, 180)    # FG pull-up to 3V3

# pull-ups where their signal already passes (v3 routing pass 6 boxed them in)
P["R14"] = (2125, 1862.6, 90)   # ENC_SW pull-up on the pin-23 escape (pin 2 top = ENC_SW)
P["U6"] = (2250, 560, 180)      # encoder turned 180: shaft at the origin (PEC11R dwg), A/B now on top
P["R7"] = (2348.43, 990, 270)   # ENC_A pull-up right above U6 pin 1
P["R8"] = (2151.57, 990, 270)   # ENC_B pull-up right above U6 pin 2
P["R12"] = (2520, 2597.1, 180)  # BLOWER_FG pull-up next to U1 pin 38
P["C4"] = (1220, 925, 180)      # U7 decoupling: pin 1 (5V) beside U7 pin 8
