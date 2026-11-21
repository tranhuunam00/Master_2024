"""SiPAP enclosure v0.3: real blower (Wonsmart WS4540-12-NZ03), every screw,
hose and cable modelled and checked (parametric FreeCAD, run headless).

    set SIPAP_OUT=<ascii output folder>      (optional; default ./out)
    "D:\\Program Files\\FreeCAD 1.1\\bin\\freecadcmd.exe" sipap_enclosure.py
    python tub_tilt_check.py out/report.json   (no-spill check, also run inside)

Axes: x right, y back (front face y=0), z up. Units mm.

Sources for the dimensions (all in Cpap/tools/datasheets/):
  * Blower: Wonsmart_WS4540-12-NZ03.pdf (drawing 2023-07-12): volute D48, 54 x 48,
    45 thick incl. the D10 x 10 inlet stub, outlet D10 OD / D8 ID tangential at the
    bottom, 30 from the axis; 4 x D1.6 holes, 15 deep, on PCD 36.8 (26 x 26 square)
    on the motor side; wires leave the motor board on the outlet side, 150 +- 20 mm.
  * SFM3300-D: SFM3300.pdf v1.4 sec 3.2: length 69.3, flow channel 11.25 mm,
    inlet = 22 mm female socket, outlet = 22 mm male cone (ISO 5356-1).
  * Encoder: PEC11R.pdf: flatted shaft 25 mm option -> PEC11R-4225F-S0024.
  * PCB: SiPAP_PCB_RevC.step (Altium) + pad coordinates of J2/U4/U5/U2/U3.

Layout (left bay = electronics + air path, right bay = humidifier):
  * Base floor -> blower on an L-bracket (rubber washers + rubber grommets),
    inside an open-bottom acoustic HOOD (foam-lined) with an intake PLENUM
    behind it (filter pad + 2-baffle labyrinth). PCB on 4 standoffs above the hood.
  * Blower outlet -> silicone sleeve -> printed DUCT (ID 12 = SFM channel, DS 4.2)
    -> riser behind the PCB -> 22 mm male cone into the SFM3300 inlet.
  * SFM3300 on 2 cradles (cable ties) -> 22 mm female DOCK in the partition wall
    (pressure tap for U2/U3) -> tub nozzle with O-ring (slides out to the right).
  * Tub: internal snorkels with both mouths at the tub centre, so water cannot
    reach them at any tilt (checked numerically, see tub_tilt_check.py).
"""
import json
import math
import os
import sys

import FreeCAD as App  # noqa: F401  (needed for the Part kernel)
import Part
from FreeCAD import Vector as V

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
sys.path.insert(0, HERE)
OUT = os.environ.get("SIPAP_OUT", os.path.join(HERE, "out"))
PCB_STEP = os.environ.get(
    "SIPAP_PCB_STEP",
    os.path.join(HERE, "..", "..", "hardware", "CPAP_VSSM", "3D", "SiPAP_PCB_RevC.step"))
FONT = os.environ.get("SIPAP_FONT", r"C:\Windows\Fonts\arialbd.ttf")
os.makedirs(OUT, exist_ok=True)
_LOG = open(os.path.join(OUT, "progress.log"), "w")


def log(msg):
    _LOG.write(msg + "\n")
    _LOG.flush()


# =========================================================== parameters
T = 3.0             # outer wall
FLOOR = 3.0
LID_T = 3.0
E_W = 103.6         # left bay inner width (PCB 101.6 + 2 x 1)
U_W = 100.0         # humidifier bay inner width
D_IN = 118.0        # inner depth

PCB_W, PCB_D, PCB_T = 101.6, 76.2, 1.6
PCB_HOLES = [(3.81, 3.81), (97.79, 3.81), (3.81, 72.39), (97.79, 72.39)]   # D3.2 (126 mil)
PCB_TO_LID = 22.0   # Dupont housing 2.54 + 14 + wire bend over J2/U4/U5 needs ~21 mm
LEAD_CLEAR = 3.0    # hood top -> PCB bottom (THT leads cut to <= 2 mm)

# Wonsmart WS4540-12-NZ03 (drawing 2023-07-12)
BL = dict(d=48.0, body=35.0, stub_l=10.0, stub_d=10.0, out_od=10.0, out_id=8.0,
          out_reach=30.0, out_drop=17.5, out_from_inlet=9.0, pcd=36.8, hole=1.6, hole_depth=15.0)
FOAM = 2.0          # acoustic PU foam lining in the hood (floor, walls, top)
GAP = 1.5           # free gap blower <-> foam
HOOD_W = 2.0        # hood wall
BRK_T = 3.0         # bracket plate
WASHER = 1.5        # rubber washer, bracket <-> blower motor face

SFM = dict(len=69.3, body_d=25.3, cone=15.0, cone_od=22.0)
DUCT_ID, DUCT_OD = 12.0, 16.0          # = SFM channel 11.25 (DS 4.2)
TUBE_ID, TUBE_OD = 15.0, 19.0          # tub nozzle / snorkels
CABLE_R = 1.6                           # 4..6 x 26 AWG bundle
PTUBE_OD = 5.0                          # silicone 3 x 5 mm pressure tubing

# ---------------------------------------------------------- derived heights
BZ = FLOOR + FOAM + GAP + BL["d"] / 2                         # blower axis height
HOOD_TOP = BZ + BL["d"] / 2 + GAP + FOAM + HOOD_W             # hood outer top
PCB_BOT = HOOD_TOP + LEAD_CLEAR
PCB_TOP = PCB_BOT + PCB_T
Z_TOP = PCB_TOP + PCB_TO_LID
H_IN = Z_TOP - FLOOR

W = 3 * T + E_W + U_W
D = D_IN + 2 * T
EX0, EX1 = T, T + E_W
UX0, UX1 = EX1 + T, EX1 + T + U_W
GX0, GY0 = T + 1.0, T + 2.0
AIR_Y, AIR_Z = 100.0, 63.0             # SFM / dock / tub-inlet axis (behind the PCB)

HOOD = dict(x0=12.5, x1=94.0, y0=8.0, y1=80.0)
PLEN = dict(x0=66.0, x1=94.0, y1=116.0, top=45.0)
BX_M = HOOD["x0"] + HOOD_W + FOAM + 8.0 + BRK_T + WASHER      # blower motor face x (foot 8 mm)
BY = 38.0                                                      # blower axis y
SFM_X0 = 30.0                                                  # SFM inlet face (distal, 22F)
SFM_X1 = SFM_X0 + SFM["len"]
DOCK_X0 = SFM_X1 - SFM["cone"]

TUB = dict(x0=UX0 + 2.0, y0=T + 8.5, z0=FLOOR + 6.0, w=96.0, d=101.0, h=71.0, wall=2.5,
           face=UX1 + T)                       # face plate flush with the outer wall
TUB_IN = dict(x0=TUB["x0"] + TUB["wall"], x1=TUB["x0"] + TUB["w"] - TUB["wall"],
              y0=TUB["y0"] + TUB["wall"], y1=TUB["y0"] + TUB["d"] - TUB["wall"],
              z0=TUB["z0"] + TUB["wall"], z1=TUB["z0"] + TUB["h"])
TUB_C = ((TUB_IN["x0"] + TUB_IN["x1"]) / 2, (TUB_IN["y0"] + TUB_IN["y1"]) / 2,
         (TUB_IN["z0"] + TUB_IN["z1"]) / 2)
LIP = 3.0                                      # inner rim lip carrying the gasket groove
Z_MOUTH = TUB_C[2]                             # snorkel mouths, facing down (sweep: best of 48 layouts)
MOUTH_IN = (TUB_C[0] - 7.0, TUB_C[1] + 7.0)    # 19.8 mm apart (tube OD 19)
MOUTH_OUT = (TUB_C[0] + 7.0, TUB_C[1] - 7.0)
TUB_OUT_Y = TUB["y0"] + 22.0
SLOSH_MARGIN = 3.0                             # mm of dry margin required at every tilt


def pcb(x, y):
    return GX0 + x, GY0 + y


def box(x, y, z, dx, dy, dz):
    return Part.makeBox(dx, dy, dz, V(x, y, z))


def cyl(x, y, z, r, h, d=(0, 0, 1)):
    return Part.makeCylinder(r, h, V(x, y, z), V(*d))


def cone(x, y, z, r1, r2, h, d=(0, 0, 1)):
    return Part.makeCone(r1, r2, h, V(x, y, z), V(*d))


def fuse_all(base, parts):
    for p in parts:
        base = base.fuse(p)
    return base.removeSplitter()


def cut_all(base, parts):
    for p in parts:
        base = base.cut(p)
    return base


def compound(parts):
    return Part.makeCompound([p for p in parts if p is not None])


def round_vertical(shape, r):
    try:
        vert = [e for e in shape.Edges if len(e.Vertexes) == 2 and abs(e.Vertexes[0].Z - e.Vertexes[1].Z) > 1]
        return shape.makeFillet(r, vert)
    except Exception:  # noqa: BLE001
        return shape


def path_solid(pts, r, end_spheres=False):
    """Round tube (solid) along a polyline; sphere at each corner. Returns (solid, length)."""
    solids, length = [], 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        va, vb = V(*a), V(*b)
        L = (vb - va).Length
        if L < 1e-6:
            continue
        length += L
        solids.append(Part.makeCylinder(r, L, va, vb - va))
    joints = pts[1:-1] if not end_spheres else pts
    for p in joints:
        solids.append(Part.makeSphere(r, V(*p)))
    s = solids[0]
    for o in solids[1:]:
        s = s.fuse(o)
    return s.removeSplitter(), length


# ---- fasteners (reference solids; drawn at nominal size, clearance holes cut)
def screw_csk(x, y, z_head, L, up=True, d=3.0):
    """ISO 10642 countersunk; head top flush at z_head, shank along +z (up) or -z."""
    dk = 2 * d + 0.7 if d == 3.0 else 2 * d
    k = (dk - d) / 2
    s = 1 if up else -1
    head = cone(x, y, z_head, dk / 2, d / 2, k, (0, 0, s))
    shank = cyl(x, y, z_head + s * k, d / 2 - 0.05, L - k, (0, 0, s))
    return head.fuse(shank)


def screw_button(x, y, z_seat, L, d=3.0, down=True, core=None):
    """ISO 7380 button head sitting on z_seat, shank going down (or up).
    core: drawn shank diameter for thread-forming screws (= pilot hole)."""
    s = -1 if down else 1
    head = cyl(x, y, z_seat, 0.95 * d, 0.55 * d, (0, 0, -s))
    if core:
        d = core + 0.1
    shank = cyl(x, y, z_seat, d / 2 - 0.05, L, (0, 0, s))
    return head.fuse(shank)


def insert(x, y, z_top, L=5.7, od=4.0):
    """Heat-set insert (drawn at hole size; melts 0.3 mm into the plastic)."""
    return cyl(x, y, z_top - L, od / 2, L).cut(cyl(x, y, z_top - L - 1, 1.5, L + 2))


def nut(x, y, z, af=5.5, h=2.4):
    r = af / math.sqrt(3)
    pts = [V(x + r * math.cos(k * math.pi / 3), y + r * math.sin(k * math.pi / 3), z) for k in range(7)]
    return Part.Face(Part.makePolygon(pts)).extrude(V(0, 0, h)).cut(cyl(x, y, z - 1, 1.6, h + 2))


FASTENERS = []      # (group, qty, description)


def fbom(group, qty, desc):
    FASTENERS.append({"group": group, "qty": qty, "item": desc})


# =========================================================== blower + bracket
def blower_solid():
    xi = BX_M + BL["body"]                                       # inlet face
    body = cyl(BX_M, BY, BZ, BL["d"] / 2, BL["body"], (1, 0, 0))
    stub = cyl(xi, BY, BZ, BL["stub_d"] / 2, BL["stub_l"], (1, 0, 0))
    ox, oz = xi - BL["out_from_inlet"], BZ - BL["out_drop"]
    outlet = cyl(ox, BY, oz, BL["out_od"] / 2, BL["out_reach"], (0, 1, 0))
    b = fuse_all(body, [stub, outlet])
    b = b.cut(cyl(ox, BY + 10, oz, BL["out_id"] / 2, BL["out_reach"], (0, 1, 0)))
    b = b.cut(cyl(xi - 1, BY, BZ, 3.5, BL["stub_l"] + 2, (1, 0, 0)))
    for hy, hz in blower_holes():
        b = b.cut(cyl(BX_M - 0.1, hy, hz, 1.1, BL["hole_depth"], (1, 0, 0)))   # ST2.2 core
    return b, (ox, BY + BL["out_reach"], oz)


def blower_holes():
    h = BL["pcd"] / 2 / math.sqrt(2)                             # 13.0 -> 26 x 26 square
    return [(BY + sy * h, BZ + sz * h) for sy in (-1, 1) for sz in (-1, 1)]


FOOT = dict(x0=HOOD["x0"] + HOOD_W + FOAM, x1=BX_M - WASHER - BRK_T, z0=FLOOR + 1.0, t=3.0)
FOOT_HOLES = [(FOOT["x1"] - 4.5, BY + s * 12.0) for s in (-1, 1)]


def make_bracket():
    px0 = BX_M - WASHER - BRK_T
    plate = box(px0, BY - 24, FOOT["z0"], BRK_T, 48, BZ + 20 - FOOT["z0"])
    foot = box(FOOT["x0"], BY - 24, FOOT["z0"], px0 - FOOT["x0"] + 0.01, 48, FOOT["t"])
    gus = []
    for gy in (BY - 24, BY + 21):                                 # 2 gussets
        tri = Part.makePolygon([V(FOOT["x0"] + 1, gy, FOOT["z0"] + FOOT["t"]), V(px0, gy, FOOT["z0"] + FOOT["t"]),
                                V(px0, gy, FOOT["z0"] + FOOT["t"] + 18), V(FOOT["x0"] + 1, gy, FOOT["z0"] + FOOT["t"])])
        gus.append(Part.Face(tri).extrude(V(0, 3, 0)))
    b = fuse_all(plate, [foot] + gus)
    cuts = [cyl(px0 - 1, BY, BZ, 8.0, BRK_T + 2, (1, 0, 0))]       # hub / motor-board clearance
    for hy, hz in blower_holes():
        cuts.append(cyl(px0 - 1, hy, hz, 1.3, BRK_T + 2, (1, 0, 0)))   # ST2.2 clearance D2.6
    for x, y in FOOT_HOLES:
        cuts.append(cyl(x, y, FOOT["z0"] - 1, 3.2, FOOT["t"] + 2))     # grommet D6.4
    return cut_all(b, cuts)


def bracket_hardware():
    px0 = BX_M - WASHER - BRK_T
    parts = []
    for hy, hz in blower_holes():                                 # ST2.2 x 13 pan head
        parts.append(cyl(px0, hy, hz, 2.1, 1.6, (-1, 0, 0)))
        parts.append(cyl(px0, hy, hz, 1.1, 13.0, (1, 0, 0)))
    washers = [cyl(px0 + BRK_T, hy, hz, 3.0, WASHER, (1, 0, 0)).cut(cyl(px0 + BRK_T - 1, hy, hz, 1.3, 4, (1, 0, 0)))
               for hy, hz in blower_holes()]
    grommets, csk = [], []
    for x, y in FOOT_HOLES:
        g = cyl(x, y, FLOOR, 4.5, 1.0).fuse(cyl(x, y, FLOOR + 1.0, 3.2, FOOT["t"])) \
            .fuse(cyl(x, y, FOOT["z0"] + FOOT["t"], 4.5, 1.0))
        grommets.append(g.cut(cyl(x, y, FLOOR - 1, 1.7, 10)))
        csk.append(screw_csk(x, y, 0.0, 12.0))
        csk.append(nut(x, y, FOOT["z0"] + FOOT["t"] + 1.0))
    fbom("blower", 4, "Vit tu ren ST2.2 x 13 dau tru (pan head) - vao lo D1.6 sau 15 cua blower")
    fbom("blower", 4, "Long den cao su D6 x D2.4 x 1.5 mm (giua ga va mat motor)")
    fbom("blower", 2, "Grommet cao su cho lo D6.4, tam 3 mm (chan ga)")
    fbom("blower", 2, "Vit M3 x 12 dau bang ISO 10642 + 2 dai oc M3 (tu duoi day vo len)")
    return compound(parts), compound(washers), compound(grommets), compound(csk)


# =========================================================== hood + plenum (one print, open bottom)
HOOD_TABS = [(HOOD["x0"] - 4.0, (HOOD["y0"] + HOOD["y1"]) / 2), (HOOD["x1"] + 4.0, (HOOD["y0"] + HOOD["y1"]) / 2),
             (PLEN["x1"] + 4.0, PLEN["y1"] - 10.0)]


def make_hood():
    h, p, w = HOOD, PLEN, HOOD_W
    ch = box(h["x0"], h["y0"], FLOOR, h["x1"] - h["x0"], h["y1"] - h["y0"], HOOD_TOP - FLOOR)
    ch = ch.cut(box(h["x0"] + w, h["y0"] + w, FLOOR - 1, h["x1"] - h["x0"] - 2 * w, h["y1"] - h["y0"] - 2 * w,
                    HOOD_TOP - FLOOR - w + 1))
    pl = box(p["x0"], h["y1"] - w, FLOOR, p["x1"] - p["x0"], p["y1"] - h["y1"] + w, p["top"] - FLOOR)
    pl = pl.cut(box(p["x0"] + w, h["y1"], FLOOR - 1, p["x1"] - p["x0"] - 2 * w, p["y1"] - h["y1"] - w,
                    p["top"] - FLOOR - w + 1))
    # labyrinth: b2 (rear) open at the bottom, b1 (front) open at the top
    b2 = box(p["x0"] + w, 100.0, FLOOR + 10.0, p["x1"] - p["x0"] - 2 * w, 2.0, p["top"] - w - FLOOR - 10.0 + 0.01)
    b1 = box(p["x0"] + w, 90.0, FLOOR, p["x1"] - p["x0"] - 2 * w, 2.0, p["top"] - w - FLOOR - 9.0)
    tabs = [box(x - 4, y - 5, FLOOR, 8, 10, 3) for x, y in HOOD_TABS]
    body = fuse_all(ch, [pl, b1, b2] + tabs)
    # tab webs into the hood wall
    cuts = [box(p["x0"] + 4, h["y1"] - w - 0.5, 22.0, p["x1"] - p["x0"] - 8, w + 1, 18.0),   # plenum -> chamber
            cyl(BX_M + BL["body"] - BL["out_from_inlet"], h["y1"] - w - 0.5, BZ - BL["out_drop"], 7.0, w + 1, (0, 1, 0)),
            cyl(BX_M + 1.6, h["y1"] - w - 0.5, 48.0, 3.5, w + 1, (0, 1, 0))]                  # blower cable grommet
    for i in range(4):                                               # plenum rear slots (filter behind)
        cuts.append(box(p["x0"] + 4 + i * 5.5, p["y1"] - w - 0.5, FLOOR + 6, 3.5, w + 1, 30))
    for x, y in HOOD_TABS:
        cuts.append(cyl(x, y, FLOOR - 1, 1.7, 5))
    return cut_all(body, cuts)


def hood_hardware():
    parts = []
    for x, y in HOOD_TABS:
        parts.append(screw_csk(x, y, 0.0, 10.0))
        parts.append(nut(x, y, FLOOR + 3.0))
    fbom("hood", 3, "Vit M3 x 10 ISO 10642 + dai oc M3 (tai hop tieu am -> day vo)")
    fbom("hood", 1, "Mut PU tieu am 2 mm, tu dinh, ~260 cm2 (lot day, 4 vach, tran hop)")
    fbom("hood", 1, "Doan gioang mut 3 x 3 mm duoi mep hop tieu am (~330 mm)")
    fbom("hood", 1, "Tam loc khi 28 x 42 x 5 mm (soi polyester) giua vach sau va buong hut")
    fbom("hood", 1, "Grommet cao su lo D14 / long D10 (ong ra blower qua vach hop)")
    fbom("hood", 1, "Grommet cao su lo D7 / long D4 (cap blower qua vach hop)")
    return compound(parts)


# =========================================================== duct (blower -> SFM inlet)
def duct_path():
    ox = BX_M + BL["body"] - BL["out_from_inlet"]
    oz = BZ - BL["out_drop"]
    return ox, oz, [(ox, 86.0, oz), (ox, AIR_Y, oz), (16.0, AIR_Y, oz), (16.0, AIR_Y, AIR_Z), (SFM_X0 - 4.0, AIR_Y, AIR_Z)]


def make_duct():
    ox, oz, pts = duct_path()
    outer, length = path_solid(pts, DUCT_OD / 2)
    inner, _ = path_solid(pts, DUCT_ID / 2)
    spig_y0 = HOOD["y1"] - 8.0                                    # spigot D10/8, through the hood wall
    spig = cyl(ox, spig_y0, oz, BL["out_od"] / 2, 86.0 - spig_y0, (0, 1, 0))
    flare = cone(ox, 82.0, oz, BL["out_od"] / 2, DUCT_OD / 2, 4.0, (0, 1, 0))
    collar = cyl(SFM_X0 - 4.0, AIR_Y, AIR_Z, 13.0, 4.0, (1, 0, 0))
    male = cone(SFM_X0, AIR_Y, AIR_Z, SFM["cone_od"] / 2, SFM["cone_od"] / 2 - 0.3, 14.0, (1, 0, 0))  # ISO 22M
    post = box(16.0 - 6, AIR_Y + DUCT_OD / 2 - 1, FLOOR, 12, 7, oz + 6 - FLOOR)
    body = fuse_all(outer, [spig, flare, collar, male, post])
    cuts = [inner, cyl(ox, spig_y0 - 1, oz, BL["out_id"] / 2, 86.0 - spig_y0 + 2, (0, 1, 0)),
            cone(ox, 82.0, oz, BL["out_id"] / 2, DUCT_ID / 2, 4.01, (0, 1, 0)),
            cyl(SFM_X0 - 5.0, AIR_Y, AIR_Z, DUCT_ID / 2, 20.0, (1, 0, 0)),
            cyl(16.0, AIR_Y + DUCT_OD / 2 + 2.5, FLOOR - 0.1, 2.0, 6.1)]             # insert from below
    return cut_all(body, cuts), length + (86.0 - spig_y0) + 18.0


def duct_hardware():
    ox, oz, _ = duct_path()
    y_out = BY + BL["out_reach"]
    sleeve = cyl(ox, y_out - 6.0, oz, 6.5, 6.0 + 4.0 + 6.0, (0, 1, 0)).cut(
        cyl(ox, y_out - 7.0, oz, BL["out_od"] / 2, 20, (0, 1, 0)))
    x, y = 16.0, AIR_Y + DUCT_OD / 2 + 2.5
    hw = compound([insert(x, y, FLOOR + 5.7), screw_csk(x, y, 0.0, 8.0)])
    fbom("duct", 1, "Ong silicone ID 9-10 x OD 13, dai 16 mm (noi ong ra blower <-> ong dan, cach rung)")
    fbom("duct", 1, "Insert nhiet M3 x 5.7 (chan ong dan) + vit M3 x 8 ISO 10642 tu duoi day")
    return sleeve, hw


# =========================================================== SFM3300 + dock
def sfm_solid():
    body = cyl(SFM_X0, AIR_Y, AIR_Z, SFM["body_d"] / 2, DOCK_X0 - SFM_X0, (1, 0, 0))
    body = body.cut(cyl(SFM_X0 - 1, AIR_Y, AIR_Z, SFM["cone_od"] / 2, 15.0 + 1, (1, 0, 0)))   # 22F socket
    body = body.cut(cyl(SFM_X0, AIR_Y, AIR_Z, 11.25 / 2, SFM["len"], (1, 0, 0)))
    male = cyl(DOCK_X0, AIR_Y, AIR_Z, SFM["cone_od"] / 2, SFM["cone"], (1, 0, 0)).cut(
        cyl(DOCK_X0 - 1, AIR_Y, AIR_Z, 7.5, SFM["cone"] + 2, (1, 0, 0)))                     # 22M / 15F
    return body.fuse(male)


DOCK_R = 14.0
TAP = dict(x=SFM_X1 + 0.85, z=70.0)    # in the 1.7 mm gap between the SFM cone and the tub nozzle


def dock_parts():
    """Dock tube fused to the partition wall (part of the base print)."""
    d = cyl(DOCK_X0, AIR_Y, AIR_Z, DOCK_R, EX1 + T - DOCK_X0, (1, 0, 0))
    ty = AIR_Y - math.sqrt(DOCK_R ** 2 - (TAP["z"] - AIR_Z) ** 2)
    tap = cyl(TAP["x"], ty + 1.0, TAP["z"], 1.7, ty + 1.0 - 82.0, (0, -1, 0))   # barb for 3 mm ID tube
    tap = tap.fuse(cyl(TAP["x"], ty + 2.0, TAP["z"], 3.0, 3.0, (0, -1, 0)))
    cuts = [cyl(DOCK_X0 - 1, AIR_Y, AIR_Z, SFM["cone_od"] / 2, SFM["cone"] + 1, (1, 0, 0)),   # 22F for SFM
            cyl(SFM_X1 - 0.01, AIR_Y, AIR_Z, 10.2, EX1 + T - SFM_X1 + 1, (1, 0, 0)),         # nozzle bore D20.4
            cyl(TAP["x"], 92.0, TAP["z"], 0.7, 92.0 - 81.0, (0, -1, 0))]                    # tap bore D1.4
    return d.fuse(tap), cuts


def cradle(cx):
    b = box(cx - 4, 92.0, 47.5, 8, D - T - 92.0 + 0.01, AIR_Z - 47.5)
    b = b.cut(cyl(cx - 5, AIR_Y, AIR_Z, SFM["body_d"] / 2 + 0.2, 10, (1, 0, 0)))
    return b.cut(box(cx - 2.5, 114.0, 47.0, 5, 1.8, 20))           # cable-tie slot


CRADLES_X = [50.0, 77.0]


# =========================================================== base
LID_BOSSES = [(T + 4, T + 4), (T + 4, D - T - 4), (EX1 + T / 2, T + 4), (EX1 + T / 2, D - T - 4),
              (W - T - 4, T + 4), (W - T - 4, D - T - 4)]


def text_solid(txt, size, depth):
    """Flat text in the XY plane at z=0, extruded +z. Needs a TTF (Arial Bold)."""
    try:
        try:
            chars = Part.makeWireString(txt, FONT, size, 0)
        except TypeError:
            chars = Part.makeWireString(txt, os.path.dirname(FONT), os.path.basename(FONT), size, 0)
        solids = []
        for wires in chars:
            if not wires:
                continue
            if isinstance(wires[0], Part.Edge):
                wires = [Part.Wire(es) for es in Part.sortEdges(wires)]
            face = Part.makeFace(wires, "Part::FaceMakerBullseye")
            solids.append(face.extrude(V(0, 0, depth)))
        s = Part.makeCompound(solids)
        bb = s.BoundBox
        s.translate(V(-bb.XMin - bb.XLength / 2, -bb.YMin - bb.YLength / 2, 0))
        return s
    except Exception as exc:  # noqa: BLE001
        log("text failed: %s" % exc)
        return None


def make_base():
    shell = round_vertical(box(0, 0, 0, W, D, Z_TOP), 8.0)
    shell = cut_all(shell, [box(EX0, T, FLOOR, E_W, D_IN, H_IN + 1),
                            box(UX0, T, FLOOR, U_W, D_IN, H_IN + 1)])
    add = []
    for hx, hy in PCB_HOLES:                                       # PCB standoffs
        x, y = pcb(hx, hy)
        add.append(cyl(x, y, FLOOR - 0.1, 4.0, PCB_BOT - FLOOR + 0.1))
    for x, y in LID_BOSSES:                                        # lid bosses
        add.append(cyl(x, y, Z_TOP - 10, 4.0, 10))
    add.append(box(UX0 + 10, TUB["y0"] + 8, FLOOR - 0.1, 80, 3, 6.1))         # tub rails (heater space)
    add.append(box(UX0 + 10, TUB["y0"] + TUB["d"] - 11, FLOOR - 0.1, 80, 3, 6.1))
    dock, dock_cuts = dock_parts()
    add.append(dock)
    add += [cradle(cx) for cx in CRADLES_X]
    base = fuse_all(shell, add)

    m = PLEN
    cuts = list(dock_cuts)
    for hx, hy in PCB_HOLES:
        x, y = pcb(hx, hy)
        cuts.append(cyl(x, y, PCB_BOT - 6.0, 2.0, 6.1))             # M3 insert D4.0 x 6
    for x, y in LID_BOSSES:
        cuts.append(cyl(x, y, Z_TOP - 8, 2.0, 8.1))
    jx, jy = pcb(0, 62.99)                                          # DC jack J1
    cuts.append(cyl(-1, jy, PCB_TOP + 6.5, 5.5, T + 2, (1, 0, 0)))
    for i in range(4):                                              # rear intake grille
        cuts.append(box(m["x0"] + 4 + i * 5.5, D - T - 1, FLOOR + 6, 3.5, T + 2, 30))
    cuts.append(box(UX1 - 1, TUB["y0"] - 0.5, FLOOR + 5.5, T + 2, TUB["d"] + 1, H_IN - 7))   # tub opening
    cuts.append(cyl(UX0 + 50, T + 20, -1, 2.0, FLOOR + 2))          # drain
    cuts.append(Part.makeSphere(1.6, V(TUB["x0"] + 84.0, TUB["y0"] + 9.5, FLOOR + 6.0)))   # tub detent dimple
    for x, y in [(15, 15), (W - 15, 15), (15, D - 15), (W - 15, D - 15)]:
        cuts.append(cyl(x, y, -0.1, 6.0, 0.7))                      # rubber feet recesses
    for x, y in FOOT_HOLES + HOOD_TABS + [(16.0, AIR_Y + DUCT_OD / 2 + 2.5)]:   # csk screws from below
        cuts.append(cyl(x, y, -1, 1.7, FLOOR + 2))
        cuts.append(cone(x, y, -0.01, 3.45, 1.5, 1.95))
    t = text_solid("SIPAP", 11.0, 0.8)                              # engraved on the front face
    if t is not None:
        t.rotate(V(0, 0, 0), V(1, 0, 0), 90)
        t.translate(V(UX0 + U_W / 2, 0.8 - 0.01, 30.0))
        cuts.append(t)
    return cut_all(base, cuts)


def base_hardware():
    parts = []
    for hx, hy in PCB_HOLES:
        x, y = pcb(hx, hy)
        parts.append(insert(x, y, PCB_BOT))
        parts.append(screw_button(x, y, PCB_TOP, 6.0))
    for x, y in LID_BOSSES:
        parts.append(insert(x, y, Z_TOP))
        parts.append(screw_csk(x, y, Z_TOP + LID_T, 8.0, up=False))
    fbom("pcb", 4, "Vit M3 x 6 dau tron ISO 7380 + 4 insert nhiet M3 x 5.7 (tru do PCB)")
    fbom("lid", 6, "Vit M3 x 8 dau bang ISO 10642 + 6 insert nhiet M3 x 5.7 (nap)")
    fbom("base", 4, "Chan cao su dan D12 x 3 mm")
    fbom("sfm", 2, "Day rut 2.5 x 100 mm (giu SFM3300 tren 2 gia)")
    return compound(parts)


# =========================================================== lid
OLED = dict(cx=29.0, cy=30.0, w=27.3, d=27.8, pcb_t=1.6, glass_t=1.5)
F1_BOX = (pcb(15.7, 66.6), pcb(30.1, 70.5))


def make_lid():
    lid = round_vertical(box(0, 0, Z_TOP, W, D, LID_T), 8.0)
    (fx0, fy0), (fx1, fy1) = F1_BOX
    lid = lid.fuse(box(fx0 - 2, fy0 - 2, Z_TOP, fx1 - fx0 + 4, fy1 - fy0 + 4, LID_T + 2.0)).removeSplitter()
    o = OLED
    posts = [cyl(o["cx"] + sx * (o["w"] / 2 - 2.0), o["cy"] + sy * (o["d"] / 2 - 2.0),
                 Z_TOP - o["glass_t"], 1.8, o["glass_t"] + 0.01)
             for sx in (-1, 1) for sy in (-1, 1)]
    lid = fuse_all(lid, posts)
    t = text_solid("SIPAP", 16.0, 0.8)
    if t is not None:
        t.translate(V(UX0 + U_W / 2, D - 28.0, Z_TOP + LID_T - 0.01))
        lid = lid.fuse(t)
    cuts = [box(fx0 - 0.5, fy0 - 0.5, Z_TOP - 1, fx1 - fx0 + 1, fy1 - fy0 + 1, LID_T + 1 + 1.4)]
    for x, y in LID_BOSSES:
        cuts.append(cyl(x, y, Z_TOP - 1, 1.7, LID_T + 2))
        cuts.append(cone(x, y, Z_TOP + LID_T + 0.01, 3.45, 1.5, 1.95, (0, 0, -1)))
    ex, ey = pcb(57.15, 14.22)
    cuts.append(cyl(ex, ey, Z_TOP - 1, 3.75, LID_T + 2))           # encoder shaft D6 through D7.5
    bx, by = pcb(43.69, 7.11)
    cuts.append(cyl(bx, by, Z_TOP - 1, 1.0, LID_T + 2))            # buzzer holes
    for k in range(6):
        a = k * math.pi / 3
        cuts.append(cyl(bx + 3.5 * math.cos(a), by + 3.5 * math.sin(a), Z_TOP - 1, 1.0, LID_T + 2))
    cuts.append(box(o["cx"] - 12, o["cy"] - 6.5, Z_TOP - 1, 24, 13, LID_T + 2))    # OLED window (AA 21.7 x 10.9)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cuts.append(cyl(o["cx"] + sx * (o["w"] / 2 - 2.0), o["cy"] + sy * (o["d"] / 2 - 2.0),
                            Z_TOP - o["glass_t"] - 0.1, 0.8, 2.5))
    return cut_all(lid, cuts)


def oled_solid():
    o = OLED
    zg = Z_TOP - o["glass_t"]
    board = box(o["cx"] - o["w"] / 2, o["cy"] - o["d"] / 2, zg - o["pcb_t"], o["w"], o["d"], o["pcb_t"])
    glass = box(o["cx"] - 13.25, o["cy"] - 9.5, zg, 26.5, 19.0, o["glass_t"])
    for sx in (-1, 1):
        for sy in (-1, 1):
            board = board.cut(cyl(o["cx"] + sx * (o["w"] / 2 - 2.0), o["cy"] + sy * (o["d"] / 2 - 2.0),
                                  zg - 3, 1.0, 4))
    screws = compound([screw_button(o["cx"] + sx * (o["w"] / 2 - 2.0), o["cy"] + sy * (o["d"] / 2 - 2.0),
                                    zg - o["pcb_t"], 4.0, d=2.0, down=False, core=1.6)
                       for sx in (-1, 1) for sy in (-1, 1)])
    fbom("oled", 4, "Vit tu ren M2 x 5 (module OLED vao 4 tru duoi nap) - DO LAI vi tri lo module that")
    return board.fuse(glass), screws


# =========================================================== mated connectors, cables, tubes
def housing(cx, cy, n_x, n_y):
    """Mated 2.54 mm crimp housing on the header, with its contact cavities."""
    w, d = 2.54 * n_x, 2.54 * n_y
    h = box(cx - w / 2, cy - d / 2, PCB_TOP + 2.54, w, d, 14.0)
    for i in range(n_x):
        for j in range(n_y):
            px, py = cx - w / 2 + 1.27 + 2.54 * i, cy - d / 2 + 1.27 + 2.54 * j
            h = h.cut(box(px - 0.6, py - 0.6, PCB_TOP + 2.0, 1.2, 1.2, 9.0))
    return h


HOUS_TOP = PCB_TOP + 2.54 + 14.0
J2_C = pcb(8.89, 12.70)
U4_C = pcb(88.90, 10.16)
U5_C = pcb(83.82, 69.85)
U2_C = pcb(90.42, 46.99)
U3_C = pcb(90.42, 29.21)
MPXV_TOP = PCB_TOP + 13.2          # BOM: 13.2 mm tall incl. port


def cables():
    zc = Z_TOP - CABLE_R - 0.3                                       # top cable layer under the lid
    zlo = HOUS_TOP + CABLE_R + 0.2                                   # lower layer, just above the housings
    out = {}
    # J2 <- blower: leaves the motor board on the outlet side (+y), out through the hood
    # grommet, up behind the PCB, forward along the left edge, down into J2.
    bx = BX_M + 1.6
    out["cable_j2_blower"] = [(bx, BY + BL["d"] / 2 + CABLE_R + 0.2, BZ), (bx, HOOD["y1"] - 6, BZ), (bx, HOOD["y1"] - 6, 48.0),
                              (bx, 84.0, 48.0), (6.0, 84.0, 48.0), (6.0, 84.0, zc), (6.0, J2_C[1] + 6, zc),
                              (J2_C[0], J2_C[1], zc), (J2_C[0], J2_C[1], HOUS_TOP + 0.3)]
    # U4 <- SFM3300 pads (wires soldered on top of the sensor body)
    sx = (SFM_X0 + DOCK_X0) / 2
    out["cable_u4_sfm"] = [(sx, AIR_Y, AIR_Z + SFM["body_d"] / 2 + CABLE_R + 0.2), (sx, 86.0, zc), (103.5, 86.0, zc),
                           (103.5, U4_C[1], zc), (U4_C[0], U4_C[1], zc), (U4_C[0], U4_C[1], HOUS_TOP + 0.3)]
    # U5 -> OLED (wires soldered to the module pins, under the module)
    zo = Z_TOP - OLED["glass_t"] - OLED["pcb_t"] - CABLE_R - 0.8
    out["cable_u5_oled"] = [(OLED["cx"], OLED["cy"] + OLED["d"] / 2 - 2.0, zo), (OLED["cx"], OLED["cy"] + OLED["d"] / 2 + 3.0, zlo),
                            (U5_C[0], U5_C[1], zlo), (U5_C[0], U5_C[1], HOUS_TOP + 0.3)]
    return out


def ptubes():
    ty = AIR_Y - math.sqrt(DOCK_R ** 2 - (TAP["z"] - AIR_Z) ** 2)
    zt = Z_TOP - PTUBE_OD / 2 - 0.3 - 2 * CABLE_R - 0.1
    tx = TAP["x"]
    ty0 = 43.0                                                     # T fitting centre (tx, ty0, zt)
    t_main = [(tx, 82.0, TAP["z"]), (tx, 79.0, TAP["z"]), (tx, 76.0, zt), (tx, ty0 + 4.0, zt)]
    t_u2 = [(tx - 4.0, ty0, zt), (U2_C[0], ty0, zt), (U2_C[0], U2_C[1], zt), (U2_C[0], U2_C[1], MPXV_TOP + 1.0)]
    t_u3 = [(tx, ty0 - 4.0, zt), (tx, U3_C[1], zt), (U3_C[0], U3_C[1], zt), (U3_C[0], U3_C[1], MPXV_TOP + 1.0)]
    return {"ptube_tap_T": t_main, "ptube_u2": t_u2, "ptube_u3": t_u3}, ty


# =========================================================== humidifier tub (no-spill)
def snorkel_paths():
    t, tin = TUB, TUB_IN
    pin = [(SFM_X1 + 1.7, AIR_Y, AIR_Z), (MOUTH_IN[0], AIR_Y, AIR_Z), (MOUTH_IN[0], MOUTH_IN[1], AIR_Z),
           (MOUTH_IN[0], MOUTH_IN[1], Z_MOUTH)]
    pout = [(MOUTH_OUT[0], MOUTH_OUT[1], Z_MOUTH), (MOUTH_OUT[0], MOUTH_OUT[1], AIR_Z), (MOUTH_OUT[0], TUB_OUT_Y, AIR_Z),
            (t["face"] + 18.0, TUB_OUT_Y, AIR_Z)]
    return pin, pout


def make_tub():
    t, tin = TUB, TUB_IN
    body = box(t["x0"], t["y0"], t["z0"], t["w"], t["d"], t["h"])
    body = body.cut(box(tin["x0"], tin["y0"], tin["z0"], tin["x1"] - tin["x0"], tin["y1"] - tin["y0"], t["h"]))
    lip = box(tin["x0"], tin["y0"], tin["z1"] - 4, tin["x1"] - tin["x0"], tin["y1"] - tin["y0"], 4).cut(
        box(tin["x0"] + LIP, tin["y0"] + LIP, tin["z1"] - 5, tin["x1"] - tin["x0"] - 2 * LIP, tin["y1"] - tin["y0"] - 2 * LIP, 6))
    face = box(t["x0"] + t["w"] - 0.01, t["y0"], t["z0"], t["face"] - t["x0"] - t["w"] + 0.01, t["d"], H_IN - 7.5 + FLOOR + 5.5 - t["z0"] - 0.5)
    pin, pout = snorkel_paths()
    s_in, _ = path_solid(pin, TUBE_OD / 2)
    nozzle = cyl(pin[0][0], AIR_Y, AIR_Z, 10.0, t["x0"] + t["wall"] - pin[0][0] + 0.5, (1, 0, 0))
    s_out, _ = path_solid(pout[:-1], TUBE_OD / 2)
    s_out = s_out.fuse(cyl(MOUTH_OUT[0], TUB_OUT_Y, AIR_Z, TUBE_OD / 2, t["face"] - MOUTH_OUT[0], (1, 0, 0)))
    cone22 = cone(t["face"], TUB_OUT_Y, AIR_Z, 11.0, 10.7, 18.0, (1, 0, 0))     # ISO 22M patient port
    latch_ledges = [box(TUB_C[0] - 6, t["y0"] - 1.0, t["z0"] + t["h"] - 8, 12, 1.0, 2.0),
                    box(TUB_C[0] - 6, t["y0"] + t["d"], t["z0"] + t["h"] - 8, 12, 1.0, 2.0)]
    detent = Part.makeSphere(1.5, V(t["x0"] + 84.0, t["y0"] + 9.5, t["z0"] + 0.3))
    seat = cyl(t["x0"] + 0.5, AIR_Y, AIR_Z, TUBE_ID / 2 + 0.5, 1.5, (1, 0, 0)).cut(
        cyl(t["x0"], AIR_Y, AIR_Z, 5.5, 3, (1, 0, 0)))                          # valve seat ring, opening D11
    max_rib = box(tin["x0"] + 10, tin["y0"] - 0.01, 0, 30, 1.2, 1.0)          # placed later at the MAX level
    body = fuse_all(body, [lip, face, s_in, nozzle, s_out, cone22, detent] + latch_ledges)
    bore_in, _ = path_solid([(pin[0][0] - 1, AIR_Y, AIR_Z)] + pin[1:-1] + [(pin[-1][0], pin[-1][1], Z_MOUTH - 1)], TUBE_ID / 2)
    bore_out, _ = path_solid([(MOUTH_OUT[0], MOUTH_OUT[1], Z_MOUTH - 1)] + pout[1:-1] + [(t["face"] + 19, TUB_OUT_Y, AIR_Z)], TUBE_ID / 2 - 0.5)
    cuts = [bore_in, bore_out,
            cyl(pin[0][0] + 1.5, AIR_Y, AIR_Z, 10.5, 2.0, (1, 0, 0)).cut(cyl(pin[0][0], AIR_Y, AIR_Z, 8.4, 5, (1, 0, 0))),  # O-ring groove
            # gasket groove (D2 silicone cord) in the lip top
            box(tin["x0"] + 0.5, tin["y0"] + 0.5, tin["z1"] - 1.5, tin["x1"] - tin["x0"] - 1, tin["y1"] - tin["y0"] - 1, 2).cut(
                box(tin["x0"] + 2.5, tin["y0"] + 2.5, tin["z1"] - 2, tin["x1"] - tin["x0"] - 5, tin["y1"] - tin["y0"] - 5, 3)),
            box(t["face"] - 1.5, t["y0"] + t["d"] / 2 - 20, t["z0"] + 6, 2, 40, 10)]            # finger grip
    t_txt = text_solid("MAX", 5.0, 0.6)
    if t_txt is not None:
        t_txt.rotate(V(0, 0, 0), V(0, 1, 0), 90)
        t_txt.translate(V(t["face"] - 0.59, t["y0"] + 60.0, 0))
    return cut_all(body, cuts).fuse(seat).removeSplitter(), max_rib, t_txt


def make_tub_lid():
    t, tin = TUB, TUB_IN
    lid = box(t["x0"], t["y0"], tin["z1"], t["w"] - 0.3, t["d"], 3.0)
    tongue = box(tin["x0"] + 1.0, tin["y0"] + 1.0, tin["z1"] - 1.0, tin["x1"] - tin["x0"] - 2, tin["y1"] - tin["y0"] - 2, 1.01).cut(
        box(tin["x0"] + 2.0, tin["y0"] + 2.0, tin["z1"] - 2, tin["x1"] - tin["x0"] - 4, tin["y1"] - tin["y0"] - 4, 4))
    arms = []
    for y0, s in ((t["y0"] - 2.5, 1), (t["y0"] + t["d"] + 1.0, -1)):           # snap latches
        arm = box(TUB_C[0] - 6, y0, t["z0"] + t["h"] - 9.5, 12, 1.5, 9.5 + 3.0)
        hook_y = y0 + 1.5 if s == 1 else y0 - 1.0
        hook = box(TUB_C[0] - 6, hook_y, t["z0"] + t["h"] - 9.5, 12, 1.0, 1.4)
        arms += [arm, hook]
    grip = box(t["x0"] + 20, t["y0"] + 40, tin["z1"] + 3.0, 20, 22, 2.0)
    return fuse_all(lid, [tongue, grip] + arms)


def tub_refs():
    fbom("tub", 1, "Day silicone D2 mm ~ 380 mm (gioang nap binh, ep vao ranh)")
    fbom("tub", 1, "O-ring 16 x 2 mm silicone (ong vao binh <-> dock)")
    fbom("tub", 1, "(tuy chon) van 1 chieu silicone dang la/duckbill D11-15 trong ong vao - chon loai ton ap < 0.3 cmH2O @ 60 L/ph")
    fbom("ptube", 1, "Ong silicone 3 x 5 mm ~ 140 mm + 1 cut chu T 3.2 mm (diem do ap -> U2/U3) + 3 day rut")


# =========================================================== checks
def overlap(a, b):
    try:
        if not a.BoundBox.intersect(b.BoundBox):
            return 0.0
        return round(a.common(b).Volume, 2)
    except Exception as exc:  # noqa: BLE001
        return "error: %s" % exc


def duct_dp(length_mm, bends, q_lpm=60.0, d_mm=DUCT_ID):
    a = math.pi * (d_mm / 1000 / 2) ** 2
    v = q_lpm / 60000.0 / a
    q = 0.5 * 1.2 * v * v
    re = v * d_mm / 1000 / 1.5e-5
    f = 0.316 / re ** 0.25
    return round((f * length_mm / d_mm + 0.5 * bends + 1.0) * q / 98.07, 2), round(v, 1)


def main():
    log("start")
    base = make_base(); log("base")
    lid = make_lid(); log("lid")
    hood = make_hood(); log("hood")
    bracket = make_bracket(); log("bracket")
    blower, outlet_end = blower_solid(); log("blower")
    duct, duct_len = make_duct(); log("duct")
    tub, max_rib, max_txt = make_tub(); log("tub")
    tub_lid = make_tub_lid(); log("tub_lid")
    sfm = sfm_solid()
    oled, oled_screws = oled_solid()
    brk_screws, washers, grommets, brk_floor = bracket_hardware()
    hood_hw = hood_hardware()
    sleeve, duct_hw = duct_hardware()
    base_hw = base_hardware()
    tub_refs()
    hous = {"housing_j2": housing(J2_C[0], J2_C[1], 2, 2), "housing_u4": housing(U4_C[0], U4_C[1], 6, 1),
            "housing_u5": housing(U5_C[0], U5_C[1], 4, 1)}
    cab_pts = cables()
    tube_pts, _ = ptubes()
    cab = {}
    lengths = {}
    for k, pts in list(cab_pts.items()) + list(tube_pts.items()):
        r = CABLE_R if k.startswith("cable") else PTUBE_OD / 2
        cab[k], lengths[k] = path_solid(pts, r)
    log("refs")

    # --- no-spill check: export the tub cavity geometry and run the voxel tilt check
    pin, pout = snorkel_paths()
    tilt_in = {"cavity": [TUB_IN["x0"], TUB_IN["y0"], TUB_IN["z0"], TUB_IN["x1"], TUB_IN["y1"], TUB_IN["z1"]],
               "lip": LIP, "lip_h": 4.0,
               "tubes": [{"pts": pin[1:], "r": TUBE_OD / 2},
                         {"pts": pout[:-1] + [(TUB_IN["x1"], TUB_OUT_Y, AIR_Z)], "r": TUBE_OD / 2}],
               "mouths": [{"c": [MOUTH_IN[0], MOUTH_IN[1], Z_MOUTH], "n": [0, 0, -1], "r": TUBE_ID / 2},
                          {"c": [MOUTH_OUT[0], MOUTH_OUT[1], Z_MOUTH], "n": [0, 0, -1], "r": TUBE_ID / 2 - 0.5}],
               "margin_mm": SLOSH_MARGIN}
    with open(os.path.join(OUT, "tilt_input.json"), "w") as f:
        json.dump(tilt_in, f, indent=1)
    tilt = None
    try:
        import tub_tilt_check
        tilt = tub_tilt_check.run(tilt_in)
        log("tilt %s" % tilt["v_max_ml"])
    except Exception as exc:  # noqa: BLE001
        log("tilt check failed: %s" % exc)
    fill_ml = int(tilt["v_fill_ml"]) if tilt else 150
    area = (TUB_IN["x1"] - TUB_IN["x0"]) * (TUB_IN["y1"] - TUB_IN["y0"])
    level = fill_ml * 1000.0 / area
    max_rib.translate(V(0, 0, TUB_IN["z0"] + level - 0.5))
    tub = tub.fuse(max_rib)
    if max_txt is not None:
        max_txt.translate(V(0, 0, TUB_IN["z0"] + level + 1.0))
        tub = tub.cut(max_txt)
    log("tub final")

    printable = {"base": base, "lid": lid, "hood": hood, "bracket": bracket, "duct": duct,
                 "tub": tub, "tub_lid": tub_lid}
    refs = {"ref_blower": blower, "ref_sfm3300": sfm, "ref_oled": oled, "ref_sleeve": sleeve,
            "hw_bracket_screws": brk_screws, "hw_washers": washers, "hw_grommets": grommets,
            "hw_floor_screws": compound([brk_floor, hood_hw, duct_hw]), "hw_base": base_hw, "hw_oled": oled_screws}
    refs.update(hous)
    refs.update(cab)
    pcb_shape = None
    if os.path.exists(PCB_STEP):
        pcb_shape = Part.read(PCB_STEP)
        pcb_shape.translate(V(GX0, GY0, PCB_TOP))   # Altium STEP: board top at z=0
        # the Altium STEP board has no drilled holes: cut the 4 mounting holes
        # (D3.2 = 126 mil from the pad list) so the M3 screws can be checked
        try:
            holes = compound([cyl(*pcb(hx, hy), PCB_BOT - 1, 1.6, PCB_T + 2) for hx, hy in PCB_HOLES])
            pcb_shape = pcb_shape.cut(holes)
        except Exception as exc:  # noqa: BLE001
            log("pcb hole cut failed: %s" % exc)
        refs["ref_pcb"] = pcb_shape
    for name, shp in printable.items():
        shp.exportStep(os.path.join(OUT, "sipap_%s.step" % name))
        shp.exportStl(os.path.join(OUT, "sipap_%s.stl" % name))
    for name, shp in refs.items():
        shp.exportStl(os.path.join(OUT, "%s.stl" % name))
    compound(list(printable.values()) + list(refs.values())).exportStep(os.path.join(OUT, "sipap_assembly.step"))
    log("exported")

    # --- interference: every pair (bounding-box prefilter), no exceptions list
    allp = dict(printable)
    allp.update(refs)
    names = list(allp)
    inter = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = allp[a], allp[b]
            if b == "ref_pcb" and a.startswith("housing_"):
                # Altium's STEP draws header pins as ONE solid bar (13.7 x 1 x 6 mm for U4),
                # so the pin zone inside the housing's own footprint is left out of this pair.
                bb = sa.BoundBox
                sa = sa.cut(box(bb.XMin - 0.1, bb.YMin - 0.1, PCB_TOP, bb.XLength + 0.2, bb.YLength + 0.2, 8.6))
            v = overlap(sa, sb)
            if v != 0.0:
                inter["%s|%s" % (a, b)] = v
        log("ovl " + a)
    ovl_bad = {}
    for k, v in inter.items():
        if isinstance(v, str) or v > 0.5:
            a, b = k.split("|")
            bb = allp[a].common(allp[b]).BoundBox
            ovl_bad[k] = [v, [round(c, 1) for c in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)]]

    dp, vel = duct_dp(duct_len, 4)
    rep = {"version": "0.3",
           "outer_mm": [round(W, 1), round(D, 1), round(Z_TOP + LID_T, 1)],
           "outer_with_patient_cone_mm": round(TUB["face"] + 18.0, 1),
           "z": {"blower_axis": BZ, "hood_top": HOOD_TOP, "pcb_bottom": PCB_BOT, "pcb_top": PCB_TOP,
                 "lid_underside": Z_TOP, "lid_top": Z_TOP + LID_T},
           "encoder_above_lid_mm": {"PEC11R-4220F (20 mm, BOM now)": round(PCB_TOP + 26.5 - Z_TOP - LID_T, 1),
                                    "PEC11R-4225F (25 mm)": round(PCB_TOP + 31.5 - Z_TOP - LID_T, 1)},
           "housing_top_to_lid_mm": round(Z_TOP - HOUS_TOP, 1),
           "blower_gaps_mm": {"inlet_stub_to_foam": round(HOOD["x1"] - HOOD_W - FOAM - (BX_M + BL["body"] + BL["stub_l"]), 1),
                              "outlet_end_to_spigot": round(HOOD["y1"] - 8.0 - outlet_end[1], 1)},
           "duct": {"length_mm": round(duct_len), "id_mm": DUCT_ID, "velocity_m_s_at_60lpm": vel,
                    "dp_cmH2O_at_60lpm_est": dp},
           "cable_lengths_mm": {k: round(v) for k, v in lengths.items()},
           "tub": {"inner_mm": [round(TUB_IN["x1"] - TUB_IN["x0"], 1), round(TUB_IN["y1"] - TUB_IN["y0"], 1),
                                round(TUB_IN["z1"] - TUB_IN["z0"], 1)],
                   "fill_ml_MAX": fill_ml, "max_level_above_floor_mm": round(level, 1), "tilt_check": tilt},
           "volumes_cm3": {k: round(v.Volume / 1000, 1) for k, v in printable.items()},
           "valid": {k: v.isValid() for k, v in printable.items()},
           "interference_mm3_nonzero": inter,
           "interference_fail": ovl_bad,
           "pairs_checked": len(names) * (len(names) - 1) // 2,
           "fasteners_bom": FASTENERS}
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
