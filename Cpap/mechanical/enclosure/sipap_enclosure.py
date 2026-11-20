"""SiPAP enclosure v0.2 — compact, quiet layout (parametric FreeCAD, run headless).

    set SIPAP_OUT=<ascii output folder>      (optional; default ./out)
    "D:\\Program Files\\FreeCAD 1.1\\bin\\freecadcmd.exe" sipap_enclosure.py

Axes: x right, y back (front face y=0), z up. Units mm.

Two bays instead of three (v0.1 was 309 mm wide, this one ~213 mm):
  LEFT  - PCB on tall standoffs near the lid (encoder through the lid).
          UNDER the PCB: a closed muffler box with the blower on rubber
          grommets, 2-3 mm foam all round, and an intake plenum with two
          baffles (labyrinth) behind it; air enters through a filter pad and
          the rear grille. The blower outlet goes up through the muffler lid,
          by silicone hose, to the SFM3300 behind the PCB, then through the
          wall into the humidifier tub (top, above the water line).
  RIGHT - removable pass-over humidifier tub (Rev C: no heater), sliding out
          to the right, 22 mm ISO outlet cone; 6 mm left under it for a
          Rev D heater plate. The tub volume also works as an outlet muffler.
Placeholders (blower, SFM3300, PCB STEP) go only into the assembly.
"""
import json
import math
import os

import FreeCAD as App  # noqa: F401  (needed for the Part kernel)
import Part
from FreeCAD import Vector as V

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
OUT = os.environ.get("SIPAP_OUT", os.path.join(HERE, "out"))
PCB_STEP = os.environ.get(
    "SIPAP_PCB_STEP",
    os.path.join(HERE, "..", "..", "hardware", "CPAP_VSSM", "3D", "SiPAP_PCB_RevC.step"))
os.makedirs(OUT, exist_ok=True)
_LOG = open(os.path.join(OUT, "progress.log"), "w")


def log(msg):
    _LOG.write(msg + "\n")
    _LOG.flush()


# ----------------------------------------------------------------- parameters
T = 3.0            # outer wall (3 mm: stiffer, quieter)
FLOOR = 3.0
H_IN = 76.0        # floor -> lid
LID_T = 3.0
E_W = 103.6        # left bay inner width (PCB 101.6 + 2 x 1)
U_W = 100.0        # humidifier bay inner width
D_IN = 118.0       # inner depth

PCB_W, PCB_D, PCB_T = 101.6, 76.2, 1.6
PCB_HOLES = [(3.81, 3.81), (97.79, 3.81), (3.81, 72.39), (97.79, 72.39)]
PCB_TO_LID = 18.0  # encoder shaft (26.5 mm above PCB) -> ~5.5 mm above the lid

BLOWER = dict(w=70.0, d=70.0, h=40.0)        # WM7040-class placeholder (maker: L70 H40)
GROMMET = 3.0                                 # blower on rubber grommets inside the box
SFM = dict(len=69.3, dia=25.3)               # Sensirion SFM3300 (DS v1.4 sec. 3)
WATER_MAX = 35.0
MUF_WALL = 2.0

# derived ---------------------------------------------------------------
W = 3 * T + E_W + U_W
D = D_IN + 2 * T
Z_TOP = FLOOR + H_IN
EX0, EX1 = T, T + E_W
UX0, UX1 = EX1 + T, EX1 + T + U_W
PCB_TOP = Z_TOP - PCB_TO_LID
PCB_BOT = PCB_TOP - PCB_T
GX0, GY0 = T + 1.0, T + 2.0
AIR_Y, AIR_Z = 100.0, 63.0                    # SFM / duct / tub-inlet axis (behind the PCB)

# muffler box under the PCB (between the PCB standoffs in x)
MUF = dict(x0=14.0, x1=94.0, y0=8.0, y1=86.0, z0=FLOOR + 2.0, top=53.0,   # blower chamber
           py1=116.0, ptop=47.0)                                            # intake plenum (behind)
TUB = dict(x0=UX0 + 2.0, y0=T + 8.0, z0=FLOOR + 6.0, w=96.0, d=D_IN - 16.0, h=64.0, wall=2.5)


def pcb(x, y):
    return GX0 + x, GY0 + y


def box(x, y, z, dx, dy, dz):
    return Part.makeBox(dx, dy, dz, V(x, y, z))


def cyl(x, y, z, r, h, d=(0, 0, 1)):
    return Part.makeCylinder(r, h, V(x, y, z), V(*d))


def fuse_all(base, parts):
    for p in parts:
        base = base.fuse(p)
    return base.removeSplitter()


def cut_all(base, parts):
    for p in parts:
        base = base.cut(p)
    return base


def round_vertical(shape, r):
    try:
        vert = [e for e in shape.Edges if len(e.Vertexes) == 2 and abs(e.Vertexes[0].Z - e.Vertexes[1].Z) > 1]
        return shape.makeFillet(r, vert)
    except Exception:  # noqa: BLE001
        return shape


LID_BOSSES = [(T + 4, T + 4), (T + 4, D - T - 4), (EX1 + T / 2, T + 4), (EX1 + T / 2, D - T - 4),
              (W - T - 4, T + 4), (W - T - 4, D - T - 4)]


# ----------------------------------------------------------------- base
def make_base():
    shell = round_vertical(box(0, 0, 0, W, D, Z_TOP), 8.0)
    shell = cut_all(shell, [box(EX0, T, FLOOR, E_W, D_IN, H_IN + 1),
                            box(UX0, T, FLOOR, U_W, D_IN, H_IN + 1)])
    add = []
    for hx, hy in PCB_HOLES:                                   # PCB standoffs
        x, y = pcb(hx, hy)
        add.append(cyl(x, y, FLOOR - 0.1, 4.0, PCB_BOT - FLOOR + 0.1))
    for x, y in LID_BOSSES:                                    # hanging lid bosses
        add.append(cyl(x, y, Z_TOP - 10, 4.0, 10))
    m = MUF                                                    # rubber-foot posts under the muffler
    feet = [(m["x0"] + 8, m["y0"] + 8), (m["x1"] - 8, m["y0"] + 8),
            (m["x0"] + 8, m["py1"] - 8), (m["x1"] - 8, m["py1"] - 8)]
    for x, y in feet:
        add.append(cyl(x, y, FLOOR - 0.1, 3.0, 2.1))
    add.append(box(UX0 + 10, TUB["y0"] + 8, FLOOR - 0.1, 80, 3, 6.1))   # tub rails (heater space)
    add.append(box(UX0 + 10, TUB["y0"] + TUB["d"] - 11, FLOOR - 0.1, 80, 3, 6.1))
    base = fuse_all(shell, add)

    cuts = []
    for hx, hy in PCB_HOLES:
        x, y = pcb(hx, hy)
        cuts.append(cyl(x, y, PCB_BOT - 12, 1.25, 12.1))
    for x, y in LID_BOSSES:
        cuts.append(cyl(x, y, Z_TOP - 8, 2.0, 8.1))             # M3 heat-set insert
    jx, jy = pcb(0, 62.99)                                      # DC jack J1, left wall
    cuts.append(cyl(-1, jy, PCB_TOP + 6.5, 5.5, T + 2, (1, 0, 0)))
    cuts.append(cyl(EX1 - 1, AIR_Y, AIR_Z, 10.5, T + 2, (1, 0, 0)))     # duct into the tub
    for i in range(6):                                          # rear intake grille (behind filter pad)
        cuts.append(box(m["x0"] + 8 + i * 11, D - T - 1, 12, 6, T + 2, 28))
    cuts.append(box(UX1 - 1, TUB["y0"] - 0.5, FLOOR + 5, T + 2, TUB["d"] + 1, H_IN - 7))  # tub opening
    cuts.append(cyl(UX0 + 50, T + 20, -1, 2.0, FLOOR + 2))      # drain
    for x, y in [(15, 15), (W - 15, 15), (15, D - 15), (W - 15, D - 15)]:
        cuts.append(cyl(x, y, -0.1, 6.0, 0.7))                  # rubber feet recesses
    return cut_all(base, cuts)


# ----------------------------------------------------------------- muffler (printed, separate part)
def make_muffler():
    m, w = MUF, MUF_WALL
    ch = box(m["x0"], m["y0"], m["z0"], m["x1"] - m["x0"], m["y1"] - m["y0"], m["top"] - m["z0"])
    pl = box(m["x0"], m["y1"] - w, m["z0"], m["x1"] - m["x0"], m["py1"] - m["y1"] + w, m["ptop"] - m["z0"])
    body = ch.fuse(pl).removeSplitter()
    body = cut_all(body, [
        box(m["x0"] + w, m["y0"] + w, m["z0"] + w, m["x1"] - m["x0"] - 2 * w, m["y1"] - m["y0"] - 2 * w, 60),
        box(m["x0"] + w, m["y1"], m["z0"] + w, m["x1"] - m["x0"] - 2 * w, m["py1"] - m["y1"] - w, 60),
    ])
    cx = (m["x0"] + m["x1"]) / 2
    # labyrinth: two baffles, gaps at alternate ends (top / bottom)
    b1 = box(m["x0"] + w, m["y1"] + 9, m["z0"], m["x1"] - m["x0"] - 2 * w, 2, (m["ptop"] - m["z0"]) - 10)
    b2 = box(m["x0"] + w, m["y1"] + 19, m["z0"] + 10, m["x1"] - m["x0"] - 2 * w, 2, (m["ptop"] - m["z0"]) - 10)
    posts = []
    bx0, by0 = cx - BLOWER["w"] / 2, (m["y0"] + m["y1"]) / 2 - BLOWER["d"] / 2
    for x, y in [(bx0 + 5, by0 + 5), (bx0 + BLOWER["w"] - 5, by0 + 5),
                 (bx0 + 5, by0 + BLOWER["d"] - 5), (bx0 + BLOWER["w"] - 5, by0 + BLOWER["d"] - 5)]:
        posts.append(cyl(x, y, m["z0"] + w - 0.1, 3.0, GROMMET + 0.1))
    body = fuse_all(body, [b1, b2] + posts)
    cuts = [cyl(cx, m["y1"] - w - 0.5, (m["z0"] + m["ptop"]) / 2, 9.0, w + 1, (0, 1, 0)),   # plenum -> chamber
            ]
    for i in range(6):                                         # plenum rear slots (filter pad behind)
        cuts.append(box(m["x0"] + 8 + i * 11, m["py1"] - w - 0.5, m["z0"] + 7, 6, w + 1, 28))
    for x, y in [(bx0 + 5, by0 + 5), (bx0 + BLOWER["w"] - 5, by0 + 5),
                 (bx0 + 5, by0 + BLOWER["d"] - 5), (bx0 + BLOWER["w"] - 5, by0 + BLOWER["d"] - 5)]:
        cuts.append(cyl(x, y, m["z0"] + w, 1.6, GROMMET + 0.2))
    body = cut_all(body, cuts)
    # lids: chamber lid with the blower-outlet pass-through (grommeted hose), plenum lid
    lid = box(m["x0"], m["y0"], m["top"], m["x1"] - m["x0"], m["y1"] - m["y0"], 2.0)
    lid = lid.cut(cyl(cx + 20, by0 + 15, m["top"] - 1, 9.0, 4))
    plid = box(m["x0"], m["y1"] - w, m["ptop"], m["x1"] - m["x0"], m["py1"] - m["y1"] + w, 2.0)
    blower = box(bx0, by0, m["z0"] + w + GROMMET, BLOWER["w"], BLOWER["d"], BLOWER["h"])
    blower = blower.fuse(cyl(cx + 20, by0 + 15, m["z0"] + w + GROMMET + BLOWER["h"], 7.0, 6))
    return body, lid.fuse(plid).removeSplitter(), blower


# ----------------------------------------------------------------- lid
def make_lid():
    lid = round_vertical(box(0, 0, Z_TOP, W, D, LID_T), 8.0)
    fx, fy = pcb(22.86, 68.58)                                 # bump over tall PTC F1
    lid = lid.fuse(box(fx - 10, fy - 4, Z_TOP, 20, 8, 11.5)).removeSplitter()
    cuts = [box(fx - 8.5, fy - 2.5, Z_TOP - 1, 17, 5, 10)]
    for x, y in LID_BOSSES:
        cuts.append(cyl(x, y, Z_TOP - 1, 1.7, LID_T + 2))
    ex, ey = pcb(57.15, 14.22)
    cuts.append(cyl(ex, ey, Z_TOP - 1, 3.75, LID_T + 2))       # encoder
    bx, by = pcb(43.69, 7.11)
    cuts.append(cyl(bx, by, Z_TOP - 1, 1.0, LID_T + 2))        # buzzer holes
    for k in range(6):
        a = k * math.pi / 3
        cuts.append(cyl(bx + 3.5 * math.cos(a), by + 3.5 * math.sin(a), Z_TOP - 1, 1.0, LID_T + 2))
    ox, oy = pcb(25.0, 25.0)
    cuts.append(box(ox - 13.5, oy - 8, Z_TOP - 1, 27, 16, LID_T + 2))   # OLED window
    return cut_all(lid, cuts)


# ----------------------------------------------------------------- humidifier tub
def make_tub():
    t = TUB
    body = box(t["x0"], t["y0"], t["z0"], t["w"], t["d"], t["h"])
    body = body.cut(box(t["x0"] + t["wall"], t["y0"] + t["wall"], t["z0"] + t["wall"],
                        t["w"] - 2 * t["wall"], t["d"] - 2 * t["wall"], t["h"]))
    inlet = cyl(t["x0"] - 8, AIR_Y, AIR_Z, 10.0, 8 + t["wall"], (1, 0, 0))
    oy = t["y0"] + 22                                          # outlet front-right, diagonal to the inlet
    outlet = cyl(t["x0"] + t["w"] - t["wall"], oy, AIR_Z, 11.0, 22 + t["wall"], (1, 0, 0))
    shelf = box(t["x0"] + t["wall"], AIR_Y - 12, AIR_Z - 11, 20, 24, 2)
    fill = box(t["x0"] + t["w"], t["y0"] + 40, t["z0"] + t["wall"] + WATER_MAX, 0.8, 50, 0.8)
    body = fuse_all(body, [inlet, outlet, shelf, fill])
    body = cut_all(body, [cyl(t["x0"] - 9, AIR_Y, AIR_Z, 8.0, 12, (1, 0, 0)),
                          cyl(t["x0"] + t["w"] - 4, oy, AIR_Z, 9.5, 30, (1, 0, 0))])
    lid = box(t["x0"], t["y0"], t["z0"] + t["h"], t["w"], t["d"], 2.5)
    ml = (t["w"] - 2 * t["wall"]) * (t["d"] - 2 * t["wall"]) * WATER_MAX / 1000.0
    return body, lid, ml


def make_refs():
    sfm = cyl(EX1 - 7.3 - SFM["len"], AIR_Y, AIR_Z, SFM["dia"] / 2, SFM["len"], (1, 0, 0))
    pcb_shape = None
    if os.path.exists(PCB_STEP):
        pcb_shape = Part.read(PCB_STEP)
        # Altium STEP: board top at z=0 (board solid 0.39 mm from the layer-stack
        # setting); parts are measured from the top, so align z=0 with the real top.
        pcb_shape.translate(V(GX0, GY0, PCB_TOP))
    return sfm, pcb_shape


def overlap(a, b):
    try:
        return round(a.common(b).Volume, 2)
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def main():
    log("start")
    base = make_base(); log("base")
    lid = make_lid(); log("lid")
    muffler, muffler_lid, blower = make_muffler(); log("muffler")
    tub, tub_lid, water_ml = make_tub(); log("tub")
    sfm, pcb_shape = make_refs(); log("refs pcb=%s" % (pcb_shape is not None))

    printable = {"base": base, "lid": lid, "muffler": muffler, "muffler_lid": muffler_lid,
                 "tub": tub, "tub_lid": tub_lid}
    for name, shp in printable.items():
        shp.exportStep(os.path.join(OUT, f"sipap_{name}.step"))
        shp.exportStl(os.path.join(OUT, f"sipap_{name}.stl"))
    refs = [blower, sfm] + ([pcb_shape] if pcb_shape else [])
    Part.makeCompound(list(printable.values()) + refs).exportStep(os.path.join(OUT, "sipap_assembly.step"))
    blower.exportStl(os.path.join(OUT, "ref_blower.stl"))
    sfm.exportStl(os.path.join(OUT, "ref_sfm3300.stl"))
    if pcb_shape:
        pcb_shape.exportStl(os.path.join(OUT, "ref_pcb.stl"))
    log("exported")

    rep = {"version": "0.2",
           "outer_mm": [round(W, 1), round(D, 1), round(Z_TOP + LID_T, 1)],
           "pcb_top_z": PCB_TOP, "encoder_above_lid_mm": round(PCB_TOP + 26.5 - Z_TOP - LID_T, 1),
           "tub_water_ml_at_max": round(water_ml),
           "muffler_top_to_pcb_bottom_mm": round(PCB_BOT - MUF["top"] - 2.0, 1),
           "volumes_cm3": {k: round(v.Volume / 1000, 1) for k, v in printable.items()},
           "valid": {k: v.isValid() for k, v in printable.items()},
           "interference_mm3": {}}
    pairs = {"tub-base": (tub, base), "tub-lid": (tub, lid), "tub_lid-lid": (tub_lid, lid),
             "muffler-base": (muffler, base), "muffler_lid-base": (muffler_lid, base),
             "blower-muffler": (blower, muffler), "blower-muffler_lid": (blower, muffler_lid),
             "sfm-base": (sfm, base), "sfm-lid": (sfm, lid), "sfm-muffler": (sfm, muffler),
             "sfm-muffler_lid": (sfm, muffler_lid), "sfm-tub": (sfm, tub)}
    if pcb_shape:
        pairs.update({"pcb-base": (pcb_shape, base), "pcb-lid": (pcb_shape, lid),
                      "pcb-muffler_lid": (pcb_shape, muffler_lid), "pcb-sfm": (pcb_shape, sfm)})
    for k, (a, b) in pairs.items():
        log("overlap " + k)
        rep["interference_mm3"][k] = overlap(a, b)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log("done")


main()
