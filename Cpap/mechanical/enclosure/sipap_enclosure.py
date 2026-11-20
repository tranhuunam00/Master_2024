"""SiPAP enclosure v0.1 — parametric FreeCAD model (run headless with freecadcmd).

    set SIPAP_OUT=<output folder>   (optional; default ./out next to this file)
    "D:\\Program Files\\FreeCAD 1.1\\bin\\freecadcmd.exe" sipap_enclosure.py

Global axes: x to the right, y to the back (front face at y=0), z up. Units: mm.

Three bays, left to right:
  1. Electronics: PCB Rev C on 4 tall M3 standoffs, 18 mm under the lid so the
     encoder shaft comes through the lid; DC jack J1 through the left wall.
  2. Blower: placeholder WM7040-class blower (70 x 70 x 40) on 4 rubber-grommet
     posts, rear intake grille + filter, SFM3300 flow sensor inline at the top.
  3. Humidifier: removable pass-over water tub (no heater in Rev C) sliding out
     to the right; 6 mm free under the tub for a heater plate in Rev D.
     Air enters the tub from the blower bay at the top (above the max water
     line, with a splash shelf) and leaves through a 22 mm ISO cone on the
     tub's right face, where the patient hose plugs in.

Placeholders (blower, SFM3300, PCB) are written to the assembly only, never
to the printable parts. Change the blower block once the real blower is known.
"""
import json
import os

import FreeCAD as App
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
T = 2.5            # wall thickness
FLOOR = 3.0        # floor thickness
H_IN = 76.0        # inner height floor -> lid
LID_T = 2.5
E_W = 103.6        # electronics bay inner width (PCB 101.6 + 2 x 1 mm)
B_W = 95.0         # blower bay inner width
U_W = 100.0        # humidifier bay inner width
D_IN = 120.0       # inner depth (front -> back)

PCB_W, PCB_D, PCB_T = 101.6, 76.2, 1.6
PCB_HOLES = [(3.81, 3.81), (97.79, 3.81), (3.81, 72.39), (97.79, 72.39)]
PCB_TO_LID = 18.0  # PCB top -> lid underside (encoder shaft 26.5 mm above PCB)

# placeholder blower (WM7040 class: L 70, H 40 per maker sheet)
BLOWER = dict(w=70.0, d=70.0, h=40.0, y0=45.0, grommet=5.0)
SFM = dict(len=69.3, dia=25.3)             # Sensirion SFM3300 body (DS v1.4 fig. 3)
WATER_MAX = 35.0                           # max fill height in the tub (mm)

# derived ---------------------------------------------------------------
W = 4 * T + E_W + B_W + U_W
D = D_IN + 2 * T
Z_TOP = FLOOR + H_IN                       # top of the walls = lid underside
EX0, EX1 = T, T + E_W                      # electronics bay inner x
BX0 = EX1 + T
BX1 = BX0 + B_W                            # blower bay inner x
UX0 = BX1 + T
UX1 = UX0 + U_W                            # humidifier bay inner x
PCB_TOP = Z_TOP - PCB_TO_LID
PCB_BOT = PCB_TOP - PCB_T
GX0, GY0 = T + 1.0, T + 2.0                # PCB origin (board lower-left) in global coords
AIR_Y, AIR_Z = 30.0, 63.0                  # SFM / tub-inlet axis
TUB = dict(x0=UX0 + 2.0, y0=T + 8.0, z0=FLOOR + 6.0, w=96.0, d=D_IN - 16.0, h=64.0, wall=2.5)


def pcb(x, y):
    """PCB coordinates (mm, board origin) -> global x, y."""
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


# ----------------------------------------------------------------- base
def make_base():
    shell = box(0, 0, 0, W, D, Z_TOP)
    try:  # round the four vertical outside edges
        vert = [e for e in shell.Edges if abs(e.Vertexes[0].Z - e.Vertexes[1].Z) > 1]
        shell = shell.makeFillet(6.0, vert)
    except Exception:
        pass
    shell = cut_all(shell, [
        box(EX0, T, FLOOR, E_W, D_IN, H_IN + 1),
        box(BX0, T, FLOOR, B_W, D_IN, H_IN + 1),
        box(UX0, T, FLOOR, U_W, D_IN, H_IN + 1),
    ])
    add = []
    # PCB standoffs (M3 self-tapping, pilot 2.5)
    for hx, hy in PCB_HOLES:
        x, y = pcb(hx, hy)
        add.append(cyl(x, y, FLOOR - 0.1, 4.0, PCB_BOT - FLOOR + 0.1))
    # hanging lid-screw bosses (top 10 mm only, so they clear the PCB parts)
    lid_bosses = [(T + 4, T + 4), (T + 4, D - T - 4),
                  (EX1 + T / 2, T + 4), (EX1 + T / 2, D - T - 4),
                  (W - T - 4, T + 4), (W - T - 4, D - T - 4)]
    for x, y in lid_bosses:
        add.append(cyl(x, y, Z_TOP - 10, 4.0, 10))
    # blower grommet posts
    bxc = (BX0 + BX1) / 2
    bpost = [(bxc - 30, BLOWER["y0"] + 5), (bxc + 30, BLOWER["y0"] + 5),
             (bxc - 30, BLOWER["y0"] + BLOWER["d"] - 5), (bxc + 30, BLOWER["y0"] + BLOWER["d"] - 5)]
    for x, y in bpost:
        add.append(cyl(x, y, FLOOR - 0.1, 4.0, BLOWER["grommet"] + 0.1))
    # tub rails: tub sits 6 mm up -> room for a Rev D heater plate underneath
    add.append(box(UX0 + 10, TUB["y0"] + 8, FLOOR - 0.1, 80, 3, 6.1))
    add.append(box(UX0 + 10, TUB["y0"] + TUB["d"] - 11, FLOOR - 0.1, 80, 3, 6.1))
    # filter retaining ribs behind the rear intake
    add.append(box(bxc - 36, D - T - 6, FLOOR - 0.1, 2, 6, 50))
    add.append(box(bxc + 34, D - T - 6, FLOOR - 0.1, 2, 6, 50))
    base = fuse_all(shell, add)

    cuts = []
    for hx, hy in PCB_HOLES:
        x, y = pcb(hx, hy)
        cuts.append(cyl(x, y, PCB_BOT - 12, 1.25, 12.1))
    for x, y in lid_bosses:
        cuts.append(cyl(x, y, Z_TOP - 8, 2.0, 8.1))          # M3 heat-set insert
    for x, y in bpost:
        cuts.append(cyl(x, y, FLOOR, 2.0, BLOWER["grommet"] + 0.1))
    # J1 DC jack through the left wall (PJ-102AH: barrel centre 6.5 mm above PCB)
    jx, jy = pcb(0, 62.99)
    cuts.append(cyl(-1, jy, PCB_TOP + 6.5, 5.5, T + 2, (1, 0, 0)))
    # cable notch + pressure-tube hole in the electronics/blower wall
    cuts.append(box(EX1 - 1, T + 8, Z_TOP - 8, T + 2, 14, 9))
    cuts.append(cyl(EX1 - 1, 45, 70, 2.5, T + 2, (1, 0, 0)))
    # air duct from the blower bay into the tub inlet
    cuts.append(cyl(BX1 - 1, AIR_Y, AIR_Z, 10.5, T + 2, (1, 0, 0)))
    # rear intake grille (blower bay)
    for i in range(7):
        cuts.append(box(bxc - 33 + i * 10, D - T - 1, 15, 5, T + 2, 30))
    # right side opening for the tub
    cuts.append(box(UX1 - 1, TUB["y0"] - 0.5, FLOOR + 5, T + 2, TUB["d"] + 1, H_IN - 7))
    # drain hole in the tub bay, rubber-foot recesses underneath
    cuts.append(cyl(UX0 + 50, T + 20, -1, 2.0, FLOOR + 2))
    for x, y in [(15, 15), (W - 15, 15), (15, D - 15), (W - 15, D - 15)]:
        cuts.append(cyl(x, y, -0.1, 6.0, 0.7))
    return cut_all(base, cuts), lid_bosses


# ----------------------------------------------------------------- lid
def make_lid(lid_bosses):
    lid = box(0, 0, Z_TOP, W, D, LID_T)
    try:
        vert = [e for e in lid.Edges if abs(e.Vertexes[0].Z - e.Vertexes[1].Z) > 1]
        lid = lid.makeFillet(6.0, vert)
    except Exception:
        pass
    # bump over the tall PTC fuse F1 (24.8 mm) — goes away with an SMD PTC in Rev D
    fx, fy = pcb(22.86, 68.58)
    lid = lid.fuse(box(fx - 10, fy - 4, Z_TOP, 20, 8, 11.5)).removeSplitter()
    cuts = [box(fx - 8.5, fy - 2.5, Z_TOP - 1, 17, 5, 10)]
    for x, y in lid_bosses:
        cuts.append(cyl(x, y, Z_TOP - 1, 1.7, LID_T + 2))        # M3 clearance
    ex, ey = pcb(57.15, 14.22)                                   # encoder U6 shaft
    cuts.append(cyl(ex, ey, Z_TOP - 1, 3.75, LID_T + 2))
    bx, by = pcb(43.69, 7.11)                                    # buzzer BZ1 sound holes
    cuts.append(cyl(bx, by, Z_TOP - 1, 1.0, LID_T + 2))
    for k in range(6):
        import math
        a = k * math.pi / 3
        cuts.append(cyl(bx + 3.5 * math.cos(a), by + 3.5 * math.sin(a), Z_TOP - 1, 1.0, LID_T + 2))
    ox, oy = pcb(25.0, 25.0)                                     # OLED 0.96" window (module under the lid)
    cuts.append(box(ox - 13.5, oy - 8, Z_TOP - 1, 27, 16, LID_T + 2))
    return cut_all(lid, cuts)


# ----------------------------------------------------------------- humidifier tub
def make_tub():
    t = TUB
    body = box(t["x0"], t["y0"], t["z0"], t["w"], t["d"], t["h"])
    body = body.cut(box(t["x0"] + t["wall"], t["y0"] + t["wall"], t["z0"] + t["wall"],
                        t["w"] - 2 * t["wall"], t["d"] - 2 * t["wall"], t["h"]))
    # inlet stub on the left face (push-fit into the duct, silicone O-ring)
    inlet = cyl(t["x0"] - 8, AIR_Y, AIR_Z, 10.0, 8 + t["wall"], (1, 0, 0))
    # 22 mm ISO 5356-1 male outlet cone on the right face (simplified as a cylinder)
    oy = t["y0"] + t["d"] - 22
    outlet = cyl(t["x0"] + t["w"] - t["wall"], oy, AIR_Z, 11.0, 22 + t["wall"], (1, 0, 0))
    # splash shelf under the inlet jet, max-fill rib on the outside
    shelf = box(t["x0"] + t["wall"], AIR_Y - 12, AIR_Z - 11, 20, 24, 2)
    fill = box(t["x0"] + t["w"], t["y0"] + 10, t["z0"] + t["wall"] + WATER_MAX, 0.8, 50, 0.8)
    body = fuse_all(body, [inlet, outlet, shelf, fill])
    body = cut_all(body, [cyl(t["x0"] - 9, AIR_Y, AIR_Z, 8.0, 12, (1, 0, 0)),
                          cyl(t["x0"] + t["w"] - 4, oy, AIR_Z, 9.5, 30, (1, 0, 0))])
    lid = box(t["x0"], t["y0"], t["z0"] + t["h"], t["w"], t["d"], 2.5)
    water_ml = (t["w"] - 2 * t["wall"]) * (t["d"] - 2 * t["wall"]) * WATER_MAX / 1000.0
    return body, lid, water_ml


# ----------------------------------------------------------------- placeholders
def make_placeholders():
    bxc = (BX0 + BX1) / 2
    zb = FLOOR + BLOWER["grommet"]
    blower = box(bxc - BLOWER["w"] / 2, BLOWER["y0"], zb, BLOWER["w"], BLOWER["d"], BLOWER["h"])
    blower = blower.fuse(cyl(bxc + 20, BLOWER["y0"], zb + 20, 7.0, 10, (0, -1, 0)))
    sfm = cyl(BX0 + 8, AIR_Y, AIR_Z, SFM["dia"] / 2, SFM["len"], (1, 0, 0))
    pcb_shape = None
    if os.path.exists(PCB_STEP):
        pcb_shape = Part.read(PCB_STEP)
        # Altium's STEP has the board TOP at z=0 (board solid only 0.39 mm thick,
        # from the layer-stack setting); parts are measured from the top, so
        # align z=0 with the real top of the 1.6 mm board.
        pcb_shape.translate(V(GX0, GY0, PCB_TOP))
    return blower, sfm, pcb_shape


def overlap(a, b):
    try:
        return round(a.common(b).Volume, 2)
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def main():
    log("start " + OUT)
    base, lid_bosses = make_base()
    log("base ok")
    lid = make_lid(lid_bosses)
    log("lid ok")
    tub, tub_lid, water_ml = make_tub()
    log("tub ok")
    blower, sfm, pcb_shape = make_placeholders()
    log("placeholders ok, pcb=%s" % (pcb_shape is not None))

    printable = {"base": base, "lid": lid, "tub": tub, "tub_lid": tub_lid}
    for name, shp in printable.items():
        shp.exportStep(os.path.join(OUT, f"sipap_{name}.step"))
        mesh_path = os.path.join(OUT, f"sipap_{name}.stl")
        shp.exportStl(mesh_path)

    asm = [base, lid, tub, tub_lid, blower, sfm] + ([pcb_shape] if pcb_shape else [])
    Part.makeCompound(asm).exportStep(os.path.join(OUT, "sipap_assembly.step"))
    blower.exportStl(os.path.join(OUT, "ref_blower.stl"))
    sfm.exportStl(os.path.join(OUT, "ref_sfm3300.stl"))
    if pcb_shape:
        pcb_shape.exportStl(os.path.join(OUT, "ref_pcb.stl"))

    rep = {
        "outer_mm": [round(W, 1), round(D, 1), round(Z_TOP + LID_T, 1)],
        "pcb_top_z": PCB_TOP, "tub_water_ml_at_max": round(water_ml),
        "volumes_cm3": {k: round(v.Volume / 1000, 1) for k, v in printable.items()},
        "valid": {k: v.isValid() for k, v in printable.items()},
        "pcb_bbox": None, "interference_mm3": {},
    }
    pairs = {"tub-base": (tub, base), "tub-lid": (tub, lid), "tub_lid-lid": (tub_lid, lid),
             "blower-base": (blower, base), "sfm-base": (sfm, base), "sfm-blower": (sfm, blower),
             "sfm-lid": (sfm, lid), "blower-tub": (blower, tub)}
    if pcb_shape:
        bb = pcb_shape.BoundBox
        rep["pcb_bbox"] = [round(v, 2) for v in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)]
        pairs.update({"pcb-base": (pcb_shape, base), "pcb-lid": (pcb_shape, lid)})
    log("exports ok")
    for k, (a, b) in pairs.items():
        log("overlap " + k)
        rep["interference_mm3"][k] = overlap(a, b)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)


main()
