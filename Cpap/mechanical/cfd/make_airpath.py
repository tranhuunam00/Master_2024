"""Build the AIR volume of the SiPAP v0.3 air path (FreeCAD, headless) and export
it as a multi-region ASCII STL in metres for OpenFOAM snappyHexMesh.

Air path: blower outlet spigot (ID 8) -> printed duct (ID 12, 4 bends) -> 22M cone
-> SFM3300 channel (D11.25, DS 3.2; its internal flow conditioner is NOT modelled,
its datasheet pressure drop is added afterwards) -> SFM 22M cone bore -> dock gap ->
tub nozzle (valve-seat ring D11) -> inlet snorkel -> tub air space above the water
at the MAX line -> outlet snorkel -> 22 mm patient cone.

    set SIPAP_CFD_OUT=<ascii folder>
    freecadcmd make_airpath.py
Writes airpath.stl (regions inlet / outlet / water / walls) and airpath.json
(bounding box, a point inside the air, probe points, MAX water level).
"""
import json
import os
import sys

import FreeCAD as App  # noqa: F401
import Part
import Mesh
import MeshPart
from FreeCAD import Vector as V

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, os.environ.get("SIPAP_ENC_DIR", os.path.join(HERE, "..", "enclosure")))
import sipap_enclosure as E  # noqa: E402

OUT = os.environ.get("SIPAP_CFD_OUT", os.path.join(HERE, "case_geom"))
os.makedirs(OUT, exist_ok=True)
FILL_ML = float(os.environ.get("SIPAP_FILL_ML", "120"))


def build():
    cyl, cone, box, path = E.cyl, E.cone, E.box, E.path_solid
    ox, oz, pts = E.duct_path()
    y_in = E.HOOD["y1"] - 8.0                                    # spigot mouth = CFD inlet
    X0, AY, AZ = E.SFM_X0, E.AIR_Y, E.AIR_Z
    parts = [cyl(ox, y_in, oz, E.BL["out_id"] / 2, 82.0 - y_in + 0.01, (0, 1, 0)),
             cone(ox, 82.0, oz, E.BL["out_id"] / 2, E.DUCT_ID / 2, 4.0, (0, 1, 0)),
             path(pts, E.DUCT_ID / 2)[0],
             cyl(X0 - 4.5, AY, AZ, E.DUCT_ID / 2, 14.0 + 4.5, (1, 0, 0)),          # collar + 22M cone bore
             cyl(X0 + 14.0, AY, AZ, E.SFM["cone_od"] / 2, 1.0, (1, 0, 0)),          # bottom of SFM 22F socket
             cyl(X0 + 15.0, AY, AZ, 11.25 / 2, E.DOCK_X0 - X0 - 15.0, (1, 0, 0)),   # SFM channel
             cyl(E.DOCK_X0 - 0.01, AY, AZ, 7.5, E.SFM["cone"] + 0.02, (1, 0, 0)),   # SFM 22M/15F bore
             cyl(E.SFM_X1, AY, AZ, 10.2, 1.7, (1, 0, 0))]                            # dock gap (pressure tap here)
    pin, pout = E.snorkel_paths()
    t, tin = E.TUB, E.TUB_IN
    bore_in = path([(pin[0][0] - 0.01, AY, AZ)] + pin[1:], E.TUBE_ID / 2)[0]
    seat = cyl(t["x0"] + 0.5, AY, AZ, E.TUBE_ID / 2 + 0.5, 1.5, (1, 0, 0)).cut(cyl(t["x0"], AY, AZ, 5.5, 3, (1, 0, 0)))
    parts.append(bore_in.cut(seat))
    x_out = t["face"] + 18.0
    parts.append(path(pout[:-1] + [(x_out, E.TUB_OUT_Y, AZ)], E.TUBE_ID / 2 - 0.5)[0])
    area = (tin["x1"] - tin["x0"]) * (tin["y1"] - tin["y0"])
    z_w = tin["z0"] + FILL_ML * 1000.0 / area                     # water surface at MAX
    cav = box(tin["x0"], tin["y0"], z_w, tin["x1"] - tin["x0"], tin["y1"] - tin["y0"], tin["z1"] - z_w)
    lip = box(tin["x0"], tin["y0"], tin["z1"] - 4, tin["x1"] - tin["x0"], tin["y1"] - tin["y0"], 4).cut(
        box(tin["x0"] + E.LIP, tin["y0"] + E.LIP, tin["z1"] - 5, tin["x1"] - tin["x0"] - 2 * E.LIP,
            tin["y1"] - tin["y0"] - 2 * E.LIP, 6))
    tubes = path(pin, E.TUBE_OD / 2)[0].fuse(path(pout[:-1] + [(tin["x1"] + 1, E.TUB_OUT_Y, AZ)], E.TUBE_OD / 2)[0])
    cav = cav.cut(lip).cut(tubes)
    fluid = cav
    for p in parts:
        fluid = fluid.fuse(p)
    fluid = fluid.removeSplitter()
    sol = fluid.Solids
    sol.sort(key=lambda s: -s.Volume)
    return sol[0], len(sol), y_in, x_out, z_w, (ox, oz), pin, pout


def classify(fluid, y_in, x_out, z_w):
    groups = {"inlet": [], "outlet": [], "water": [], "walls": []}
    for f in fluid.Faces:
        bb = f.BoundBox
        if bb.YLength < 1e-6 and abs(bb.YMin - y_in) < 1e-4:
            groups["inlet"].append(f)
        elif bb.XLength < 1e-6 and abs(bb.XMin - x_out) < 1e-4:
            groups["outlet"].append(f)
        elif bb.ZLength < 1e-6 and abs(bb.ZMin - z_w) < 1e-4:
            groups["water"].append(f)
        else:
            groups["walls"].append(f)
    return groups


def write_stl(groups, path):
    lines = []
    tmp = os.path.join(OUT, "_tmp.ast")
    for name, faces in groups.items():
        m = MeshPart.meshFromShape(Shape=Part.makeCompound(faces), LinearDeflection=0.03,
                                   AngularDeflection=0.12, Relative=False)
        m.transform(App.Matrix(0.001, 0, 0, 0, 0, 0.001, 0, 0, 0, 0, 0.001, 0))   # mm -> m
        m.write(tmp)
        txt = open(tmp).read().splitlines()
        txt[0] = "solid " + name
        txt[-1] = "endsolid " + name
        lines += txt
    os.remove(tmp)
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    fluid, nsol, y_in, x_out, z_w, (ox, oz), pin, pout = build()
    groups = classify(fluid, y_in, x_out, z_w)
    write_stl(groups, os.path.join(OUT, "airpath.stl"))
    fluid.exportStep(os.path.join(OUT, "airpath.step"))
    bb = fluid.BoundBox
    mi, mo = pin[-1], pout[0]
    probes = {  # mm, on the centre line of each section
        "p01_inlet_spigot": (ox, y_in + 2.0, oz), "p02_duct_riser": (16.0, E.AIR_Y, 40.0),
        "p03_before_sfm": (E.SFM_X0 + 8.0, E.AIR_Y, E.AIR_Z), "p04_sfm_mid": ((E.SFM_X0 + E.DOCK_X0) / 2, E.AIR_Y, E.AIR_Z),
        "p05_after_sfm": (E.SFM_X1 - 2.0, E.AIR_Y, E.AIR_Z), "p06_nozzle": (E.TUB["x0"] - 3.0, E.AIR_Y, E.AIR_Z),
        "p07_inlet_snorkel": (mi[0], mi[1], mi[2] + 6.0), "p08_tub_air": (E.TUB_C[0], E.TUB_C[1] + 30.0, z_w + 8.0),
        "p09_outlet_snorkel": (mo[0], mo[1], mo[2] + 6.0), "p10_outlet_tube": (190.0, E.TUB_OUT_Y, E.AIR_Z),
        "p11_patient_cone": (x_out - 3.0, E.TUB_OUT_Y, E.AIR_Z)}
    for k, p in probes.items():
        if not fluid.isInside(V(*p), 1e-6, True):
            raise RuntimeError("probe outside the air: %s %s" % (k, p))
    inside = (E.TUB_C[0], E.TUB_C[1] + 30.0, z_w + 15.0)
    info = {"n_solids_after_fuse": nsol, "volume_ml": round(fluid.Volume / 1000, 1),
            "bbox_mm": [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax],
            "location_in_mesh_mm": inside, "fill_ml": FILL_ML, "water_z_mm": round(z_w, 2),
            "inlet_area_mm2": round(sum(f.Area for f in groups["inlet"]), 2),
            "outlet_area_mm2": round(sum(f.Area for f in groups["outlet"]), 2),
            "faces": {k: len(v) for k, v in groups.items()}, "probes_mm": probes,
            "valid": fluid.isValid()}
    json.dump(info, open(os.path.join(OUT, "airpath.json"), "w"), indent=1)


main()
