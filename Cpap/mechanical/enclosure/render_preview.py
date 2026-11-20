"""Preview renders of the enclosure STLs (matplotlib, no CAD GUI needed).

python render_preview.py <out_dir>
"""
import json
import os
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"


def load_stl(path):
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        tris, cur = [], []
        for line in data.decode(errors="ignore").splitlines():
            p = line.split()
            if p and p[0] == "vertex":
                cur.append([float(v) for v in p[1:4]])
                if len(cur) == 3:
                    tris.append(cur)
                    cur = []
        return np.array(tris)
    n = struct.unpack("<I", data[80:84])[0]
    arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return arr["v"].reshape(-1, 3, 3).astype(float)


C = {  # name: (color, alpha)
    "sipap_base": ("#9aa7b8", 0.22), "sipap_lid": ("#c9d1dc", 0.35),
    "sipap_hood": ("#7b6fb0", 0.25), "sipap_bracket": ("#555555", 0.95), "sipap_duct": ("#d9b23a", 0.95),
    "sipap_tub": ("#5fa8d3", 0.40), "sipap_tub_lid": ("#8cc4e6", 0.35),
    "ref_blower": ("#e07a2f", 0.95), "ref_sfm3300": ("#3f9b6b", 0.95), "ref_pcb": ("#2f6b3a", 0.95),
    "ref_oled": ("#202020", 0.95), "ref_sleeve": ("#b33a3a", 0.95),
    "hw_bracket_screws": ("#222222", 1.0), "hw_washers": ("#111111", 1.0), "hw_grommets": ("#111111", 1.0),
    "hw_floor_screws": ("#333333", 1.0), "hw_base": ("#333333", 1.0), "hw_oled": ("#333333", 1.0),
    "housing_j2": ("#f2f2f2", 1.0), "housing_u4": ("#f2f2f2", 1.0), "housing_u5": ("#f2f2f2", 1.0),
    "cable_j2_blower": ("#d62728", 1.0), "cable_u4_sfm": ("#1f77b4", 1.0), "cable_u5_oled": ("#9467bd", 1.0),
    "ptube_tap_T": ("#ff7f0e", 1.0), "ptube_u2": ("#ff7f0e", 1.0), "ptube_u3": ("#ff7f0e", 1.0),
    "water": ("#1f6fd1", 0.35),
}
CACHE = {}


def shade(tris, color):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    k = 0.45 + 0.55 * np.abs(n @ light)
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return np.clip(rgb[None, :] * k[:, None], 0, 1)


def tris_of(name):
    if name not in CACHE:
        path = os.path.join(OUT, name + ".stl")
        CACHE[name] = load_stl(path) if os.path.exists(path) else None
    return CACHE[name]


def water_box(rep):
    t = rep["tub"]
    x0, y0, z0, x1, y1, _ = json.load(open(os.path.join(OUT, "tilt_input.json")))["cavity"]
    z1 = z0 + t["max_level_above_floor_mm"]
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
         [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4]]
    return v[np.array(f)]


def render(names, fname, elev, azim, title, box=None, extra=None, size=(12, 6.5)):
    fig = plt.figure(figsize=size, dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    allpts = []
    items = [(n, tris_of(n)) for n in names] + (extra or [])
    for name, tris in items:
        if tris is None:
            continue
        if box is not None:
            c = tris.mean(1)
            keep = np.all((c >= box[0]) & (c <= box[1]), axis=1)
            tris = tris[keep]
            if not len(tris):
                continue
        color, alpha = C[name]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=shade(tris, color), edgecolors="none", alpha=alpha))
        allpts.append(tris.reshape(-1, 3))
    pts = np.vstack(allpts)
    mn, mx = pts.min(0), pts.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    ax.set_box_aspect(mx - mn)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)


rep = json.load(open(os.path.join(OUT, "report.json")))
HW = ["hw_bracket_screws", "hw_washers", "hw_grommets", "hw_floor_screws", "hw_base", "hw_oled"]
WIRES = ["housing_j2", "housing_u4", "housing_u5", "cable_j2_blower", "cable_u4_sfm", "cable_u5_oled",
         "ptube_tap_T", "ptube_u2", "ptube_u3"]
AIR = ["sipap_bracket", "ref_blower", "ref_sleeve", "sipap_duct", "ref_sfm3300"]
inside = ["sipap_base", "sipap_hood", "sipap_tub", "sipap_tub_lid", "ref_pcb", "ref_oled"] + AIR + HW + WIRES
water = [("water", water_box(rep))]
v = rep["version"]
render(inside, "preview_open.png", 28, -60, "SiPAP v%s - lid removed (cables red/blue/purple, pressure tubes orange)" % v,
       extra=water)
render(inside + ["sipap_lid"], "preview_closed.png", 30, -58, "SiPAP v%s - closed, %s mm" % (v, " x ".join(str(round(x)) for x in rep["outer_mm"])))
render(["sipap_base", "sipap_hood", "sipap_tub"] + AIR + HW + ["cable_j2_blower", "ptube_tap_T"],
       "preview_air_path.png", 24, -50, "Air path without PCB: blower -> duct -> SFM3300 -> dock -> tub (hood transparent)",
       extra=water)
render(["sipap_hood", "sipap_bracket", "ref_blower", "ref_sleeve", "sipap_duct"] + HW[:4],
       "preview_blower_mount.png", 18, -130, "WS4540 on L-bracket: 4x ST2.2x13 + rubber washers, 2x M3 + grommets to floor",
       box=(np.array([0, 0, -2]), np.array([75, 115, 62])), size=(11, 7))
render(["sipap_tub", "sipap_tub_lid"], "preview_tub.png", 22, -35,
       "No-spill tub: both snorkel mouths at the centre, water at MAX %d mL" % rep["tub"]["fill_ml_MAX"],
       extra=water, size=(10, 7))
render(inside, "preview_top.png", 90, -90, "Top view (lid off)", extra=water)
print("ok")
