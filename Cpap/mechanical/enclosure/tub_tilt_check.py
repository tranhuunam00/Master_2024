"""No-spill check for the SiPAP humidifier tub (static, voxel based).

For every direction of gravity (Fibonacci sphere + the 6 axes + 45-degree
tilts) the water settles in the lowest part of the tub cavity. The tub is
"no-spill" for a fill volume V when, for EVERY direction, the water surface
stays at least `margin_mm` below every point of both snorkel mouths (inlet
and outlet), so no water can run back into the device or out of the patient
port, at a 90-degree tilt in any direction and upside down.

Result: v_max_ml = the largest fill that passes in all directions;
v_fill_ml = that, rounded down to 10 mL = the MAX line moulded in the tub.

    python tub_tilt_check.py out/tilt_input.json
(the FreeCAD script calls run() itself and stores the result in report.json)
"""
import json
import math
import sys

import numpy as np


def _capsule_mask(p, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0.0, 1.0)
    d = p - (a + t[:, None] * ab)
    return np.einsum("ij,ij->i", d, d) < r * r


def cavity_points(cfg, step):
    x0, y0, z0, x1, y1, z1 = cfg["cavity"]
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    zs = np.arange(z0 + step / 2, z1, step)
    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    keep = np.ones(len(g), bool)
    lip, lh = cfg.get("lip", 0.0), cfg.get("lip_h", 0.0)
    if lip > 0:
        in_band = (g[:, 2] > z1 - lh) & ((g[:, 0] < x0 + lip) | (g[:, 0] > x1 - lip) |
                                        (g[:, 1] < y0 + lip) | (g[:, 1] > y1 - lip))
        keep &= ~in_band
    for tube in cfg["tubes"]:
        pts = tube["pts"]
        for a, b in zip(pts[:-1], pts[1:]):
            keep &= ~_capsule_mask(g, a, b, tube["r"])
    return g[keep]


def mouth_rims(cfg, n=48):
    rims = []
    for m in cfg["mouths"]:
        c, nrm, r = np.asarray(m["c"], float), np.asarray(m["n"], float), m["r"]
        nrm /= np.linalg.norm(nrm)
        e1 = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nrm, e1)
        u = np.linspace(0, 2 * math.pi, n, endpoint=False)
        rim = c + r * (np.cos(u)[:, None] * e1 + np.sin(u)[:, None] * e2)
        rims.append(np.vstack([rim, c]))
    return np.vstack(rims)


def directions(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = math.pi * (1 + 5 ** 0.5) * k
    d = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    named = {"upright": (0, 0, -1), "upside_down": (0, 0, 1),
             "tilt90_right(+x down)": (1, 0, 0), "tilt90_left(-x down)": (-1, 0, 0),
             "tilt90_back(+y down)": (0, 1, 0), "tilt90_front(-y down)": (0, -1, 0)}
    s = 1 / math.sqrt(2)
    for nm, v in [("tilt45_right", (s, 0, -s)), ("tilt45_left", (-s, 0, -s)),
                  ("tilt45_back", (0, s, -s)), ("tilt45_front", (0, -s, -s))]:
        named[nm] = v
    return d, {k: np.asarray(v, float) for k, v in named.items()}


def run(cfg, step=1.0, n_dirs=2000):
    pts = cavity_points(cfg, step)
    rim = mouth_rims(cfg)
    vox = step ** 3 / 1000.0                       # mL per voxel
    margin = cfg.get("margin_mm", 3.0)
    dirs, named = directions(n_dirs)

    def vmax(g):
        s = pts @ g
        s_m = (rim @ g).max()                      # lowest mouth point (largest depth along g)
        return float((s > s_m + margin).sum() * vox)

    worst, worst_dir = 1e9, None
    for g in dirs:
        v = vmax(g)
        if v < worst:
            worst, worst_dir = v, g
    named_v = {k: round(vmax(g)) for k, g in named.items()}
    worst = min(worst, min(named_v.values()))
    return {"cavity_ml": round(len(pts) * vox), "v_max_ml": round(worst, 1),
            "v_fill_ml": int(worst // 10 * 10),
            "worst_gravity_dir": [round(float(c), 3) for c in worst_dir],
            "max_fill_by_case_ml": named_v, "margin_mm": margin,
            "n_directions": int(n_dirs + len(named)), "voxel_mm": step,
            "note": "static check; water must stay >= margin below both snorkel mouths for every gravity direction"}


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    print(json.dumps(run(cfg), indent=1))
