"""Board model: footprints (local pads + body), placement, pad nets."""
import json, math

import os
_ALIAS = {"c_comps.txt": "board_parts_before_v3.json", "c_pads.txt": "board_pads_before_v3.json"}

def _load(f):
    if not os.path.exists(f) and f in _ALIAS:   # the dumps this layout was made from
        f = _ALIAS[f]
    return json.load(open(f, encoding="utf-8"))

COMPS = {c["designator"]: c for c in _load("c_comps.txt")["components"]}
PADS = _load("c_pads.txt")["objects"]

def rot(x, y, deg):
    r = math.radians(deg)
    c, s = round(math.cos(r), 9), round(math.sin(r), 9)
    return x * c - y * s, x * s + y * c

# local pad geometry per component (rotation 0 frame)
LOCAL = {}
FREEPADS = []
for p in PADS:
    d = p["Component"]
    w, h = float(p["TopXSize"]), float(p["TopYSize"])
    if not d:
        FREEPADS.append(dict(x=float(p["X"]), y=float(p["Y"]), w=w, h=h, net=p["Net"],
                             layer=p["Layer"], hole=float(p["HoleSize"] or 0)))
        continue
    c = COMPS[d]
    lx, ly = rot(float(p["X"]) - c["x"], float(p["Y"]) - c["y"], -c["rotation"])
    pr = (float(p["Rotation"]) - c["rotation"]) % 180
    if abs(pr - 90) < 1:
        w, h = h, w
    LOCAL.setdefault(d, []).append(dict(name=p["Name"], lx=lx, ly=ly, w=w, h=h, net=p["Net"],
                                        layer=p["Layer"], hole=float(p["HoleSize"] or 0),
                                        shape=p["TopShape"]))
# local body box (no name/comment) in rotation-0 frame
BODY = {}
for d, c in COMPS.items():
    b = c["bbox_no_name_comment"]
    pts = [rot(x - c["x"], y - c["y"], -c["rotation"]) for x, y in
           ((b["x1"], b["y1"]), (b["x2"], b["y2"]), (b["x1"], b["y2"]), (b["x2"], b["y1"]))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    BODY[d] = (min(xs), min(ys), max(xs), max(ys))

def place_pads(d, x, y, r):
    out = []
    for p in LOCAL[d]:
        px, py = rot(p["lx"], p["ly"], r)
        w, h = (p["h"], p["w"]) if r % 180 == 90 else (p["w"], p["h"])
        out.append(dict(comp=d, name=p["name"], x=x + px, y=y + py, w=w, h=h, net=p["net"],
                        layer=p["layer"], hole=p["hole"], shape=p["shape"]))
    return out

def place_body(d, x, y, r):
    x1, y1, x2, y2 = BODY[d]
    pts = [rot(a, b, r) for a, b in ((x1, y1), (x2, y2), (x1, y2), (x2, y1))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (x + min(xs), y + min(ys), x + max(xs), y + max(ys))

CUR = {d: (c["x"], c["y"], c["rotation"]) for d, c in COMPS.items()}

# test points are free pads: new positions (net -> (x, y))
TP_MOVE = {"VIN12V": (1225, 2140), "5V": (480, 1440), "3V3": (1170, 1440), "GND": (200, 1420)}
for f in FREEPADS:
    if f["net"] in TP_MOVE and f["w"] < 100:
        f["ox"], f["oy"] = f["x"], f["y"]
        f["x"], f["y"] = TP_MOVE[f["net"]]
