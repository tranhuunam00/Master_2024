"""Board model for SiPAP v3: local footprint geometry (pads + body) from the v2 dump, placement helpers."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "v2")

def _load(f):
    return json.load(open(os.path.join(SRC, f), encoding="utf-8"))

COMPS = {c["designator"]: c for c in _load("comps.json")["components"]}
PADS = _load("pads.json")["objects"]

def rot(x, y, deg):
    r = math.radians(deg)
    c, s = round(math.cos(r), 12), round(math.sin(r), 12)
    return x * c - y * s, x * s + y * c

LOCAL, FREEPADS = {}, []
for p in PADS:
    d = p["Component"]
    w, h = float(p["TopXSize"]), float(p["TopYSize"])
    if not d:
        FREEPADS.append(dict(name=p["Name"], x=float(p["X"]), y=float(p["Y"]), w=w, h=h, net=p["Net"],
                             layer=p["Layer"], hole=float(p["HoleSize"] or 0), shape=p["TopShape"]))
        continue
    c = COMPS[d]
    lx, ly = rot(float(p["X"]) - c["x"], float(p["Y"]) - c["y"], -c["rotation"])
    pr = (float(p["Rotation"]) - c["rotation"]) % 180
    if abs(pr - 90) < 1:
        w, h = h, w
    LOCAL.setdefault(d, []).append(dict(name=p["Name"], lx=round(lx, 3), ly=round(ly, 3), w=w, h=h, net=p["Net"],
                                        layer=p["Layer"], hole=float(p["HoleSize"] or 0), shape=p["TopShape"]))

import netmap
netmap.apply(LOCAL)
# C27 (new, 22 uF X5R 16 V 0805) at U1 pin 2: geometry of C3, pad1 = 3V3, pad2 = GND
import copy as _copy
COMPS['C27'] = dict(COMPS['C3'], designator='C27', comment='22uF 16V')
LOCAL['C27'] = [dict(p, net=('3V3' if p['name'] == '1' else 'GND')) for p in _copy.deepcopy(LOCAL['C3'])]

# physical body outline (rotation-0 frame, mils), from package drawings; None = use pad extents
MM = 1 / 0.0254
FP_BODY = {
    "R_0805": (35.8, 1.25 * MM),          # gap between the two pads (terminations are pad copper)
    "SOIC-8": (4.9 * MM, 3.9 * MM),            # TI D0008A
    "SOT-23": (2.9 * MM, 1.3 * MM),
    "TI_DDC0006A_SOT23-6": (2.9 * MM, 1.6 * MM),
    "IND_BOURNS_SRN6045TA": (6.0 * MM, 6.0 * MM),
    "DIOM_SMB_DO-214AA_BOURNS": (4.3 * MM, 3.6 * MM),
}

def footprint(d):
    return COMPS[d]["footprint"]

def body_local(d):
    """axis-aligned body box in the part's own frame, centred on the pad centroid for small parts."""
    fp = footprint(d)
    pads = LOCAL.get(d, [])
    if fp in FP_BODY:
        bw, bh = FP_BODY[fp]
        # orientation: the long body side follows the pad row direction
        xs = [p["lx"] for p in pads]; ys = [p["ly"] for p in pads]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        spanx, spany = max(xs) - min(xs), max(ys) - min(ys)
        if fp in ("R_0805", "DIOM_SMB_DO-214AA_BOURNS"):
            if spany > spanx: bw, bh = bh, bw
        elif fp in ("SOIC-8", "SOT-23", "TI_DDC0006A_SOT23-6"):
            # pins in rows along one axis; body long side along the row
            if spanx > spany: bw, bh = bh, bw   # rows left/right -> body tall
        return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    c = COMPS[d]
    b = c["bbox_no_name_comment"]
    pts = [rot(x - c["x"], y - c["y"], -c["rotation"]) for x, y in
           ((b["x1"], b["y1"]), (b["x2"], b["y2"]), (b["x1"], b["y2"]), (b["x2"], b["y1"]))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

BODY = {d: body_local(d) for d in COMPS}

def place_pads(d, x, y, r):
    out = []
    for p in LOCAL.get(d, []):
        px, py = rot(p["lx"], p["ly"], r)
        w, h = (p["h"], p["w"]) if round(r) % 180 == 90 else (p["w"], p["h"])
        out.append(dict(comp=d, name=p["name"], x=round(x + px, 3), y=round(y + py, 3), w=w, h=h, net=p["net"],
                        layer=p["layer"], hole=p["hole"], shape=p["shape"]))
    return out

def place_body(d, x, y, r):
    x1, y1, x2, y2 = BODY[d]
    pts = [rot(a, b, r) for a, b in ((x1, y1), (x2, y2), (x1, y2), (x2, y1))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (x + min(xs), y + min(ys), x + max(xs), y + max(ys))

CUR = {d: (c["x"], c["y"], c["rotation"]) for d, c in COMPS.items()}
