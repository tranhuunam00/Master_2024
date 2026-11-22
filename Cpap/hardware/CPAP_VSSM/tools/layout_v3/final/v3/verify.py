"""Offline checks on route json: clearance (different nets, per layer), under-body copper, corners, via-in-pad."""
import json, sys, math, itertools
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree
import place_v3 as PL
from model import place_pads, place_body, FREEPADS

route = json.load(open(sys.argv[1]))
CLR = 9.9
items = []   # (geom, net, layer, kind, ref)
for d, (x, y, r) in PL.P.items():
    for p in place_pads(d, x, y, r):
        g = Point(p["x"], p["y"]).buffer(p["w"] / 2) if (p["layer"] == "MultiLayer" and abs(p["w"] - p["h"]) < 1 and p["shape"] != "1") \
            else box(p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2)
        n = p["net"] if p["net"] and not p["net"].startswith("Net") else f"NC:{d}.{p['name']}"
        for L in (["TopLayer", "BottomLayer"] if p["layer"] == "MultiLayer" else [p["layer"]]):
            items.append((g, n, L, "pad", f"{d}.{p['name']}"))
for f in FREEPADS:
    g = Point(f["x"], f["y"]).buffer(f["w"] / 2)
    for L in (["TopLayer", "BottomLayer"] if f["layer"] == "MultiLayer" else [f["layer"]]):
        items.append((g, f["net"] or f"NC:{f['name']}", L, "pad", f["name"]))
for k, t in enumerate(route["tracks"]):
    g = LineString([(t["x1"], t["y1"]), (t["x2"], t["y2"])]).buffer(t["width"] / 2, cap_style=1) if (t["x1"], t["y1"]) != (t["x2"], t["y2"]) \
        else Point(t["x1"], t["y1"]).buffer(t["width"] / 2)
    items.append((g, t["net"], t["layer"], "track", f"t{k}"))
for k, v in enumerate(route["vias"]):
    g = Point(v["x"], v["y"]).buffer(v["size"] / 2)
    for L in ("TopLayer", "BottomLayer"):
        items.append((g, v["net"], L, "via", f"v{k}"))
for k, rg in enumerate(route["regions"]):
    items.append((Polygon(rg["pts"]), rg["net"], rg["layer"], "region", f"r{k}"))

viol = []
for L in ("TopLayer", "BottomLayer"):
    li = [it for it in items if it[2] == L]
    tree = STRtree([it[0] for it in li])
    for i, a in enumerate(li):
        for j in tree.query(a[0].buffer(CLR)):
            if j <= i:
                continue
            b = li[j]
            if a[1] == b[1]:
                continue
            dmin = a[0].distance(b[0])
            if dmin < CLR:
                viol.append((round(dmin, 2), L, a[4], a[1], b[4], b[1]))
print("clearance violations:", len(viol))
for v in sorted(viol)[:40]:
    print("  ", v)

# board edge
edge = 0
for it in items:
    if it[3] in ("track", "via", "region"):
        b = it[0].bounds
        if b[0] < 15 or b[1] < 15 or b[2] > 3985 or b[3] > 2985:
            edge += 1; print("  edge", it[4], it[1], [round(t) for t in b])
print("edge problems:", edge)

# antenna keep-out
ant = box(1560, 2745, 2440, 3000)
na = [it[4] for it in items if it[3] in ("track", "via", "region") and it[0].intersects(ant)]
print("copper in antenna keep-out:", na)

# vias in SMD pads
smd = [it for it in items if it[3] == "pad" and it[2] == "TopLayer"]
vip = []
for k, v in enumerate(route["vias"]):
    c = Point(v["x"], v["y"]).buffer(v["size"] / 2)
    for it in smd:
        if it[0].intersects(c) and not it[4].startswith("U1.41"):
            vip.append((f"v{k}", v["net"], it[4]))
print("via in/touching SMD pad:", vip)

# under-body: foreign copper (or own net beyond an escape) inside physical bodies
ub = []
for d, (x, y, r) in PL.P.items():
    b = place_body(d, x, y, r)
    bx = box(b[0] + 1, b[1] + 1, b[2] - 1, b[3] - 1)
    own = {p["net"] for p in place_pads(d, x, y, r)}
    for it in items:
        if it[3] in ("track", "via", "region") and it[0].intersects(bx):
            if it[1] not in own:
                ub.append((d, it[1], it[2][0], it[4]))
print("foreign copper under bodies:", len(ub))
for u in ub[:40]:
    print("  ", u)

# corners: angle between consecutive collinear-joined segments of a net on a layer
import collections
ends = collections.defaultdict(list)
for k, t in enumerate(route["tracks"]):
    for (x, y) in ((t["x1"], t["y1"]), (t["x2"], t["y2"])):
        ends[(t["layer"], round(x, 1), round(y, 1))].append(k)
sharp = 0
for key, ks in ends.items():
    if len(ks) != 2:
        continue
    a, b = route["tracks"][ks[0]], route["tracks"][ks[1]]
    def vec(t, at):
        x, y = at[1], at[2]
        if (round(t["x1"], 1), round(t["y1"], 1)) == (x, y):
            return (t["x2"] - t["x1"], t["y2"] - t["y1"])
        return (t["x1"] - t["x2"], t["y1"] - t["y2"])
    u, w = vec(a, key), vec(b, key)
    nu, nw = math.hypot(*u), math.hypot(*w)
    if nu == 0 or nw == 0:
        continue
    ang = math.degrees(math.acos(max(-1, min(1, (u[0] * w[0] + u[1] * w[1]) / nu / nw))))
    if ang < 134.9:            # interior angle < 135 = bend sharper than 45 degrees
        sharp += 1
        if sharp <= 15:
            print("  sharp bend", key, round(ang, 1), a["net"])
print("bends sharper than 45 deg:", sharp)
