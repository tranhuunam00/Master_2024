"""GND via next to every GND pad that hand copper does not already serve. Adds to route.json."""
import json, math
import numpy as np
import router as R

route = json.load(open("route.json"))
hand_n = len(R.power.T), len(R.power.V)
for t in route["tracks"][hand_n[0]:]:
    R.paint_seg(0 if t["layer"] == "TopLayer" else 1, t["x1"], t["y1"], t["x2"], t["y2"], t["width"], R.nid(t["net"]))
for v in route["vias"][hand_n[1]:]:
    for L in (0, 1):
        R.paint_seg(L, v["x"], v["y"], v["x"], v["y"], v["size"], R.nid(v["net"]))

g = R.nets["GND"]
W = 15
added = []
blk, pen, vok = R.masks_for(g, W)
B = [np.frombuffer(b, dtype=bool).reshape(R.NY, R.NX) for b in blk]
VOK = np.frombuffer(vok, dtype=bool).reshape(R.NY, R.NX).copy()

def line_free(L, x1, y1, x2, y2):
    n = int(max(abs(x2 - x1), abs(y2 - y1)) / 2.5) + 1
    for k in range(n + 1):
        x, y = x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n
        i, j = int(round(x / R.G)), int(round(y / R.G))
        if B[L][j, i]:
            return False
    return True

for p in R.PADS:
    if p["nid"] != g or p["layer"] == "MultiLayer" or p["comp"] in ("", "U1"):
        continue
    if R.pad_touch(p, R.HAND, g):
        continue
    best = None
    for ang in range(0, 360, 45):
        for d in range(25, 120, 5):
            ex = max(p["w"], p["h"]) / 2 if ang % 90 == 0 else 0
            dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            vx = p["x"] + dx * (p["w"] / 2 + 14 + d - 25 + 0)
            vy = p["y"] + dy * (p["h"] / 2 + 14 + d - 25 + 0)
            i, j = int(round(vx / R.G)), int(round(vy / R.G))
            if not (0 <= i < R.NX and 0 <= j < R.NY) or not VOK[j, i]:
                continue
            if not line_free(0, p["x"], p["y"], i * R.G, j * R.G):
                continue
            cost = math.hypot(i * R.G - p["x"], j * R.G - p["y"]) + (0 if ang % 90 == 0 else 15)
            if best is None or cost < best[0]:
                best = (cost, i * R.G, j * R.G)
            break
    if best is None:
        print("no via spot for", p["comp"], p["name"])
        continue
    _, x, y = best
    route["tracks"].append(dict(x1=p["x"], y1=p["y"], x2=x, y2=y, width=W, net="GND", layer="TopLayer"))
    route["vias"].append(dict(x=x, y=y, size=24, hole=12, net="GND"))
    R.paint_seg(0, p["x"], p["y"], x, y, W, g)
    for L in (0, 1):
        R.paint_seg(L, x, y, x, y, 24, g)
    # keep later vias apart
    VOK[max(0, int(y / R.G) - 8):int(y / R.G) + 9, max(0, int(x / R.G) - 8):int(x / R.G) + 9] = False
    added.append(f'{p["comp"]}.{p["name"]}')
json.dump(route, open("route.json", "w"))
print("GND vias added:", len(added), added)
