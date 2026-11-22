"""SiPAP v3 router: fine raster (2.5 mil) copper model for clearance, 10 mil routing grid, numba A*.
Hard rules: 45-degree bends only, no foreign copper under bodies (own-pad escape corridor only),
no via in an SMD pad, edge / antenna / mounting-hole keep-outs, 10 mil clearance.
Output: route.json = hand copper + routed tracks/vias."""
import json, math, sys, time
import numpy as np
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path
from model import place_pads, place_body, FREEPADS
import place_v3 as PL
import hand
from astar import astar

F = 2.5                      # fine raster for clearance
G = 10.0                     # routing grid
K = int(G / F)
W_MIL, H_MIL = 4000, 3000
FX, FY = int(W_MIL / F) + 1, int(H_MIL / F) + 1
NX, NY = int(W_MIL / G) + 1, int(H_MIL / G) + 1
CLR = 10.0
TOP, BOT = 0, 1
LN = ["TopLayer", "BottomLayer"]
VIA_SIZE, VIA_HOLE = 24.0, 12.0
WIDTH = {"VIN12V": 50, "VIN_RAW": 60, "VIN_FUSED": 60, "BLOWER_V12": 50, "5V": 20, "3V3": 20, "GND": 20,
         "SFM_5V": 20, "SW3": 25, "SW5": 25}
BLOCKER = 32767              # copper of unconnected pads (no net)

fx = np.arange(FX) * F
fy = np.arange(FY) * F
nets = {}
def nid(n):
    if not n:
        return 0
    if n not in nets:
        nets[n] = len(nets) + 1
    return nets[n]

fine = np.zeros((2, FY, FX), np.int16)        # all copper, net id
conn = np.zeros((2, FY, FX), np.int16)        # copper already connected into its net's tree
smd = np.zeros((FY, FX), bool)                # SMD pad area (no via in pad, any net)
diag = np.zeros((2, FY, FX), np.int16)        # 45-degree segment copper (not a start point)

def _rng(a1, a2, step, n):
    return max(0, int(math.floor(a1 / step))), min(n - 1, int(math.ceil(a2 / step)))

def _sub(arr, L, j1, j2, i1, i2):
    return arr[L, j1:j2 + 1, i1:i2 + 1] if arr.ndim == 3 else arr[j1:j2 + 1, i1:i2 + 1]

def paint_rect(arr, L, x1, y1, x2, y2, n):
    x1 -= F / 2; y1 -= F / 2; x2 += F / 2; y2 += F / 2      # conservative raster: every touched cell
    i1, i2 = _rng(x1, x2, F, FX); j1, j2 = _rng(y1, y2, F, FY)
    X = fx[i1:i2 + 1][None, :]; Y = fy[j1:j2 + 1][:, None]
    m = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
    _sub(arr, L, j1, j2, i1, i2)[m] = n

def paint_seg(arr, L, x1, y1, x2, y2, w, n):
    r = w / 2.0 + F * 0.71                                 # conservative raster (half cell diagonal)
    i1, i2 = _rng(min(x1, x2) - r, max(x1, x2) + r, F, FX); j1, j2 = _rng(min(y1, y2) - r, max(y1, y2) + r, F, FY)
    X = fx[i1:i2 + 1][None, :]; Y = fy[j1:j2 + 1][:, None]
    dx, dy = x2 - x1, y2 - y1; L2 = dx * dx + dy * dy
    t = np.clip(((X - x1) * dx + (Y - y1) * dy) / L2, 0, 1) if L2 > 0 else 0.0
    m = (X - x1 - t * dx) ** 2 + (Y - y1 - t * dy) ** 2 <= r * r
    _sub(arr, L, j1, j2, i1, i2)[m] = n

def paint_poly(arr, L, pts, n):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    i1, i2 = _rng(min(xs), max(xs), F, FX); j1, j2 = _rng(min(ys), max(ys), F, FY)
    X, Y = np.meshgrid(fx[i1:i2 + 1], fy[j1:j2 + 1])
    m = Path(pts).contains_points(np.c_[X.ravel(), Y.ravel()], radius=0.01).reshape(X.shape)
    _sub(arr, L, j1, j2, i1, i2)[m] = n

# ------------------------------------------------------------------ pads
PADS = []
for d, (x, y, r) in PL.P.items():
    PADS += place_pads(d, x, y, r)
for f in FREEPADS:
    PADS.append(dict(comp="", name=f["name"], x=f["x"], y=f["y"], w=f["w"], h=f["h"], net=f["net"],
                     layer=f["layer"], hole=f["hole"], shape="rect" if f["name"].startswith("TP") else "round"))

def round_pad(p):
    return p["layer"] == "MultiLayer" and p.get("shape") not in ("1", "rect") and abs(p["w"] - p["h"]) < 1

def paint_pad(arr, p, val):
    for L in p["layers"]:
        if round_pad(p) or (p["comp"] == "" and p["name"].startswith(("FID", "MH"))):
            paint_seg(arr, L, p["x"], p["y"], p["x"], p["y"], p["w"], val)
        else:
            paint_rect(arr, L, p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, val)

for p in PADS:
    n = p["net"] if p["net"] and not p["net"].startswith("Net") else ""
    p["nid"] = nid(n)
    p["layers"] = [TOP, BOT] if p["layer"] == "MultiLayer" else [TOP] if p["layer"] == "TopLayer" else [BOT]
    paint_pad(fine, p, p["nid"] if p["nid"] else BLOCKER)
    if p["layer"] != "MultiLayer":
        paint_rect(smd, 0, p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, True)

# ------------------------------------------------------------------ hand copper (fixed; connected by definition)
ROUTE = dict(regions=list(hand.R), tracks=list(hand.T), vias=list(hand.V))
def paint_item(arrs, kind, o):
    n = nid(o["net"])
    for arr in arrs:
        if kind == "region":
            paint_poly(arr, 0 if o["layer"] == "TopLayer" else 1, o["pts"], n)
        elif kind == "track":
            paint_seg(arr, 0 if o["layer"] == "TopLayer" else 1, o["x1"], o["y1"], o["x2"], o["y2"], o["width"], n)
        else:
            for L in (TOP, BOT):
                paint_seg(arr, L, o["x"], o["y"], o["x"], o["y"], o["size"], n)
for o in hand.R: paint_item((fine, conn), "region", o)
for o in hand.T:
    paint_item((fine, conn), "track", o)
    if o["x1"] != o["x2"] and o["y1"] != o["y2"]:
        paint_seg(diag, 0 if o["layer"] == "TopLayer" else 1, o["x1"], o["y1"], o["x2"], o["y2"], o["width"] + 2 * G, nid(o["net"]))
for o in hand.V: paint_item((fine, conn), "via", o)

# ------------------------------------------------------------------ keep-outs (all nets)
XXf, YYf = np.meshgrid(fx, fy)
keep = (XXf < 20) | (XXf > W_MIL - 20) | (YYf < 20) | (YYf > H_MIL - 20)
keep |= (XXf > 1560) & (XXf < 2440) & (YYf > 2745)                    # antenna: no copper on any layer
for mx, my in ((150, 150), (3850, 150), (150, 2850), (3850, 2850)):
    keep |= (XXf - mx) ** 2 + (YYf - my) ** 2 < 150 ** 2              # M3 head
del XXf, YYf
keepd = (distance_transform_edt(~keep) * F)[::K, ::K]
smdd = (distance_transform_edt(~smd) * F)[::K, ::K]

# ------------------------------------------------------------------ bodies, escape corridors (coarse)
gx = np.arange(NX) * G; gy = np.arange(NY) * G
GX, GY = np.meshgrid(gx, gy)
BODIES = {d: place_body(d, *PL.P[d]) for d in PL.P}
COMP_PADS = {}
for p in PADS:
    if p["comp"]:
        COMP_PADS.setdefault(p["comp"], []).append(p)

def body_block(n, w):
    m = np.zeros((NY, NX), bool)
    infl = w / 2 + 2
    for d, b in BODIES.items():
        x1, y1, x2, y2 = b[0] - infl, b[1] - infl, b[2] + infl, b[3] + infl
        mb = (GX >= x1) & (GX <= x2) & (GY >= y1) & (GY <= y2)
        if n:
            for p in COMP_PADS.get(d, []):
                if p["nid"] != n:
                    continue
                px, py, hw, hh = p["x"], p["y"], p["w"] / 2, p["h"] / 2
                dl, dr, dd, du = px - b[0], b[2] - px, py - b[1], b[3] - py
                k = min(dl, dr, dd, du)
                if k == dl:   cx1, cx2, cy1, cy2 = x1 - G, px, py - hh, py + hh
                elif k == dr: cx1, cx2, cy1, cy2 = px, x2 + G, py - hh, py + hh
                elif k == dd: cx1, cx2, cy1, cy2 = px - hw, px + hw, y1 - G, py
                else:         cx1, cx2, cy1, cy2 = px - hw, px + hw, py, y2 + G
                corr = (GX >= cx1 - 0.1) & (GX <= cx2 + 0.1) & (GY >= cy1 - 0.1) & (GY <= cy2 + 0.1)
                pad = (GX >= px - hw - 0.1) & (GX <= px + hw + 0.1) & (GY >= py - hh - 0.1) & (GY <= py + hh + 0.1)
                mb &= ~corr & ~pad
        m |= mb
    return m

BB_VIA = body_block(0, VIA_SIZE)

def masks(n, w):
    r = w / 2 + CLR
    blk = np.zeros((2, NY, NX), bool)
    vok = np.ones((NY, NX), bool)
    for L in (TOP, BOT):
        other = (fine[L] != 0) & (fine[L] != n)
        dk = (distance_transform_edt(~other) * F)[::K, ::K]
        blk[L] = dk < r + 3.5
        vok &= dk >= VIA_SIZE / 2 + CLR + 3.5
    bb = body_block(n, w)
    for L in (TOP, BOT):
        blk[L] |= (keepd < w / 2 + 2.0) | bb
    vok &= (smdd >= VIA_SIZE / 2 + 4.0) & (keepd >= VIA_SIZE / 2 + 1) & ~BB_VIA
    return blk, vok

def pad_cells(p):
    hw, hh = p["w"] / 2 - 1.5, p["h"] / 2 - 1.5
    if round_pad(p):
        hw = hh = p["w"] / 2 * 0.7
    m = (np.abs(GX - p["x"]) <= max(hw, 0.1)) & (np.abs(GY - p["y"]) <= max(hh, 0.1))
    if not m.any():
        m[int(round(p["y"] / G)), int(round(p["x"] / G))] = True
    jj, ii = np.nonzero(m)
    return [(L, int(j), int(i)) for L in p["layers"] for j, i in zip(jj, ii)]

def pad_touches(p, n):
    for L in p["layers"]:
        i1, i2 = _rng(p["x"] - p["w"] / 2, p["x"] + p["w"] / 2, F, FX)
        j1, j2 = _rng(p["y"] - p["h"] / 2, p["y"] + p["h"] / 2, F, FY)
        if (conn[L, j1:j2 + 1, i1:i2 + 1] == n).any():
            return True
    return False

def tree_cells(n):
    out = []
    for L in (TOP, BOT):
        jj, ii = np.nonzero((conn[L, ::K, ::K] == n) & (diag[L, ::K, ::K] != n))
        out += [(L, int(j), int(i)) for j, i in zip(jj, ii)]
    return out

def emit(path, n, net, w):
    pts = [(int(L), i * G, j * G) for (L, j, i) in path]
    segs, vias, cur = [], [], [pts[0]]
    for p in pts[1:]:
        if p[0] != cur[-1][0]:
            vias.append((p[1], p[2])); segs.append(cur); cur = [p]
        else:
            cur.append(p)
    segs.append(cur)
    out = []
    for s in segs:
        if len(s) < 2:
            continue
        L = s[0][0]
        poly = [s[0]]
        for a, b, c in zip(s, s[1:], s[2:]):
            if (b[1] - a[1], b[2] - a[2]) != (c[1] - b[1], c[2] - b[2]):
                poly.append(b)
        poly.append(s[-1])
        for a, b in zip(poly, poly[1:]):
            out.append(dict(x1=a[1], y1=a[2], x2=b[1], y2=b[2], width=w, net=net, layer=LN[L]))
            paint_seg(fine, L, a[1], a[2], b[1], b[2], w, n)
            paint_seg(conn, L, a[1], a[2], b[1], b[2], w, n)
            if a[1] != b[1] and a[2] != b[2]:
                paint_seg(diag, L, a[1], a[2], b[1], b[2], w + 2 * G, n)
    for (x, y) in vias:
        ROUTE["vias"].append(dict(x=x, y=y, size=VIA_SIZE, hole=VIA_HOLE, net=net))
        for L in (TOP, BOT):
            paint_seg(fine, L, x, y, x, y, VIA_SIZE, n)
            paint_seg(conn, L, x, y, x, y, VIA_SIZE, n)
    ROUTE["tracks"].extend(out)
    return out, vias

LCOST = np.array([1.0, 2.2])
COST = np.ones((2, NY, NX))
for (x1, y1, x2, y2) in getattr(PL, "BOTTOM_CORRIDORS", []):
    COST[1][(GX >= x1) & (GX <= x2) & (GY >= y1) & (GY <= y2)] = 1.15 / 3.0

def route_net(net, first=None, maxexp=4_000_000, t45=25.0, via_cost=250.0, log=print):
    n = nid(net); w = WIDTH.get(net, 10)
    pads = [p for p in PADS if p["nid"] == n]
    if len(pads) < 2 and not tree_cells(n):
        return []
    blk, vok = masks(n, w)
    if not tree_cells(n):
        p0 = next((p for p in pads if first and f'{p["comp"]}.{p["name"]}' == first), None) or \
             next((p for p in pads if p["comp"] == "U1"), pads[0])
        paint_pad(conn, p0, n)
    fails = []
    todo = [p for p in pads if not pad_touches(p, n)]
    for p in pads:
        if pad_touches(p, n):
            paint_pad(conn, p, n)
    while todo:
        todo = [p for p in todo if not pad_touches(p, n)]
        if not todo:
            break
        tc = tree_cells(n)
        tarr = np.array(tc, np.int64)
        # nearest pad to the tree first
        def dist(p):
            return np.min(np.abs(tarr[:, 2] * G - p["x"]) + np.abs(tarr[:, 1] * G - p["y"]))
        todo.sort(key=dist)
        p = todo.pop(0)
        src = tarr if len(tarr) <= 12000 else tarr[:: max(1, len(tarr) // 12000)]
        tgt = np.zeros((2, NY, NX), bool)
        cells = pad_cells(p)
        ok = [(L, j, i) for (L, j, i) in cells if not blk[L, j, i]]
        if not ok:   # fall back to the pad centre cell
            ok = [(L, int(round(p["y"] / G)), int(round(p["x"] / G))) for L in p["layers"]]
        for (L, j, i) in ok:
            tgt[L, j, i] = True
        t0 = time.time()
        path, exp = astar(blk, vok, COST, src, tgt, int(round(p["x"] / G)), int(round(p["y"] / G)),
                          G, LCOST, t45, via_cost, maxexp, True)
        if len(path) == 0:
            fails.append(f'{p["comp"]}.{p["name"]}')
            log(f"  FAIL {net} -> {p['comp']}.{p['name']} ({exp} exp)")
            continue
        segs, vias = emit(path, n, net, w)
        paint_pad(conn, p, n)
        blk, vok = masks(n, w)  # own new copper changes nothing for n, but keep fresh for safety
        log(f"  {net} -> {p['comp']}.{p['name']}: {len(segs)} segs {len(vias)} vias {exp} exp {time.time() - t0:.1f}s")
    return fails

def via_free(x, y, min_d=36.0):
    return all(math.hypot(v["x"] - x, v["y"] - y) >= min_d for v in ROUTE["vias"])

def cells_ok(mask, x, y):
    i0, j0 = int(math.floor(x / G)), int(math.floor(y / G))
    return all(0 <= i < NX and 0 <= j < NY and mask[j, i] for i in (i0, i0 + 1) for j in (j0, j0 + 1))

def add_gnd_vias(log=print, stub_w=15.0):
    """dog-bone: every SMD GND pad without a GND via within 70 mil gets a short stub + via."""
    g = nid("GND")
    blk, vok = masks(g, stub_w)
    vias = [(v["x"], v["y"]) for v in ROUTE["vias"] if v["net"] == "GND"]
    added = 0
    for p in PADS:
        if p["nid"] != g or p["layer"] != "TopLayer":
            continue
        if any(math.hypot(vx - p["x"], vy - p["y"]) < 70 for vx, vy in vias):
            continue
        best = None
        for ang in range(0, 360, 45):
            dx, dy = round(math.cos(math.radians(ang))), round(math.sin(math.radians(ang)))
            for dist in range(30, 160, 10):
                vx = p["x"] + dx * (p["w"] / 2 + dist) if dx else p["x"]
                vy = p["y"] + dy * (p["h"] / 2 + dist) if dy else p["y"]
                if dx and dy:                      # 45 degrees exactly from the pad centre
                    k = max(p["w"] / 2, p["h"] / 2) + dist
                    vx, vy = p["x"] + dx * k, p["y"] + dy * k
                if not cells_ok(vok, vx, vy) or not via_free(vx, vy):
                    continue
                ok = True
                steps = int(max(abs(vx - p["x"]), abs(vy - p["y"])) / 2.5) + 1
                for k in range(steps + 1):
                    x = p["x"] + (vx - p["x"]) * k / steps; y = p["y"] + (vy - p["y"]) * k / steps
                    ii, jj = int(round(x / G)), int(round(y / G))
                    inside = abs(x - p["x"]) <= p["w"] / 2 and abs(y - p["y"]) <= p["h"] / 2
                    if not inside and blk[0, jj, ii]:
                        ok = False; break
                if not ok:
                    continue
                cost = math.hypot(vx - p["x"], vy - p["y"]) + (0 if ang % 90 == 0 else 20)
                if best is None or cost < best[0]:
                    best = (cost, vx, vy)
                break
        if best is None:
            log(f"  no GND via spot for {p['comp']}.{p['name']}")
            continue
        _, vx, vy = best
        ROUTE["tracks"].append(dict(x1=p["x"], y1=p["y"], x2=vx, y2=vy, width=stub_w, net="GND", layer="TopLayer"))
        ROUTE["vias"].append(dict(x=vx, y=vy, size=VIA_SIZE, hole=VIA_HOLE, net="GND"))
        paint_seg(fine, 0, p["x"], p["y"], vx, vy, stub_w, g)
        for L in (TOP, BOT):
            paint_seg(fine, L, vx, vy, vx, vy, VIA_SIZE, g)
        vias.append((vx, vy)); added += 1
        blk, vok = masks(g, stub_w)
    log(f"GND dog-bone vias added: {added}")

def add_return_vias(log=print, reach=40.0):
    """a GND via within `reach` of every signal via, where one fits (pours connect it)."""
    g = nid("GND")
    blk, vok = masks(g, 10)
    gv = [(v["x"], v["y"]) for v in ROUTE["vias"] if v["net"] == "GND"]
    added = 0; miss = []
    for v in list(ROUTE["vias"]):
        if v["net"] == "GND":
            continue
        if any(math.hypot(x - v["x"], y - v["y"]) <= reach for x, y in gv):
            continue
        best = None
        for dx in range(-40, 41, 10):
            for dy in range(-40, 41, 10):
                d = math.hypot(dx, dy)
                if d > reach or d < 30:
                    continue
                i, j = int(round((v["x"] + dx) / G)), int(round((v["y"] + dy) / G))
                if 0 <= i < NX and 0 <= j < NY and vok[j, i] and via_free(i * G, j * G) and (best is None or d < best[0]):
                    best = (d, i * G, j * G)
        if best is None:
            miss.append(v["net"]); continue
        _, x, y = best
        ROUTE["vias"].append(dict(x=x, y=y, size=VIA_SIZE, hole=VIA_HOLE, net="GND"))
        for L in (TOP, BOT):
            paint_seg(fine, L, x, y, x, y, VIA_SIZE, g)
        gv.append((x, y)); added += 1
        blk, vok = masks(g, 10)
    log(f"return vias added: {added}; signal vias without one: {len(miss)} {sorted(set(miss))}")

def drop_overlaps():
    """remove a routed track that lies on top of another collinear track of the same net and layer."""
    T = ROUTE["tracks"]; nh = len(hand.T)
    def on(t, x, y):
        dx, dy = t["x2"] - t["x1"], t["y2"] - t["y1"]; L2 = dx * dx + dy * dy
        if L2 == 0: return False
        u = ((x - t["x1"]) * dx + (y - t["y1"]) * dy) / L2
        return -1e-6 <= u <= 1 + 1e-6 and abs((x - t["x1"]) * dy - (y - t["y1"]) * dx) / math.sqrt(L2) < 0.5
    keep = []
    for k, t in enumerate(T):
        if k >= nh and any(j != k and o["net"] == t["net"] and o["layer"] == t["layer"] and o["width"] >= t["width"]
                           and on(o, t["x1"], t["y1"]) and on(o, t["x2"], t["y2"]) and (j < k or not (on(t, o["x1"], o["y1"]) and on(t, o["x2"], o["y2"])))
                           for j, o in enumerate(T)):
            continue
        keep.append(t)
    ROUTE["tracks"] = keep

def split_at_pads():
    """a routed track that runs over a same-net SMD pad is cut at the pad (ends on it instead of crossing)."""
    nh = len(hand.T)
    out = ROUTE["tracks"][:nh]
    for t in ROUTE["tracks"][nh:]:
        pieces = [t]
        for p in PADS:
            if p["net"] != t["net"] or p["layer"] != "TopLayer" or t["layer"] != "TopLayer":
                continue
            new = []
            for s in pieces:
                x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
                dx, dy = x2 - x1, y2 - y1; L2 = dx * dx + dy * dy
                if L2 == 0:
                    new.append(s); continue
                u = ((p["x"] - x1) * dx + (p["y"] - y1) * dy) / L2
                px, py = x1 + u * dx, y1 + u * dy
                if 0.05 < u < 0.95 and abs(px - p["x"]) <= p["w"] / 2 and abs(py - p["y"]) <= p["h"] / 2:
                    cx, cy = round(px / 0.5) * 0.5, round(py / 0.5) * 0.5
                    new.append(dict(s, x2=cx, y2=cy)); new.append(dict(s, x1=cx, y1=cy))
                else:
                    new.append(s)
            pieces = new
        out.extend(pieces)
    ROUTE["tracks"] = out

def run(plan, out="route.json", log=print):
    fails = {}
    for item in plan:
        f = route_net(item, log=log)
        if f:
            fails[item] = f
        json.dump(ROUTE, open(out, "w"))
    if not fails:
        add_gnd_vias(log=log)
        add_return_vias(log=log)
        drop_overlaps()
        split_at_pads()
        json.dump(ROUTE, open(out, "w"))
    return fails

if __name__ == "__main__":
    plan = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    if plan is None:
        sig = sorted({p["net"] for p in PADS if p["nid"] and p["net"] != "GND"})
        plan = sig
    t = time.time()
    fails = run(plan)
    print("FAILS", fails, "time", round(time.time() - t, 1))
