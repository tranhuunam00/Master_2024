"""Octilinear 2-layer grid router for SiPAP v3. Output: route.json (power copper + routed tracks/vias)."""
import json, math, heapq, sys, time
import numpy as np
from scipy.ndimage import distance_transform_edt
from model import place_pads, place_body, FREEPADS
from placement import P
import power

G = 5.0
NX, NY = int(4000 / G) + 1, int(3000 / G) + 1
CLR = 10.0
TOP, BOT = 0, 1
LNAME = ["TopLayer", "BottomLayer"]
LAYER_COST = [1.0, 2.2]
TURN45, TURN90, VIA_COST = 4.0, 30.0, 110.0
DIAG = 1.25
LAYER_COST_R, VIA_R = [1.0, 1.3], 35.0   # fallback when the strict costs find no path   # diagonal runs cost more: H/V with short 45-degree chamfers
BODY_COST = 6.0
HW = 1.3

XX, YY = np.meshgrid(np.arange(NX) * G, np.arange(NY) * G)  # [j, i]

# ---------------- copper model ----------------
nets = {}
def nid(n):
    if not n:
        return -1
    if n not in nets:
        nets[n] = len(nets) + 1
    return nets[n]

own = [np.zeros((NY, NX), np.int32), np.zeros((NY, NX), np.int32)]

def paint_rect(layer, x1, y1, x2, y2, n, arr=None):
    arr = own if arr is None else arr
    i1, i2 = max(0, math.ceil(x1 / G)), min(NX - 1, math.floor(x2 / G))
    j1, j2 = max(0, math.ceil(y1 / G)), min(NY - 1, math.floor(y2 / G))
    if i2 >= i1 and j2 >= j1:
        arr[layer][j1:j2 + 1, i1:i2 + 1] = n

def paint_seg(layer, x1, y1, x2, y2, w, n, arr=None):
    arr = own if arr is None else arr
    r = w / 2
    i1, i2 = max(0, int((min(x1, x2) - r) / G)), min(NX - 1, int((max(x1, x2) + r) / G) + 1)
    j1, j2 = max(0, int((min(y1, y2) - r) / G)), min(NY - 1, int((max(y1, y2) + r) / G) + 1)
    X, Y = XX[j1:j2 + 1, i1:i2 + 1], YY[j1:j2 + 1, i1:i2 + 1]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = np.clip(((X - x1) * dx + (Y - y1) * dy) / L2, 0, 1) if L2 > 0 else 0
    d2 = (X - x1 - t * dx) ** 2 + (Y - y1 - t * dy) ** 2
    m = d2 <= r * r
    arr[layer][j1:j2 + 1, i1:i2 + 1][m] = n

def paint_poly(layer, pts, n, arr=None):
    arr = own if arr is None else arr
    from matplotlib.path import Path
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    i1, i2 = max(0, int(min(xs) / G)), min(NX - 1, int(max(xs) / G) + 1)
    j1, j2 = max(0, int(min(ys) / G)), min(NY - 1, int(max(ys) / G) + 1)
    X, Y = XX[j1:j2 + 1, i1:i2 + 1], YY[j1:j2 + 1, i1:i2 + 1]
    m = Path(pts).contains_points(np.c_[X.ravel(), Y.ravel()], radius=0.01).reshape(X.shape)
    arr[layer][j1:j2 + 1, i1:i2 + 1][m] = n

PADS = []
for d, (x, y, r) in P.items():
    PADS += place_pads(d, x, y, r)
for f in FREEPADS:
    PADS.append(dict(comp="", name="TP" if f["w"] < 100 else "MH", x=f["x"], y=f["y"], w=f["w"], h=f["h"],
                     net=f["net"], layer=f["layer"], hole=f["hole"], shape="1"))
for p in PADS:
    n = nid(p["net"] if p["net"] and not p["net"].startswith("Net") else "")
    p["nid"] = n
    layers = [TOP, BOT] if p["layer"] == "MultiLayer" else [TOP] if p["layer"] == "TopLayer" else [BOT]
    p["layers"] = layers
    for L in layers:
        paint_rect(L, p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, n)

ROUTE = dict(regions=list(power.R), tracks=list(power.T), vias=list(power.V))
for rg in power.R:
    paint_poly(0 if rg["layer"] == "TopLayer" else 1, rg["pts"], nid(rg["net"]))
for t in power.T:
    paint_seg(0 if t["layer"] == "TopLayer" else 1, t["x1"], t["y1"], t["x2"], t["y2"], t["width"], nid(t["net"]))
for v in power.V:
    for L in (TOP, BOT):
        paint_seg(L, v["x"], v["y"], v["x"], v["y"], v["size"], nid(v["net"]))

HAND = [np.zeros((NY, NX), np.int32), np.zeros((NY, NX), np.int32)]
for rg in power.R:
    paint_poly(0 if rg["layer"] == "TopLayer" else 1, rg["pts"], nid(rg["net"]), HAND)
for t in power.T:
    paint_seg(0 if t["layer"] == "TopLayer" else 1, t["x1"], t["y1"], t["x2"], t["y2"], t["width"], nid(t["net"]), HAND)
for v in power.V:
    for L in (TOP, BOT):
        paint_seg(L, v["x"], v["y"], v["x"], v["y"], v["size"], nid(v["net"]), HAND)

BODIES = {d: place_body(d, *P[d]) for d in P}
COMP_NETS = {}
for p in PADS:
    if p["comp"]:
        COMP_NETS.setdefault(p["comp"], set()).add(p["nid"])

# static keepouts: board edge band, antenna area
static_block = np.zeros((NY, NX), bool)
EDGE = 25.0
static_block[(XX < EDGE) | (XX > 4000 - EDGE) | (YY < EDGE) | (YY > 3000 - EDGE)] = True
static_block[(XX > 1560) & (XX < 2440) & (YY > 2745)] = True

# SMD pad areas (no vias in pads, any net)
smd_mask = np.zeros((NY, NX), bool)
for p in PADS:
    if p["hole"] == 0 or p["layer"] != "MultiLayer":
        x1, y1 = p["x"] - p["w"] / 2 - 12 - 1, p["y"] - p["h"] / 2 - 12 - 1
        x2, y2 = p["x"] + p["w"] / 2 + 12 + 1, p["y"] + p["h"] / 2 + 12 + 1
        smd_mask[(XX >= x1) & (XX <= x2) & (YY >= y1) & (YY <= y2)] = True

DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
SQ2 = math.sqrt(2)

def masks_for(n, w):
    """blocked (per layer, bytes), body-penalty (bytes), via-ok (bytes) for net n, width w."""
    r = w / 2 + CLR + G / 2 + 0.5
    rv = 12 + CLR + G / 2 + 0.5
    blk, vok = [], None
    body_any = np.zeros((NY, NX), bool)
    body_own = np.zeros((NY, NX), bool)
    for d, b in BODIES.items():
        SH = 12.0   # courtyard -> body: the box is the courtyard, the body sits ~0.25 mm inside
        bx1, by1, bx2, by2 = b[0] + SH, b[1] + SH, b[2] - SH, b[3] - SH
        if bx2 <= bx1 or by2 <= by1:
            continue
        m = (XX >= bx1) & (XX <= bx2) & (YY >= by1) & (YY <= by2)
        if n in COMP_NETS.get(d, ()):
            body_own |= m
            # escape corridors: each own pad straight out to the nearest body edge
            for p in PADS:
                if p["comp"] != d or p["nid"] != n:
                    continue
                px, py, hw, hh = p["x"], p["y"], p["w"] / 2, p["h"] / 2
                dl, dr, dd, du = px - bx1, bx2 - px, py - by1, by2 - py
                k = min(dl, dr, dd, du)
                if k == dl:   cx1, cx2, cy1, cy2 = bx1 - G, px, py - hh, py + hh
                elif k == dr: cx1, cx2, cy1, cy2 = px, bx2 + G, py - hh, py + hh
                elif k == dd: cx1, cx2, cy1, cy2 = px - hw, px + hw, by1 - G, py
                else:         cx1, cx2, cy1, cy2 = px - hw, px + hw, py, by2 + G
                corr = (XX >= cx1) & (XX <= cx2) & (YY >= cy1) & (YY <= cy2)
                pad = (XX >= px - hw) & (XX <= px + hw) & (YY >= py - hh) & (YY <= py + hh)
                m = m & ~corr & ~pad
            body_any |= m
        else:
            body_any |= m
    vb = np.ones((NY, NX), bool)
    for L in (TOP, BOT):
        other = (own[L] != 0) & (own[L] != n)
        dist = distance_transform_edt(~other) * G
        b = (dist < r) | static_block | body_any
        blk.append(b)
        vb &= (dist >= rv)
    vb &= ~static_block & ~body_any & ~body_own & ~smd_mask
    pen = np.zeros((NY, NX), bool)
    return [b.ravel().tobytes() for b in blk], pen.ravel().tobytes(), vb.ravel().tobytes()

SOFT = None          # (bytes top, bytes bottom) of routed copper that may be crossed at a price
SOFT_COST = 60.0

def astar(sources, target, n, w, maxexp=1_500_000, relaxed=False, masks=None):
    blk, pen, vok = masks if masks else masks_for(n, w)
    tx, ty, tlayers = target
    ti, tj = int(round(tx / G)), int(round(ty / G))
    def h(i, j):
        dx, dy = abs(i - ti), abs(j - tj)
        return HW * ((max(dx, dy) - min(dx, dy)) + SQ2 * min(dx, dy))
    g = {}
    parent = {}
    pq = []
    for (L, i, j) in sources:
        s = (L, i, j, 8)
        g[s] = 0.0
        parent[s] = None
        heapq.heappush(pq, (h(i, j), 0.0, s))
    goal = None
    exp = 0
    while pq:
        f, gc, s = heapq.heappop(pq)
        if gc > g.get(s, 1e18):
            continue
        L, i, j, d = s
        if i == ti and j == tj and L in tlayers:
            goal = s
            break
        exp += 1
        if exp > maxexp:
            break
        for nd, (di, dj) in enumerate(DIRS):
            if d != 8:
                diff = abs(nd - d); diff = min(diff, 8 - diff)
                if diff > 2:
                    continue
                tc = 0 if diff == 0 else TURN45 if diff == 1 else TURN90
            else:
                tc = 0
            ni, nj = i + di, j + dj
            if not (0 <= ni < NX and 0 <= nj < NY):
                continue
            k = nj * NX + ni
            if blk[L][k] and not (ni == ti and nj == tj):
                continue
            step = (SQ2 * DIAG if di and dj else 1.0) * (LAYER_COST_R[L] if relaxed else LAYER_COST[L]) * (BODY_COST if pen[k] else 1.0)
            if SOFT is not None and SOFT[L][k]:
                step += SOFT_COST
            ns = (L, ni, nj, nd)
            ng = gc + step + tc
            if ng < g.get(ns, 1e18):
                g[ns] = ng; parent[ns] = s
                heapq.heappush(pq, (ng + h(ni, nj), ng, ns))
        k = j * NX + i
        if vok[k] and d != 8:
            ns = (1 - L, i, j, d)
            if not blk[1 - L][k]:
                ng = gc + (VIA_R if relaxed else VIA_COST)
                if ng < g.get(ns, 1e18):
                    g[ns] = ng; parent[ns] = s
                    heapq.heappush(pq, (ng + h(i, j), ng, ns))
    if goal is None:
        return None, exp
    path = []
    s = goal
    while s is not None:
        path.append(s); s = parent[s]
    return path[::-1], exp

def emit(path, n, net, w):
    """path -> merged segments + vias, painted into the copper model."""
    pts = [(L, i * G, j * G) for (L, i, j, d) in path]
    segs, vias = [], []
    cur = [pts[0]]
    for p in pts[1:]:
        if p[0] != cur[-1][0]:
            vias.append((p[1], p[2]))
            segs.append(cur); cur = [p]
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
            t = dict(x1=a[1], y1=a[2], x2=b[1], y2=b[2], width=w, net=net, layer=LNAME[L])
            out.append(t)
            paint_seg(L, a[1], a[2], b[1], b[2], w, n)
            paint_seg(L, a[1], a[2], b[1], b[2], w, 1, TREE)
    for (x, y) in vias:
        ROUTE["vias"].append(dict(x=x, y=y, size=24, hole=12, net=net))
        for L in (TOP, BOT):
            paint_seg(L, x, y, x, y, 24, n)
            paint_seg(L, x, y, x, y, 24, 1, TREE)
    ROUTE["tracks"].extend(out)
    return out, vias

WIDTH = {"5V": 20, "3V3": 20, "VIN12V": 20, "SFM_5V": 20}
TREE = None

def tree_cells():
    src = []
    for L in (TOP, BOT):
        jj, ii = np.nonzero(TREE[L])
        src += [(L, int(i), int(j)) for i, j in zip(ii, jj)]
    return src

def pad_touch(p, arr, n):
    for L in p["layers"]:
        i1, i2 = int(math.ceil((p["x"] - p["w"] / 2) / G)), int((p["x"] + p["w"] / 2) / G)
        j1, j2 = int(math.ceil((p["y"] - p["h"] / 2) / G)), int((p["y"] + p["h"] / 2) / G)
        if (arr[L][j1:j2 + 1, i1:i2 + 1] == n).any():
            return True
    return False

def add_pad_to_tree(p):
    for L in p["layers"]:
        paint_rect(L, p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, 1, TREE)

TREES, DONE = {}, {}

def route_net(net, only=None):
    global TREE
    n = nets[net]
    w = WIDTH.get(net, 10)
    pads = [p for p in PADS if p["nid"] == n]
    if net not in TREES:
        TREES[net] = [(HAND[0] == n).astype(np.int32), (HAND[1] == n).astype(np.int32)]
        DONE[net] = [p for p in pads if pad_touch(p, HAND, n)]
    TREE = TREES[net]
    done = DONE[net]
    todo = [p for p in pads if p not in done]
    if only:
        todo = [p for p in todo if f'{p["comp"]}.{p["name"]}' in only]
        # order as given
        todo.sort(key=lambda p: only.index(f'{p["comp"]}.{p["name"]}'))
    if not done:
        first = next((p for p in todo if p["comp"] == "U1"), todo[0])
        done = [first]; todo.remove(first)
    for p in done:
        add_pad_to_tree(p)
    fails = []
    while todo:
        if not only:
            u1 = [p for p in todo if p["comp"] == "U1"]
            todo.sort(key=lambda p: min(abs(p["x"] - q["x"]) + abs(p["y"] - q["y"]) for q in done))
            if u1:
                todo.remove(u1[0]); todo.insert(0, u1[0])
        p = todo.pop(0)
        src = tree_cells()
        if len(src) > 6000:
            src = src[::max(1, len(src) // 6000)]
        t0 = time.time()
        path, exp = astar(src, (p["x"], p["y"], p["layers"]), n, w)
        if path is None:
            path, exp = astar(src, (p["x"], p["y"], p["layers"]), n, w, maxexp=1_000_000, relaxed=True)
        if path is None:
            fails.append(f'{p["comp"]}.{p["name"]}')
            print(f"  FAIL {net} -> {p['comp']}.{p['name']} ({exp} exp)", flush=True)
            continue
        segs, vias = emit(path, n, net, w)
        add_pad_to_tree(p)
        done.append(p)
        print(f"  {net} -> {p['comp']}.{p['name']}: {len(segs)} segs {len(vias)} vias {exp} exp {time.time()-t0:.1f}s", flush=True)
    return fails

PLAN = ["EN", "PRESS_PRI_ADC", "PRESS_RED_ADC", "SAFETY_TRIP", "BUZZER_CTRL", "SFM_EN",
        "ENC_A", "ENC_B", "I2C_SDA", "BLOWER_PWM", "I2C_SCL", "ENC_SW", "BOOT", "UART_RX", "UART_TX",
        "BLOWER_FG", "5V", "3V3"]

if __name__ == "__main__":
    def hpwl(net):
        ps = [p for p in PADS if p["nid"] == nets[net]]
        return (max(p["x"] for p in ps) - min(p["x"] for p in ps)) + (max(p["y"] for p in ps) - min(p["y"] for p in ps))
    if len(sys.argv) > 1:
        plan = sys.argv[1].split(";")
    else:   # short local nets first, then rails, then long nets shortest-first
        rest = [k for k in nets if k not in ("GND", "5V", "3V3")]
        short = sorted([k for k in rest if hpwl(k) < 700], key=hpwl)
        long_ = sorted([k for k in rest if hpwl(k) >= 700], key=hpwl)
        plan = short + ["5V", "3V3"] + long_
        print("ORDER", plan, flush=True)
    seen = set()
    fails = {}
    steps = []
    for item in plan:
        net, _, only = item.partition(":")
        steps.append((net, only.split(",") if only else None)); seen.add(net)
    steps += [(k, None) for k in nets if k not in seen and k != "GND"]
    for net, only in steps:
        if net not in nets:
            continue
        f = route_net(net, only)
        if f:
            fails[net] = f
        json.dump(ROUTE, open("route.json", "w"))
    print("FAILS", fails, flush=True)
