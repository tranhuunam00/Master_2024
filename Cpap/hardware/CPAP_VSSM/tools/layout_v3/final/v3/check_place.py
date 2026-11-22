"""Placement check: courtyard (body + pads) overlaps, pad-to-pad gaps < 10 mil, off-board, keepouts."""
import itertools, importlib
from shapely.geometry import box, Point
from shapely.ops import unary_union
from model import place_pads, place_body, FREEPADS

def courtyard(d, x, y, r, m=5):
    b = place_body(d, x, y, r)
    g = [box(b[0] - m, b[1] - m, b[2] + m, b[3] + m)]
    for p in place_pads(d, x, y, r):
        g.append(box(p["x"] - p["w"] / 2 - m, p["y"] - p["h"] / 2 - m, p["x"] + p["w"] / 2 + m, p["y"] + p["h"] / 2 + m))
    return unary_union(g)

def run(P, verbose=True):
    cy = {d: courtyard(d, *v) for d, v in P.items()}
    probs = []
    for a, b in itertools.combinations(P, 2):
        if cy[a].intersects(cy[b]) and cy[a].intersection(cy[b]).area > 1:
            probs.append(f"overlap {a}-{b}")
    pads = {d: place_pads(d, *v) for d, v in P.items()}
    for d, v in P.items():
        bb = cy[d].bounds
        if bb[0] < 20 or bb[1] < 20 or bb[2] > 3980 or bb[3] > 2980:
            if d not in ("J1",):
                probs.append(f"edge {d} {tuple(round(t) for t in bb)}")
        for mx, my in ((150, 150), (3850, 150), (150, 2850), (3850, 2850)):
            if cy[d].distance(Point(mx, my)) < 140:
                probs.append(f"MH {d}")
        if d != "U1" and cy[d].intersects(box(1560, 2745, 2440, 3000)):
            probs.append(f"antenna-keepout {d}")
    import place_v3
    def fpos(f):
        if f["name"].startswith("TP") and f["net"] in place_v3.TP_MOVE: return place_v3.TP_MOVE[f["net"]]
        if f["name"] in place_v3.FID_MOVE: return place_v3.FID_MOVE[f["name"]]
        return (f["x"], f["y"])
    fp = unary_union([Point(*fpos(f)).buffer(f["w"] / 2 + 10) for f in FREEPADS if f["name"].startswith(("TP", "FID"))])
    for d in P:
        if cy[d].intersects(fp):
            probs.append(f"testpad/fid near {d}")
    if verbose:
        print(len(probs), "problems"); [print(" ", p) for p in probs]
    return probs

if __name__ == "__main__":
    import place_v3
    run(place_v3.P)

def hand_vs_pads(P):
    """pads of other nets closer than 10 mil to hand-drawn copper."""
    import hand
    from shapely.geometry import LineString, Polygon, Point, box
    geo = []
    for t in hand.T:
        g = LineString([(t["x1"], t["y1"]), (t["x2"], t["y2"])]).buffer(t["width"] / 2) if (t["x1"], t["y1"]) != (t["x2"], t["y2"]) else Point(t["x1"], t["y1"]).buffer(t["width"] / 2)
        geo.append((g, t["net"], t["layer"]))
    for rg in hand.R:
        geo.append((Polygon(rg["pts"]), rg["net"], rg["layer"]))
    for v in hand.V:
        for L in ("TopLayer", "BottomLayer"):
            geo.append((Point(v["x"], v["y"]).buffer(v["size"] / 2), v["net"], L))
    bad = []
    for d, v in P.items():
        for p in place_pads(d, *v):
            pg = box(p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2)
            layers = ("TopLayer", "BottomLayer") if p["layer"] == "MultiLayer" else (p["layer"],)
            for g, net, L in geo:
                if L in layers and net != p["net"] and g.distance(pg) < 10:
                    bad.append((d, p["name"], p["net"], net, round(g.distance(pg), 1)))
    return bad

if __name__ == "__main__":
    import place_v3
    hb = hand_vs_pads(place_v3.P)
    print(len(hb), "pad-vs-hand-copper problems"); [print(" ", b) for b in hb]
