"""Rip-up and reroute for connections the first pass could not make. Reads/writes route.json."""
import json, sys
import numpy as np
import router as R

route = json.load(open("route.json"))
_ht = [json.dumps(t, sort_keys=True) for t in R.power.T]
_hv = [json.dumps(v, sort_keys=True) for v in R.power.V]
route["tracks"] = list(R.power.T) + [t for t in route["tracks"] if json.dumps(t, sort_keys=True) not in _ht]
route["vias"] = list(R.power.V) + [v for v in route["vias"] if json.dumps(v, sort_keys=True) not in _hv]
route["regions"] = list(R.power.R)
NH_T, NH_V = len(R.power.T), len(R.power.V)
R.ROUTE = route

def repaint_all():
    """rebuild copper from pads + hand copper + routed copper; rebuild per-net trees."""
    for L in (0, 1):
        R.own[L][:] = 0
    for p in R.PADS:
        for L in p["layers"]:
            R.paint_rect(L, p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2, p["nid"])
    for L in (0, 1):
        m = R.HAND[L] != 0
        R.own[L][m] = R.HAND[L][m]
    for t in route["tracks"][NH_T:]:
        R.paint_seg(0 if t["layer"] == "TopLayer" else 1, t["x1"], t["y1"], t["x2"], t["y2"], t["width"], R.nid(t["net"]))
    for v in route["vias"][NH_V:]:
        for L in (0, 1):
            R.paint_seg(L, v["x"], v["y"], v["x"], v["y"], v["size"], R.nid(v["net"]))

def reset_trees():
    R.TREES.clear(); R.DONE.clear()
    for net in R.nets:
        if net == "GND":
            continue
        n = R.nets[net]
        tree = [(R.HAND[0] == n).astype(np.int32), (R.HAND[1] == n).astype(np.int32)]
        for t in route["tracks"][NH_T:]:
            if t["net"] == net:
                R.paint_seg(0 if t["layer"] == "TopLayer" else 1, t["x1"], t["y1"], t["x2"], t["y2"], t["width"], 1, tree)
        for v in route["vias"][NH_V:]:
            if v["net"] == net:
                for L in (0, 1):
                    R.paint_seg(L, v["x"], v["y"], v["x"], v["y"], v["size"], 1, tree)
        pads = [p for p in R.PADS if p["nid"] == n]
        done = [p for p in pads if R.pad_touch(p, [tree[0] * n, tree[1] * n], n)]
        R.TREES[net], R.DONE[net] = tree, done
        R.TREE = tree
        for p in done:
            R.add_pad_to_tree(p)

def rip(net):
    route["tracks"][NH_T:] = [t for t in route["tracks"][NH_T:] if t["net"] != net]
    route["vias"][NH_V:] = [v for v in route["vias"][NH_V:] if v["net"] != net]

def unconnected():
    out = {}
    for net in R.nets:
        if net == "GND":
            continue
        n = R.nets[net]
        pads = [p for p in R.PADS if p["nid"] == n]
        miss = [p for p in pads if p not in R.DONE[net]]
        if miss and len(pads) > 1:
            out[net] = miss
    return out

def blockers(net, p):
    """route p with other routed copper as SOFT obstacles; return the nets the path crosses."""
    from scipy.ndimage import distance_transform_edt
    n = R.nets[net]
    w = R.WIDTH.get(net, 10)
    saved = [R.own[0].copy(), R.own[1].copy()]
    routed = set(t["net"] for t in route["tracks"][NH_T:]) | set(v["net"] for v in route["vias"][NH_V:])
    routed.discard(net)
    rmask = [np.zeros_like(R.own[0], bool), np.zeros_like(R.own[1], bool)]
    for L in (0, 1):
        for other in routed:
            m = (R.own[L] == R.nets[other]) & (R.HAND[L] != R.nets[other])
            rmask[L] |= m
            R.own[L][m] = 0
    for q in R.PADS:
        for L in q["layers"]:
            R.paint_rect(L, q["x"] - q["w"] / 2, q["y"] - q["h"] / 2, q["x"] + q["w"] / 2, q["y"] + q["h"] / 2, q["nid"])
    masks = R.masks_for(n, w)
    r = w / 2 + R.CLR + R.G / 2 + 0.5
    soft = []
    for L in (0, 1):
        d = distance_transform_edt(~rmask[L]) * R.G
        soft.append((d < r).ravel().tobytes())
    R.own[0][:], R.own[1][:] = saved
    R.TREE = R.TREES[net]
    src = R.tree_cells()
    if len(src) > 6000:
        src = src[::len(src) // 6000]
    R.SOFT = soft
    path, _ = R.astar(src, (p["x"], p["y"], p["layers"]), n, w, maxexp=2_500_000, relaxed=True, masks=masks)
    R.SOFT = None
    if not path:
        return None
    hit = set()
    rr = int(r / R.G) + 1
    for (L, i, j, d) in path:
        win = saved[L][max(0, j - rr):j + rr + 1, max(0, i - rr):i + rr + 1]
        rm = rmask[L][max(0, j - rr):j + rr + 1, max(0, i - rr):i + rr + 1]
        for v in np.unique(win[rm]):
            if v > 0 and v != n:
                hit.add(next(k for k, x in R.nets.items() if x == v))
    return hit

for it in range(int(sys.argv[1]) if len(sys.argv) > 1 and __name__ == "__main__" else 0):
    repaint_all(); reset_trees()
    miss = unconnected()
    print(f"iter {it}: unconnected {[(k, [q['comp'] + '.' + q['name'] for q in v]) for k, v in miss.items()]}", flush=True)
    if not miss:
        break
    net, pads = next(iter(miss.items()))
    hit = blockers(net, pads[0])
    print("  blockers", hit, flush=True)
    if hit is None:
        print("  no path even alone; skip", net); break
    for h in hit:
        rip(h)
    repaint_all(); reset_trees()
    R.route_net(net)                                  # the failed net first
    for h in sorted(hit, key=lambda k: len([p for p in R.PADS if p["nid"] == R.nets[k]])):
        R.route_net(h)
    json.dump(route, open("route.json", "w"))
if __name__ == "__main__":
    repaint_all(); reset_trees()
    print("FINAL unconnected", {k: [q["comp"] + "." + q["name"] for q in v] for k, v in unconnected().items()})
    json.dump(route, open("route.json", "w"))
