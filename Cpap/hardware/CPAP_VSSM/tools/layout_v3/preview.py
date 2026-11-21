"""Preview placement (+ optional routing json): python preview.py out.png [route.json] [x0 y0 x1 y1]"""
import sys, json, itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from model import place_pads, place_body, FREEPADS
from placement import P

out = sys.argv[1]
route = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[2].endswith(".json") else None
win = [float(v) for v in sys.argv[-4:]] if len(sys.argv) >= 6 else None
fig, ax = plt.subplots(figsize=(24, 18) if not win else (16, 16 * (win[3] - win[1]) / (win[2] - win[0])), dpi=100)
ax.set_facecolor("#0b0b0b")
pads = []
for d, (x, y, r) in P.items():
    pads += place_pads(d, x, y, r)
    b = place_body(d, x, y, r)
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#5f5", lw=0.7, zorder=6))
    ax.text((b[0] + b[2]) / 2, b[3] + 4, d, color="#5f5", fontsize=8 if not win else 11, ha="center", va="bottom", zorder=7)
if route:
    col = {"TopLayer": "#e23", "BottomLayer": "#29f"}
    for poly in route.get("regions", []):
        ax.add_patch(Polygon(poly["pts"], closed=True, color=col[poly["layer"]], alpha=0.55, zorder=1))
    for t in route.get("tracks", []):
        ax.plot([t["x1"], t["x2"]], [t["y1"], t["y2"]], color=col[t["layer"]], alpha=0.8,
                lw=t["width"] * (0.072 if not win else 72 * 16 / (win[2] - win[0])), solid_capstyle="round", zorder=2 if t["layer"] == "TopLayer" else 1.5)
    for v in route.get("vias", []):
        ax.add_patch(Circle((v["x"], v["y"]), v["size"] / 2, color="#ddd", zorder=5))
for p in pads + [dict(f, comp="", name="") for f in FREEPADS]:
    c = "#aaa" if p["layer"] == "MultiLayer" else "#db3"
    ax.add_patch(Rectangle((p["x"] - p["w"] / 2, p["y"] - p["h"] / 2), p["w"], p["h"], color=c, zorder=4))
    if win and p["net"]:
        ax.text(p["x"], p["y"], p["net"][:7], fontsize=6, color="k", ha="center", va="center", zorder=8)
if not route:  # ratsnest (star to first pad) for a feel of the flow
    nets = {}
    for p in pads:
        if p["net"] and p["net"] != "GND" and not p["net"].startswith("Net"):
            nets.setdefault(p["net"], []).append(p)
    for n, ps in nets.items():
        rest = ps[1:]
        tree = [ps[0]]
        while rest:  # prim
            a, b = min(((a, b) for a in tree for b in rest), key=lambda ab: (ab[0]["x"] - ab[1]["x"]) ** 2 + (ab[0]["y"] - ab[1]["y"]) ** 2)
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#0ff", lw=0.5, alpha=0.6, zorder=3)
            tree.append(b); rest.remove(b)
ax.add_patch(Rectangle((0, 0), 4000, 3000, fill=False, ec="#f0f", lw=1))
ax.set_aspect("equal")
if win:
    ax.set_xlim(win[0], win[2]); ax.set_ylim(win[1], win[3])
else:
    ax.set_xlim(-30, 4030); ax.set_ylim(-30, 3030)
ax.grid(color="#222", lw=0.4)
plt.tight_layout(); plt.savefig(out); print("ok", out)
