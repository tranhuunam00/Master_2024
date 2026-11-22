"""Render placement (+ optional route json): python render.py out.png [route.json] [--win x0 y0 x1 y1] [--nets]"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from model import place_pads, place_body, FREEPADS
import importlib

def draw(out, P, route=None, win=None, labels=True, extra=None, title=""):
    fs = (24, 18) if not win else ((win[2] - win[0]) / 110.0, (win[3] - win[1]) / 110.0)
    fig, ax = plt.subplots(figsize=fs, dpi=90 if not win else 100)
    ax.set_facecolor("#0b0b0b")
    pads = []
    for d, (x, y, r) in P.items():
        pads += place_pads(d, x, y, r)
        b = place_body(d, x, y, r)
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#4c4", lw=0.8, zorder=6))
        if labels:
            ax.text((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, d, color="#7f7", fontsize=7 if not win else 10,
                    ha="center", va="center", zorder=9, alpha=0.9, clip_on=True)
    col = {"TopLayer": "#e23", "BottomLayer": "#39f"}
    if route:
        for poly in route.get("regions", []):
            ax.add_patch(Polygon(poly["pts"], closed=True, color=col[poly["layer"]], alpha=0.45, zorder=1 if poly["layer"] == "BottomLayer" else 2))
        for t in route.get("tracks", []):
            sc = 72 * fs[0] / ((win[2] - win[0]) if win else 4060) * 0.82
            ax.plot([t["x1"], t["x2"]], [t["y1"], t["y2"]], color=col[t["layer"]], alpha=0.85,
                    lw=t["width"] * sc, solid_capstyle="round", zorder=3 if t["layer"] == "TopLayer" else 1.5)
        for v in route.get("vias", []):
            ax.add_patch(Circle((v["x"], v["y"]), v["size"] / 2, color="#ccc", zorder=5))
            ax.add_patch(Circle((v["x"], v["y"]), v["hole"] / 2, color="#333", zorder=5.1))
    fp = [dict(f, comp="", ) for f in FREEPADS]
    for p in pads + fp:
        c = "#999" if p["layer"] == "MultiLayer" else "#d9b13b"
        if p.get("shape") in ("1", 1) and p["layer"] == "MultiLayer" and p.get("w", 0) < 150:
            ax.add_patch(Rectangle((p["x"] - p["w"] / 2, p["y"] - p["h"] / 2), p["w"], p["h"], color=c, zorder=4))
        elif p["layer"] == "MultiLayer":
            ax.add_patch(Circle((p["x"], p["y"]), p["w"] / 2, color=c, zorder=4))
        else:
            ax.add_patch(Rectangle((p["x"] - p["w"] / 2, p["y"] - p["h"] / 2), p["w"], p["h"], color=c, zorder=4))
        if win and p.get("net") and labels:
            ax.text(p["x"], p["y"], p["net"][:8], fontsize=6, color="k", ha="center", va="center", zorder=8, rotation=0, clip_on=True)
    if not route:
        nets = {}
        for p in pads + fp:
            if p.get("net") and p["net"] not in ("GND",) and not p["net"].startswith("Net"):
                nets.setdefault(p["net"], []).append(p)
        for n, ps in nets.items():
            if len(ps) < 2: continue
            rest, tree = ps[1:], [ps[0]]
            while rest:
                a, b = min(((a, b) for a in tree for b in rest), key=lambda ab: (ab[0]["x"] - ab[1]["x"]) ** 2 + (ab[0]["y"] - ab[1]["y"]) ** 2)
                ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#0ff" if n not in ("5V", "3V3", "VIN12V") else "#f80", lw=0.6, alpha=0.7, zorder=7)
                tree.append(b); rest.remove(b)
    if extra:
        extra(ax)
    ax.add_patch(Rectangle((0, 0), 4000, 3000, fill=False, ec="#f0f", lw=1))
    ax.add_patch(Rectangle((1560, 2745), 880, 255, fill=False, ec="#f0f", lw=0.8, ls="--"))
    for (mx, my) in ((150, 150), (3850, 150), (150, 2850), (3850, 2850)):
        ax.add_patch(Circle((mx, my), 150, fill=False, ec="#f0f", lw=0.6, ls=":"))
    ax.set_aspect("equal")
    if win:
        ax.set_xlim(win[0], win[2]); ax.set_ylim(win[1], win[3])
    else:
        ax.set_xlim(-30, 4030); ax.set_ylim(-30, 3030)
    ax.grid(color="#222", lw=0.4)
    if title: ax.set_title(title)
    plt.tight_layout(); plt.savefig(out); plt.close(fig)
    return out

if __name__ == "__main__":
    out = sys.argv[1]
    args = sys.argv[2:]
    route = None; win = None
    if args and args[0].endswith(".json"):
        route = json.load(open(args.pop(0)))
    if "--win" in args:
        i = args.index("--win"); win = [float(v) for v in args[i + 1:i + 5]]
    import place_v3
    print(draw(out, place_v3.P, route, win))
