"""Quick preview renders of the enclosure STLs (matplotlib, no CAD GUI needed).

python render_preview.py <out_dir>
Writes preview_open.png (lid off, iso), preview_closed.png (iso) and preview_top.png.
"""
import struct
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"


def load_stl(path):
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        tris, cur = [], []
        for line in data.decode(errors="ignore").splitlines():
            p = line.split()
            if p and p[0] == "vertex":
                cur.append([float(v) for v in p[1:4]])
                if len(cur) == 3:
                    tris.append(cur)
                    cur = []
        return np.array(tris)
    n = struct.unpack("<I", data[80:84])[0]
    arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return arr["v"].reshape(-1, 3, 3).astype(float)


PARTS = {  # name: (color, alpha)
    "sipap_base": ("#9aa7b8", 0.35),
    "sipap_lid": ("#c9d1dc", 0.30),
    "sipap_tub": ("#5fa8d3", 0.55),
    "sipap_tub_lid": ("#8cc4e6", 0.45),
    "ref_blower": ("#e07a2f", 0.9),
    "ref_sfm3300": ("#3f9b6b", 0.9),
    "ref_pcb": ("#2f6b3a", 0.95),
}


def shade(tris, color):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    light = np.array([0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    k = 0.45 + 0.55 * np.abs(n @ light)
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return np.clip(rgb[None, :] * k[:, None], 0, 1)


def render(names, fname, elev, azim, title):
    fig = plt.figure(figsize=(12, 6), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    allpts = []
    for name in names:
        path = os.path.join(OUT, name + ".stl")
        if not os.path.exists(path):
            continue
        tris = load_stl(path)
        color, alpha = PARTS[name]
        pc = Poly3DCollection(tris, facecolors=shade(tris, color), edgecolors="none", alpha=alpha)
        ax.add_collection3d(pc)
        allpts.append(tris.reshape(-1, 3))
    pts = np.vstack(allpts)
    mn, mx = pts.min(0), pts.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    ax.set_box_aspect(mx - mn)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)


inside = ["sipap_base", "sipap_tub", "sipap_tub_lid", "ref_blower", "ref_sfm3300", "ref_pcb"]
render(inside, "preview_open.png", 32, -60, "SiPAP enclosure v0.1 - lid removed")
render(inside + ["sipap_lid"], "preview_closed.png", 25, -55, "SiPAP enclosure v0.1 - closed")
render(inside, "preview_top.png", 90, -90, "Top view: electronics | blower | humidifier")
print("ok")
