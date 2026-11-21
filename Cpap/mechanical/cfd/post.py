"""Post-process the SiPAP air-path CFD runs from the ASCII OpenFOAM files.

    python post.py <cases_root> <airpath.json> <out_dir>

(OpenFOAM's own function objects crash in the Ubuntu v1912 build, so this script
reads the mesh and the final p/U fields directly.)

For each case qNN (NN = L/min):
  * static and total pressure at the inlet (area-weighted over the owner cells of
    the inlet faces; p is kinematic in simpleFoam -> x rho 1.2 kg/m3); outlet p = 0
  * volume flow through inlet and outlet (check of continuity)
  * p and |U| at the probe points along the path (nearest cell centre)
  * the SFM3300 datasheet pressure drop is added (its internal flow conditioner
    is not in the model): typ 180 Pa, max 230 Pa at 60 slm (DS table 1.1),
    scaled ~Q^1.6 between the 60 and 100 slm points (380 / 550 Pa)
  * section images of |U| (slab plots of cell centres)
"""
import json
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO = 1.2
PA_PER_CMH2O = 98.0665


def _body(path):
    txt = open(path, encoding="latin-1").read()
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//[^\n]*", "", txt)
    i = txt.index("FoamFile")
    j = txt.index("}", i)
    return txt[j + 1:]


def read_list(path, kind):
    b = _body(path)
    m = re.search(r"(\d+)\s*\(", b)
    n = int(m.group(1))
    s = b[m.end():]
    if kind == "label":
        return np.array(s[:s.rindex(")")].split()[:n], dtype=np.int64)
    if kind == "vector":
        v = re.findall(r"\(([^()]*)\)", s)[:n]
        return np.array([x.split() for x in v], dtype=float)
    if kind == "faces":
        faces = re.findall(r"\d+\(([^()]*)\)", s)[:n]
        return [np.array(f.split(), dtype=np.int64) for f in faces]
    raise ValueError(kind)


def read_boundary(path):
    b = _body(path)
    out = {}
    for name, body in re.findall(r"(\w+)\s*\{([^{}]*)\}", b):
        nf = re.search(r"nFaces\s+(\d+)", body)
        sf = re.search(r"startFace\s+(\d+)", body)
        if nf and sf:
            out[name] = (int(sf.group(1)), int(nf.group(1)))
    return out


def read_field(path, vector):
    b = _body(path)
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(", b)
    n = int(m.group(1))
    s = b[m.end():]
    if vector:
        v = re.findall(r"\(([^()]*)\)", s[: s.index("boundaryField") if "boundaryField" in s else None])[:n]
        return np.array([x.split() for x in v], dtype=float)
    end = s.index(")")
    return np.array(s[:end].split()[:n], dtype=float)


def latest_time(case):
    ts = [d for d in os.listdir(case) if re.fullmatch(r"\d+(\.\d+)?", d) and d != "0"]
    return max(ts, key=float)


def face_geom(points, faces):
    ctr = np.array([points[f].mean(0) for f in faces])
    area = np.zeros((len(faces), 3))
    for i, f in enumerate(faces):
        p = points[f]
        c = ctr[i]
        a = np.zeros(3)
        for k in range(len(f)):
            a += np.cross(p[k] - c, p[(k + 1) % len(f)] - c)
        area[i] = 0.5 * a
    return ctr, area


def sfm_dp(q_lpm):
    """SFM3300 datasheet pressure drop (typ, max) in Pa; power-law through 60/100 slm points."""
    nt = math.log(380 / 180) / math.log(100 / 60)
    nm = math.log(550 / 230) / math.log(100 / 60)
    return 180 * (q_lpm / 60) ** nt, 230 * (q_lpm / 60) ** nm


def slab(ax, cc, mag, axis, value, half, uv, title, vmax):
    m = np.abs(cc[:, axis] - value) < half
    a, b = uv
    sc = ax.scatter(cc[m, a] * 1000, cc[m, b] * 1000, c=mag[m], s=1.2, cmap="turbo", vmin=0, vmax=vmax, marker="s")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("xyz"[a] + " [mm]")
    ax.set_ylabel("xyz"[b] + " [mm]")
    return sc


def main(root, info_path, out):
    os.makedirs(out, exist_ok=True)
    info = json.load(open(info_path))
    mesh = os.path.join(root, "q60", "constant", "polyMesh")
    pts = read_list(os.path.join(mesh, "points"), "vector")
    faces = read_list(os.path.join(mesh, "faces"), "faces")
    own = read_list(os.path.join(mesh, "owner"), "label")
    nei = read_list(os.path.join(mesh, "neighbour"), "label")
    bnd = read_boundary(os.path.join(mesh, "boundary"))
    ncell = int(own.max()) + 1
    fc, fa = face_geom(pts, faces)
    # cell centres ~ mean of face centres
    acc = np.zeros((ncell, 3))
    cnt = np.zeros(ncell)
    np.add.at(acc, own, fc)
    np.add.at(cnt, own, 1)
    np.add.at(acc, nei, fc[: len(nei)])
    np.add.at(cnt, nei, 1)
    cc = acc / cnt[:, None]
    probes = {k: np.array(v) / 1000.0 for k, v in info["probes_mm"].items()}
    pidx = {k: int(np.argmin(((cc - p) ** 2).sum(1))) for k, p in probes.items()}

    results = {"n_cells": ncell, "rho": RHO, "cases": {}}
    for d in sorted(os.listdir(root)):
        m = re.fullmatch(r"q(\d+)", d)
        if not m:
            continue
        q = float(m.group(1))
        case = os.path.join(root, d)
        try:
            t = latest_time(case)
        except ValueError:
            continue
        p = read_field(os.path.join(case, t, "p"), False) * RHO
        U = read_field(os.path.join(case, t, "U"), True)
        mag = np.linalg.norm(U, axis=1)
        r = {"iterations": int(float(t))}
        for patch in ("inlet", "outlet"):
            s, n = bnd[patch]
            fi = np.arange(s, s + n)
            A = np.linalg.norm(fa[fi], axis=1)
            c = own[fi]
            nrm = fa[fi] / A[:, None]
            flow = float(-(U[c] * nrm).sum(1) @ A) if patch == "inlet" else float((U[c] * nrm).sum(1) @ A)
            ps = float((p[c] * A).sum() / A.sum())
            pt = float(((p[c] + 0.5 * RHO * mag[c] ** 2) * A).sum() / A.sum())
            r[patch] = {"area_mm2": round(A.sum() * 1e6, 2), "p_static_Pa": round(ps, 1),
                        "p_total_Pa": round(pt, 1), "flow_L_min": round(flow * 60000, 2)}
        dp_static = r["inlet"]["p_static_Pa"]                       # outlet p = 0
        dp_total = r["inlet"]["p_total_Pa"] - r["outlet"]["p_total_Pa"]
        st, sm = sfm_dp(q)
        r["dp_model_static_Pa"] = round(dp_static, 1)
        r["dp_model_total_Pa"] = round(dp_total, 1)
        r["dp_sfm_datasheet_typ_max_Pa"] = [round(st), round(sm)]
        r["dp_total_path_typ_Pa"] = round(dp_static + st, 1)
        r["dp_total_path_max_Pa"] = round(dp_static + sm, 1)
        r["dp_total_path_typ_cmH2O"] = round((dp_static + st) / PA_PER_CMH2O, 2)
        r["dp_total_path_max_cmH2O"] = round((dp_static + sm) / PA_PER_CMH2O, 2)
        r["probes"] = {k: {"p_Pa": round(float(p[i]), 1), "U_m_s": round(float(mag[i]), 2)} for k, i in pidx.items()}
        log = os.path.join(case, "log.simpleFoam")
        if os.path.exists(log):
            txt = open(log, encoding="latin-1").read()
            res = {}
            for fld in ("Ux", "Uy", "Uz", "p", "k", "omega"):
                mm = re.findall(r"Solving for %s, Initial residual = ([0-9.eE+-]+)" % fld, txt)
                if mm:
                    res[fld] = float(mm[-1])
            r["final_initial_residuals"] = res
            r["converged_by_residualControl"] = "reached convergence" in txt
        results["cases"][d] = r

        vmax = float(np.percentile(mag, 99.5))
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
        slab(axs[0, 0], cc, mag, 1, 0.100, 0.0006, (0, 2), "Section y = 100 mm: duct riser -> SFM3300 -> dock -> tub inlet", vmax)
        slab(axs[0, 1], cc, mag, 2, info["probes_mm"]["p01_inlet_spigot"][2] / 1000, 0.0006, (0, 1),
             "Section z = %.1f mm: blower spigot + duct along the floor" % info["probes_mm"]["p01_inlet_spigot"][2], vmax)
        mi = info["probes_mm"]["p07_inlet_snorkel"]
        slab(axs[1, 0], cc, mag, 1, mi[1] / 1000, 0.0006, (0, 2), "Tub section y = %.0f mm: inlet snorkel mouth, water at z = %.1f" % (mi[1], info["water_z_mm"]), vmax)
        sc = slab(axs[1, 1], cc, mag, 2, (info["water_z_mm"] + 4.0) / 1000, 0.0006, (0, 1),
                  "Tub plan z = %.1f mm (4 mm above the water)" % (info["water_z_mm"] + 4.0), vmax)
        fig.colorbar(sc, ax=axs, shrink=0.7, label="|U| [m/s]")
        fig.suptitle("SiPAP air path, %d L/min: velocity magnitude (OpenFOAM simpleFoam, k-omega SST, %d cells)" % (q, ncell))
        fig.savefig(os.path.join(out, "cfd_U_%s.png" % d), bbox_inches="tight")
        plt.close(fig)

    # pressure along the path
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    for d, r in results["cases"].items():
        ks = list(r["probes"])
        ax.plot(range(len(ks)), [r["probes"][k]["p_Pa"] for k in ks], "o-", label="%s L/min" % d[1:])
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([k[4:] for k in ks], rotation=35, ha="right")
    ax.set_ylabel("static p above outlet [Pa]  (SFM3300 internal drop NOT included)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Static pressure along the air path (CFD)")
    fig.savefig(os.path.join(out, "cfd_pressure_along_path.png"), bbox_inches="tight")
    plt.close(fig)
    json.dump(results, open(os.path.join(out, "cfd_results.json"), "w"), indent=1)
    print(json.dumps({d: {k: r[k] for k in ("iterations", "dp_model_static_Pa", "dp_total_path_typ_cmH2O",
                                             "dp_total_path_max_cmH2O")} | {"flow_in": r["inlet"]["flow_L_min"],
                                                                             "flow_out": r["outlet"]["flow_L_min"]}
                      for d, r in results["cases"].items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
