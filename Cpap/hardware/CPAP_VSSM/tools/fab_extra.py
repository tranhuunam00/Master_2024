"""Write the IPC-D-356A netlist and a generic pick-and-place (CPL) file for SiPAP Rev C.

Input: an obj_query dump (pads, vias, components) of SiPAP_PCB.PcbDoc.
Usage: python fab_extra.py <query_dump.json> <out_dir>
"""
import csv, datetime, json, os, sys

dump = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
pads, vias, comps = [it["objects"] for it in dump["items"]]
f = float

# ---------------- IPC-D-356A (units: CUST 0 = inch, 0.0001 in) ----------------
def tenth_mil(v):          # mils -> 0.0001 in
    return int(round(f(v) * 10))

def coord(v):
    n = tenth_mil(v)
    return ("+" if n >= 0 else "-") + f"{abs(n):06d}"

def rec(op, net, ref, pin, hole, plated, access, x, y, sx, sy, rot):
    net = (net or "N/C")[:14]
    s = f"{op:3s}{net:<14s}   {ref:<6s}{'-' if pin else ' '}{pin:<4s} "
    s += (f"D{tenth_mil(hole):04d}{'P' if plated else 'U'}" if f(hole) > 0 else "      ")
    s += f"A{access:02d}X{coord(x)}Y{coord(y)}X{tenth_mil(sx):04d}Y{tenth_mil(sy):04d}R{int(f(rot)) % 360:03d} S0"
    return s

lines = ["C  IPC-D-356A netlist - SiPAP Rev C (generated from SiPAP_PCB.PcbDoc)",
         f"C  {datetime.date.today().isoformat()}",
         "P  JOB   SIPAP_REVC",
         "P  UNITS CUST 0",
         "P  DIM   N",
         "P  VER   IPC-D-356A",
         "P  IMAGE PRIMARY"]
nets = {}
for p in pads:
    if p["Name"].startswith("FID"):            # fiducials carry no net
        continue
    tht = f(p["HoleSize"]) > 0
    ref = p["Component"] or p["Name"]
    pin = p["Name"] if p["Component"] else ""
    op = "317" if tht else "327"
    access = 0 if tht else 1
    plated = str(p.get("Plated", "True")).lower() != "false"
    lines.append(rec(op, p["Net"], ref, pin, p["HoleSize"], plated, access, p["X"], p["Y"],
                     p["TopXSize"], p["TopYSize"], p["Rotation"]))
    if p["Net"]:
        nets.setdefault(p["Net"], 0); nets[p["Net"]] += 1
for v in vias:
    lines.append(rec("317", v["Net"], "VIA", "", v["HoleSize"], True, 0, v["X"], v["Y"], v["Size"], v["Size"], 0))
lines.append("999")
open(os.path.join(out, "SiPAP_RevC.ipc"), "w", newline="\r\n").write("\n".join(lines) + "\n")

# ---------------- Pick and place (generic CPL, mm, top side) ----------------
THT = {"J1", "J2", "J3", "U4", "U5", "U6", "F1", "BZ1", "C25"}   # hand / wave soldered
with open(os.path.join(out, "SiPAP_RevC_CPL.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Designator", "Comment", "Footprint", "Mid X (mm)", "Mid Y (mm)", "Rotation", "Layer", "Mount"])
    for c in sorted(comps, key=lambda c: (c["Designator"][0], int("".join(ch for ch in c["Designator"] if ch.isdigit()) or 0))):
        d = c["Designator"]
        w.writerow([d, c["Comment"], c["Pattern"], f"{f(c['X']) * 0.0254:.3f}", f"{f(c['Y']) * 0.0254:.3f}",
                    int(f(c["Rotation"])) % 360, "Top" if "Top" in c["Layer"] else "Bottom",
                    "THT (not machine placed)" if d in THT else "SMD"])
print(len(pads), "pads", len(vias), "vias", len(comps), "parts;", len(nets), "nets in IPC-356")
