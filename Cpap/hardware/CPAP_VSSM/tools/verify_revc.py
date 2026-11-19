"""Independent cross-checks of SiPAP Rev C (outside Altium's own ERC/DRC).

1. Every PCB pad net == schematic netlist (template nets), pin by pin.
2. Footprint outline overlap (pads + body silk, without designator/comment text).
Usage: python verify_revc.py <pad_dump.json> <components_dump.json> <template.json>
"""
import itertools, json, sys

pads_dump = json.load(open(sys.argv[1], encoding="utf-8"))
pads = pads_dump["items"][0]["objects"]
comps = json.load(open(sys.argv[2], encoding="utf-8"))["items"][0]["components"]
tpl = json.load(open(sys.argv[3], encoding="utf-8"))

# ---- 1. netlist ----
sch = {}
for net, pins in tpl["nets"].items():
    for p in pins:
        sch[p] = net
pcb = {}
for p in pads:
    if p["Component"]:
        key = f"{p['Component']}.{p['Name']}"
        pcb.setdefault(key, set()).add(p["Net"])
bad = []
for key, nets in sorted(pcb.items()):
    want = sch.get(key, "")
    if nets != {want}:
        if key.split(".")[1] == "MP" and nets == {""}:
            continue                      # encoder mounting tabs, mechanical
        bad.append((key, sorted(nets), want))
missing = [k for k in sch if k not in pcb]
print(f"[netlist] {len(pcb)} PCB pins checked against {len(sch)} schematic pins")
print("  mismatches:", bad or "none")
print("  schematic pins with no pad:", missing or "none")

# ---- 2. outline overlap ----
def box(c):
    b = c["bbox_no_name_comment"]
    return b["x1"], b["y1"], b["x2"], b["y2"]
over = []
for a, b in itertools.combinations(comps, 2):
    ax1, ay1, ax2, ay2 = box(a); bx1, by1, bx2, by2 = box(b)
    dx = min(ax2, bx2) - max(ax1, bx1); dy = min(ay2, by2) - max(ay1, by1)
    if dx > 5 and dy > 5:
        over.append((a["designator"], b["designator"], round(dx), round(dy)))
print(f"[outline] {len(comps)} parts; overlapping outlines (>5 mil):")
for o in sorted(over):
    print("  ", o)
