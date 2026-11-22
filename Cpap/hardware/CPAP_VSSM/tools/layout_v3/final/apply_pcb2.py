"""v3 PCB step 2: place regions, tracks, vias from the verified route json."""
import json, sys
from edacall import *
route = json.load(open(sys.argv[1]))
def show(tag, r):
    print(tag, json.dumps(r, ensure_ascii=False)[:600] if not isinstance(r, str) else r[:600], flush=True)
call("app_set_active_document", {"file_path": BOARD})
regs = [dict(points=[[x, y] for x, y in r["pts"]], net=r["net"], layer=r["layer"], name="V3_" + r["net"]) for r in route["regions"]]
show("regions", call("pcb_place_region", {"items": regs}))
tr = [dict(x1=t["x1"], y1=t["y1"], x2=t["x2"], y2=t["y2"], width=t["width"], layer=t["layer"], net_name=t["net"]) for t in route["tracks"]]
for k in range(0, len(tr), 200):
    show(f"tracks {k}", call("pcb_place_tracks", {"tracks": tr[k:k + 200], "board_path": BOARD}))
vi = [dict(x=v["x"], y=v["y"], size=v["size"], hole_size=v["hole"], net=v["net"]) for v in route["vias"]]
show("vias", call("pcb_place_vias", {"vias": vi, "board_path": BOARD}))
