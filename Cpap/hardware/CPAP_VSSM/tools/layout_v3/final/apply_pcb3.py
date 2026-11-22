"""v3 PCB: replace routing with route json; move parts to current placement; repour."""
import json, sys
sys.path.insert(0, "v3")
from edacall import *
import place_v3 as PL

def show(tag, r):
    print(tag, json.dumps(r, ensure_ascii=False)[:400] if not isinstance(r, str) else r[:400], flush=True)

call("app_set_active_document", {"file_path": BOARD})
show("unroute", call("pcb_unroute", {"expect_file": BOARD, "include_locked": True, "checkpoint": False}))
show("del regions", call("obj_delete", {"object_type": "eRegionObject", "filter": "Name~V3_*"}))
comps = call("pcb_get_components", {})["components"]
cur = {c["designator"]: (c["x"], c["y"], c["rotation"]) for c in comps}
moves = [dict(designator=d, x=v[0], y=v[1], rotation=v[2]) for d, v in PL.P.items()
         if d in cur and (abs(cur[d][0] - v[0]) > 0.05 or abs(cur[d][1] - v[1]) > 0.05 or (cur[d][2] - v[2]) % 360)]
print("moves", [m["designator"] for m in moves])
if moves:
    show("move", call("pcb_move_components", {"moves": moves}))
