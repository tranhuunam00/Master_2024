"""v3 PCB step 1: take up v2 routing, place C27 (synced), rebind nets, move parts and free pads."""
import json, sys
sys.path.insert(0, "v3")
from edacall import *
import place_v3 as PL
from model import CUR, FREEPADS

def show(tag, r):
    print(tag, json.dumps(r, ensure_ascii=False)[:500] if not isinstance(r, str) else r[:500], flush=True)

show("focus", call("app_set_active_document", {"file_path": BOARD}))
show("unroute", call("pcb_unroute", {"expect_file": BOARD, "include_locked": True, "checkpoint": True}))
show("del silk", call("obj_delete", {"items": [
    {"object_type": "eTextObject", "filter": "Layer=TopOverlay|InComponent=false"},
    {"object_type": "eTrackObject", "filter": "Layer=TopOverlay|InComponent=false"}]}))
show("C27", call("pcb_place_components", {"board_path": BOARD, "placements": [{
    "footprint": "R_0805", "library_path": PRJ + r"\SiPAP_Footprints.PcbLib", "x": PL.P["C27"][0], "y": PL.P["C27"][1],
    "rotation": PL.P["C27"][2], "designator": "C27", "lib_reference": "CAPACITOR", "comment": "22uF 16V",
    "layer": "TopLayer", "unique_id": "ASZKUQZD", "pad_nets": {"1": "3V3", "2": "GND"}}]}))
show("bind", call("pcb_build_from_project", {}))
moves = [dict(designator=d, x=v[0], y=v[1], rotation=v[2]) for d, v in PL.P.items() if d in CUR and tuple(v) != tuple(CUR[d])]
print(len(moves), "moves")
show("move", call("pcb_move_components", {"moves": moves}))
items = []
for f in FREEPADS:
    if (round(f["x"], 2), round(f["y"], 2)) != (round(f["ox"], 2), round(f["oy"], 2)):
        items.append({"object_type": "ePadObject", "filter": f"InComponent=false|X={f['ox']}|Y={f['oy']}", "set": f"X={f['x']}|Y={f['y']}"})
print(len(items), "free pad moves")
show("freepads", call("obj_modify", {"items": items}))
