"""Write the eda-agent argument files that put placement + route.json on the board."""
import json
from placement import P
from model import CUR, FREEPADS

BOARD = "%PRJ%/SiPAP_PCB.PcbDoc"
route = json.load(open("route.json"))

moves = [dict(designator=d, x=round(x, 3), y=round(y, 3), rotation=r) for d, (x, y, r) in P.items()
         if (x, y, r) != CUR[d]]
json.dump(dict(moves=moves), open("a_moves.json", "w"))

tp = []
for f in FREEPADS:
    if "ox" in f:
        tp.append(dict(object_type="ePadObject", filter=f"X={f['ox']}|Y={f['oy']}|InComponent=false",
                       set=f"X={f['x']}|Y={f['y']}"))
json.dump(dict(items=tp), open("a_tp.json", "w"))

regs = [dict(points=[[x, y] for x, y in r["pts"]], net=r["net"], layer=r["layer"], name="V3_" + r["net"])
        for r in route["regions"]]
json.dump(dict(items=regs), open("a_regions.json", "w"))

json.dump(dict(tracks=[dict(x1=t["x1"], y1=t["y1"], x2=t["x2"], y2=t["y2"], width=t["width"],
                            layer=t["layer"], net_name=t["net"]) for t in route["tracks"]],
               board_path=BOARD), open("a_tracks.json", "w"))
json.dump(dict(vias=[dict(x=v["x"], y=v["y"], size=v["size"], hole_size=v["hole"], net=v["net"])
                     for v in route["vias"]], board_path=BOARD), open("a_vias.json", "w"))
print(len(moves), "moves", len(tp), "test pads", len(regs), "regions", len(route["tracks"]), "tracks", len(route["vias"]), "vias")
