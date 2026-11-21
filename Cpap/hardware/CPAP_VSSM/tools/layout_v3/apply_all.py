"""Put placement + route.json on the board in ONE eda-agent session (one process)."""
import json, asyncio, os, logging, sys, time
os.environ.setdefault("EDA_AGENT_NO_DASHBOARD", "1")
logging.disable(logging.CRITICAL)
from eda_agent import server

PRJ = r"C:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\hardware\CPAP_VSSM"
BOARD = PRJ + r"\SiPAP_PCB.PcbDoc"

def load(f):
    s = open(f, encoding="utf-8").read().replace("%PRJ%", PRJ.replace("\\", "\\\\"))
    return json.loads(s)

async def call(name, args):
    t0 = time.time()
    res = await server.mcp.call_tool(name, args)
    if isinstance(res, tuple):
        res = res[0]
    txt = "\n".join(getattr(b, "text", str(b)) for b in res)
    print(f"== {name} ({time.time()-t0:.1f}s): {txt[:600]}", flush=True)
    return txt

async def main(steps):
    if "move" in steps:
        await call("app_set_active_document", {"file_path": BOARD})
        await call("pcb_move_components", load("a_moves.json"))
        await call("obj_modify", load("a_tp.json"))
    if "copper" in steps:
        regs = load("a_regions.json")["items"]
        for k in range(0, len(regs), 10):
            await call("pcb_place_region", {"items": regs[k:k + 10]})
        tr = load("a_tracks.json")
        for k in range(0, len(tr["tracks"]), 150):
            await call("pcb_place_tracks", {"tracks": tr["tracks"][k:k + 150], "board_path": tr["board_path"]})
        await call("pcb_place_vias", load("a_vias.json"))
    if "pour" in steps:
        await call("pcb_repour_polygons", {"board_path": BOARD})
        await call("app_save_document", {"file_path": BOARD})
    if "drc" in steps:
        txt = await call("pcb_run_drc", {})
        open("c_drc.txt", "w", encoding="utf-8").write(txt)
        txt = await call("pcb_layout_audit", {"limit": 400, "expect_file": BOARD})
        open("c_audit3.txt", "w", encoding="utf-8").write(txt)

asyncio.run(main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["move", "copper", "pour", "drc"]))
