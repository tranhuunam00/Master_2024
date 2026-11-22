"""Import-able eda-agent caller: from edacall import call, PRJ, BOARD; call('tool', {...}) -> parsed JSON (or text)."""
import json, asyncio, os, logging
os.environ.setdefault("EDA_AGENT_NO_DASHBOARD", "1")
logging.disable(logging.CRITICAL)
from eda_agent import server
import eda_agent.bridge.altium_bridge as _ab
_ab._SESSION_TAG = os.environ.get('EDA_TAG', 'd144cf29')   # one tag for all my processes (doc ownership)

PRJ = r"C:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\hardware\CPAP_VSSM"
BOARD = PRJ + r"\SiPAP_PCB.PcbDoc"
SCH = PRJ + r"\SiPAP_Main_RevC.SchDoc"
PRJPCB = PRJ + r"\CPAP_VSSM.PrjPcb"

async def _run(name, args):
    res = await server.mcp.call_tool(name, args)
    if isinstance(res, tuple):
        res = res[0]
    return "\n".join(getattr(b, "text", str(b)) for b in res)

def call(name, args=None, save=None, quiet=True):
    txt = asyncio.run(_run(name, args or {}))
    if save:
        open(save, "w", encoding="utf-8").write(txt)
    try:
        d = json.loads(txt)
        if isinstance(d, dict):
            d.pop("_system_reminder", None)
        return d
    except Exception:
        return txt
