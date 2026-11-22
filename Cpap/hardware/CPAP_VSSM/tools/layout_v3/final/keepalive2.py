"""Keep the EDA Agent bridge in Altium awake (auto-off after 10 min idle).
Ping every 180 s; on a failed ping run app_recover_script (restarts only a loop that does not answer).
One line per ping in keepalive.log. Stop: delete keepalive2.run."""
import asyncio, json, logging, os, time
os.environ.setdefault("EDA_AGENT_NO_DASHBOARD", "1")
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "keepalive2.run")
LOG = os.path.join(HERE, "keepalive.log")
open(RUN, "w").close()
# keep Windows awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); ends with the process
import ctypes
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception:
    pass
from eda_agent import server  # noqa: E402

async def tool(name, args):
    res = await server.mcp.call_tool(name, args)
    if isinstance(res, tuple):
        res = res[0]
    return "\n".join(getattr(b, "text", str(b)) for b in res)

def log(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%H:%M:%S"), s))

while os.path.exists(RUN):
    try:
        t = asyncio.run(tool("app_ping", {}))
        ok = json.loads(t).get("success")
        if ok:
            log("ok")
        else:
            log("FAIL ping -> recover")
            r = asyncio.run(tool("app_recover_script", {}))
            log("recover: " + r[:300].replace("\n", " "))
    except Exception as e:
        log("EXC %r" % (e,))
    for _ in range(180):
        if not os.path.exists(RUN):
            break
        time.sleep(1)
