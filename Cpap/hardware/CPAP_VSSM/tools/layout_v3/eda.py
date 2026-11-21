"""Call an eda-agent tool by name: python eda.py TOOL '{json args}' [outfile]"""
import sys, json, asyncio, os, logging
os.environ.setdefault("EDA_AGENT_NO_DASHBOARD", "1")
logging.disable(logging.CRITICAL)
from eda_agent import server

async def run(name, args):
    res = await server.mcp.call_tool(name, args)
    if isinstance(res, tuple):
        res = res[0]
    out = []
    for b in res:
        out.append(getattr(b, "text", str(b)))
    return "\n".join(out)

name = sys.argv[1]
PRJ = r"C:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\hardware\CPAP_VSSM"
a = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "{}"
if a.startswith("@"):
    a = open(a[1:], encoding="utf-8").read()
a = a.replace("%PRJ%", PRJ.replace(chr(92), chr(92)*2))
args = json.loads(a)
txt = asyncio.run(run(name, args))
if len(sys.argv) > 3:
    open(sys.argv[3], "w", encoding="utf-8").write(txt)
    print(f"[{len(txt)} chars -> {sys.argv[3]}]")
    print(txt[:1500])
else:
    print(txt)
