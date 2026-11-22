"""Print an eda-agent tool's description + input schema: python schema.py TOOL [TOOL...]"""
import sys, json, asyncio, os, logging
os.environ.setdefault("EDA_AGENT_NO_DASHBOARD", "1")
logging.disable(logging.CRITICAL)
from eda_agent import server
async def m():
    tools = {t.name: t for t in await server.mcp.list_tools()}
    for n in sys.argv[1:]:
        t = tools.get(n)
        if not t:
            print("NO TOOL", n); continue
        print("=" * 20, n); print(t.description)
        print(json.dumps(t.inputSchema.get("properties", {}), indent=1)[:6000])
asyncio.run(m())
