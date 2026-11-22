"""Route with failure-driven reordering: failed nets move to the front, up to N passes."""
import json, sys, time, importlib, subprocess
plan = json.load(open("plan2.json"))
if "C27" not in plan:
    pass
best = None
for it in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4):
    out = f"../it{it}.json"
    cmd = [sys.executable, "-c", f"""
import json, route
plan = {plan!r}
fails = route.run(plan, out={out!r}, log=lambda s: open({out!r} + '.log', 'a').write(s + chr(10)))
json.dump(fails, open({out!r} + '.fails', 'w'))
"""]
    t = time.time()
    subprocess.run(cmd, check=True)
    fails = json.load(open(out + ".fails"))
    nf = sum(len(v) for v in fails.values())
    print(f"pass {it}: {nf} failed connections {dict((k, len(v)) for k, v in fails.items())} ({time.time() - t:.0f}s)", flush=True)
    if best is None or nf < best[0]:
        best = (nf, out, list(plan))
    if nf == 0:
        break
    failed = [n for n in plan if n in fails]
    plan = failed + [n for n in plan if n not in fails]
json.dump(best[2], open("plan_best.json", "w"))
print("best", best[0], best[1])
import shutil; shutil.copy(best[1], "../route_best.json")
