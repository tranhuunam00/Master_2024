import json, ripup as U, router as R
U.route = json.load(open("rip2_route.json")); R.ROUTE = U.route
# rebuild route list split (ripup module already normalised on import from route.json)
for h in ("PRESS_RED_ADC", "3V3", "SFM_5V"):
    U.rip(h)
U.repaint_all(); U.reset_trees()
old = R.astar
def big(*a, **k):
    k["maxexp"] = 8_000_000
    return old(*a, **k)
R.astar = big
print("SCL", R.route_net("I2C_SCL"))
for h in ("SFM_5V", "PRESS_RED_ADC", "3V3"):
    print(h, R.route_net(h))
U.repaint_all(); U.reset_trees()
print("FINAL unconnected", {k: [q['comp'] + '.' + q['name'] for q in v] for k, v in U.unconnected().items()})
json.dump(U.route, open("route.json", "w"))
