"""SiPAP v3 schematic edit 1: U1 GPIO remap (net labels on U1 stubs) + U6 MPN. Prints every tool reply."""
import json
from edacall import *

def show(tag, r):
    print(tag, json.dumps(r, ensure_ascii=False)[:600] if not isinstance(r, str) else r[:600])

show("focus", call("app_set_active_document", {"file_path": SCH}))
# 1) pins that lose their connection: IO6 (pin6), IO7 (7), IO17 (10), IO18 (11), IO8 (12), IO3 (15)
gone = [("SAFETY_TRIP", 12300), ("BUZZER_CTRL", 12200), ("ENC_A", 11900), ("ENC_B", 11800), ("I2C_SDA", 11700), ("BLOWER_PWM", 11400)]
items = []
for t, y in gone:
    items.append({"object_type": "eNetLabel", "filter": f"Text={t}|Location.X=10200|Location.Y={y}"})
    items.append({"object_type": "eWire", "filter": f"Location.X=11000|Location.Y={y}"})
show("delete", call("obj_delete", {"items": items, "scope": "active_doc"}))
# 2) relabel kept stubs: IO9 (pin17) I2C_SCL->SAFETY_TRIP, IO21 (pin23) ENC_SW->ENC_B, IO2 (pin38) BLOWER_FG->I2C_SDA
show("relabel", call("obj_modify", {"items": [
    {"object_type": "eNetLabel", "filter": "Text=I2C_SCL|Location.X=10200|Location.Y=11200", "set": "Text=SAFETY_TRIP"},
    {"object_type": "eNetLabel", "filter": "Text=ENC_SW|Location.X=12900|Location.Y=12700", "set": "Text=ENC_B"},
    {"object_type": "eNetLabel", "filter": "Text=BLOWER_FG|Location.X=12900|Location.Y=11200", "set": "Text=I2C_SDA"}]}))
# 3) new stubs + labels: IO10 (pin18) BLOWER_PWM, IO11 (19) BLOWER_FG, IO12 (20) BUZZER_CTRL on the left;
#    IO47 (24) ENC_A, IO48 (25) ENC_SW, IO1 (39) I2C_SCL on the right
left = [("BLOWER_PWM", 11100), ("BLOWER_FG", 11000), ("BUZZER_CTRL", 10900)]
right = [("ENC_A", 12600), ("ENC_SW", 12500), ("I2C_SCL", 11100)]
wires = [dict(x1=11000, y1=y, x2=10200, y2=y) for _, y in left] + [dict(x1=12700, y1=y, x2=12900, y2=y) for _, y in right]
show("wires", call("sch_place_wires", {"wires": wires}))
labs = [dict(text=t, x=10200, y=y, orientation=0) for t, y in left] + [dict(text=t, x=12900, y=y, orientation=0) for t, y in right]
show("labels", call("sch_place_net_label", {"items": labs}))
# 4) U6: BOM / enclosure use PEC11R-4225F-S0024 (25 mm shaft); same footprint and pinout as 4220F
show("U6", call("sch_set_components_parameters", {"stamps": [{"designator": "U6", "Comment": "PEC11R-4225F-S0024",
     "Manufacturer Part Number": "PEC11R-4225F-S0024", "Manufacturer Part Number 1": "PEC11R-4225F-S0024"}]}))
