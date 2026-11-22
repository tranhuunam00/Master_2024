"""Electrical rule check of the SiPAP Rev C external wiring harness.

Models every cable wire: board connector pin -> net (from the schematic template)
-> external device pin (from its datasheet), then checks
  * the board pin really is on the net the harness table claims,
  * pin types are compatible (power/ground/I2C/UART/PWM/tacho),
  * supply voltage is inside the device's datasheet range,
  * logic levels: device input thresholds vs the 3.3 V board side, pull-up rails,
  * TX goes to RX, no two outputs on one wire, every device pin is accounted for.
Unknowns (no datasheet yet, e.g. the blower) are reported as OPEN, not passed.

Usage: python check_harness.py [template.json]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
tpl = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "sipap_template_revc.json"), encoding="utf-8"))
net_of = {p: n for n, pins in tpl["nets"].items() for p in pins}

# ---- board-side nets: kind + voltage (from verification_RevC.md section 3) ----
NET = {
    "GND":        ("gnd",   0.0),
    # power nets carry the worst-case band (R 1% + VFB 1%), verification_RevC.md sec 3
    "3V3":        ("power", (3.21, 3.38)),   # TPS54202 U9
    "SFM_5V":     ("power", (4.94, 5.22)),   # U8 through Q2 (AO3401A, ~50 mOhm x 5 mA -> <1 mV)
    "BLOWER_V12": ("power", (11.4, 12.6)),   # adapter 12 V +-5 %, F1/Q4 drop < 0.1 V at 1.6 A
    "I2C_SCL":    ("i2c",   3.21),           # 4.7k pull-ups R5/R6 to 3V3 (worst-case low)
    "I2C_SDA":    ("i2c",   3.21),
    "UART_TX":    ("out",   3.293),   # ESP32 TXD0
    "UART_RX":    ("in",    3.293),   # ESP32 RXD0
    "EN":         ("in",    3.293),   # 10k pull-up R9 + 1 uF
    "BOOT":       ("in",    3.293),   # IO0, internal pull-up
    "BLOWER_VSR":   ("out",   3.293),   # ESP32 IO10 PWM through R28 1k + C26 1uF -> 0-3.3 V DC (layout v3 pin map)
    "BLOWER_FG_IN": ("in_pu", 3.293),   # R29 1k series -> IO11 with R12 10k pull-up to 3V3 (layout v3 pin map)
}

# ---- external devices (datasheet pin tables) ----
# kind: gnd / power(min,max) / i2c(vih_min) / in / out / od (open-drain out) / nc / unknown
DEV = {
    "SFM3300-D": {   # Sensirion SFM3300 datasheet v1.4, sec 2.1 table + 2.2 pad layout
        "src": "Sensirion SFM3300 DS v1.4 p.3-4",
        "pins": {"HEAT_GND": ("gnd",), "VDD": ("power", 4.75, 5.25), "SCK": ("i2c", 2.5),
                 "GND": ("gnd",), "DATA": ("i2c", 2.5), "HEAT": ("nc",)},
    },
    "OLED DSP-8": {  # ProtoSupplies DSP-8 SSD1306 0.96" (VCC GND SCL SDA, on-board 3.3 V regulator)
        "src": "protosupplies.com DSP-8: 'accepts 3.3 V or 5 V' -> modelled as 3.3 V -5% .. 5 V +5%",
        "pins": {"VCC": ("power", 3.135, 5.25), "GND": ("gnd",), "SCL": ("i2c", 0.7 * 3.3), "SDA": ("i2c", 0.7 * 3.3)},
    },
    "USB-UART 3V3": {  # any CP2102/CH340 adapter set to 3.3 V
        "src": "generic 3.3 V adapter",
        "pins": {"GND": ("gnd",), "3V3": ("nc",), "RXD": ("in",), "TXD": ("out",), "RTS": ("out",), "DTR": ("out",)},
    },
    "Blower": {      # Wonsmart WS4540-12-NZ03 drawing rev 2023-07-12 (built-in driver, 4 wires)
        "src": "Wonsmart WS4540-12-NZ03 drawing: pin1 yellow VSR 0-3.3V, pin2 white FG, pin3 black GND, pin4 red +12V",
        # 12 VDC nominal; the drawing gives no range -> modelled as 12 V +-10 %
        "pins": {"V+": ("power", 10.8, 13.2), "GND": ("gnd",), "PWM": ("in",),
                 "FG": ("unknown", "FG output stage not specified (drawing: '1 Hz x 60 = speed'); "
                                   "board pulls up to 3V3 via R12 - measure that FG never exceeds 3.3 V")},
    },
}

# ---- harness: (board pin, device, device pin) — one entry per wire, as in wiring_harness.md ----
HARNESS = {
    "J2 -> blower": [("J2.1", "Blower", "V+"), ("J2.2", "Blower", "GND"), ("J2.3", "Blower", "PWM"), ("J2.4", "Blower", "FG")],
    "U4 -> SFM3300": [("U4.1", "SFM3300-D", "HEAT_GND"), ("U4.2", "SFM3300-D", "VDD"), ("U4.3", "SFM3300-D", "SCK"),
                      ("U4.4", "SFM3300-D", "GND"), ("U4.5", "SFM3300-D", "DATA"), (None, "SFM3300-D", "HEAT")],
    # module pin order is VCC GND SCL SDA, U5 is GND VCC SCL SDA -> wires 1 and 2 cross
    "U5 -> OLED": [("U5.1", "OLED DSP-8", "GND"), ("U5.2", "OLED DSP-8", "VCC"), ("U5.3", "OLED DSP-8", "SCL"),
                   ("U5.4", "OLED DSP-8", "SDA")],
    "J3 -> USB-UART": [("J3.1", "USB-UART 3V3", "GND"), (None, "USB-UART 3V3", "3V3"), ("J3.3", "USB-UART 3V3", "RXD"),
                       ("J3.4", "USB-UART 3V3", "TXD"), ("J3.5", "USB-UART 3V3", "RTS"), ("J3.6", "USB-UART 3V3", "DTR")],
}
# expected net per board pin, as written in wiring_harness.md
DOC = {"J2.1": "BLOWER_V12", "J2.2": "GND", "J2.3": "BLOWER_VSR", "J2.4": "BLOWER_FG_IN",
       "U4.1": "GND", "U4.2": "SFM_5V", "U4.3": "I2C_SCL", "U4.4": "GND", "U4.5": "I2C_SDA",
       "U5.1": "GND", "U5.2": "3V3", "U5.3": "I2C_SCL", "U5.4": "I2C_SDA",
       "J3.1": "GND", "J3.2": "3V3", "J3.3": "UART_TX", "J3.4": "UART_RX", "J3.5": "EN", "J3.6": "BOOT"}

fails = opens = passes = 0


def report(status, cable, wire, msg):
    global fails, opens, passes
    if status == "FAIL": fails += 1
    elif status == "OPEN": opens += 1
    else: passes += 1
    print(f"  [{status}] {cable:16s} {wire:24s} {msg}")


def check_wire(cable, bp, dev, dp):
    spec = DEV[dev]["pins"][dp]
    kind = spec[0]
    wire = f"{bp or '(none)'} -> {dev}.{dp}"
    if bp is None:
        if kind in ("nc",):
            report("PASS", cable, wire, "left unconnected, as required")
        else:
            report("FAIL", cable, wire, f"device pin of kind '{kind}' left unconnected")
        return
    net = net_of.get(bp, "")
    if net != DOC.get(bp):
        report("FAIL", cable, wire, f"board pin is on net {net!r}, harness table says {DOC.get(bp)!r}")
        return
    nk, nv = NET[net]
    if kind == "gnd":
        ok = nk == "gnd"
        report("PASS" if ok else "FAIL", cable, wire, f"{net} -> ground" if ok else f"{net} ({nk}) on a ground pin")
    elif kind == "power":
        lo, hi = spec[1], spec[2]
        if nk != "power":
            report("FAIL", cable, wire, f"{net} ({nk}) on a supply pin")
        elif lo is None:
            report("OPEN", cable, wire, f"{net} {nv[0]:.1f}-{nv[1]:.1f} V - device supply range unknown (no datasheet)")
        else:
            ok = lo <= nv[0] and nv[1] <= hi
            report("PASS" if ok else "FAIL", cable, wire, f"{net} {nv[0]:.2f}-{nv[1]:.2f} V worst case vs device {lo}-{hi} V")
    elif kind == "i2c":
        if nk != "i2c":
            report("FAIL", cable, wire, f"{net} ({nk}) on an I2C pin")
        else:
            ok = nv >= spec[1]
            report("PASS" if ok else "FAIL", cable, wire, f"{net} pulled to {nv:.2f} V, device VIH min {spec[1]:.2f} V")
    elif kind == "in":
        ok = nk == "out"
        report("PASS" if ok else "FAIL", cable, wire, f"board {nk} -> device input" if ok else f"device input fed by board {nk} ({net})")
    elif kind == "out":
        ok = nk in ("in", "in_pu")
        report("PASS" if ok else "FAIL", cable, wire, f"device output -> board input ({net})" if ok else f"device output into board {nk} ({net}): contention")
    elif kind == "nc":
        report("FAIL" if nk in ("power",) else "PASS", cable, wire, f"device pin must stay open; wired to {net}")
    elif kind == "unknown":
        report("OPEN", cable, wire, spec[1])


print("SiPAP Rev C harness check\n")
for cable, wires in HARNESS.items():
    seen = set()
    for bp, dev, dp in wires:
        check_wire(cable, bp, dev, dp)
        seen.add((dev, dp))
    devs = {d for _, d, _ in wires}
    for d in devs:
        for p in DEV[d]["pins"]:
            if (d, p) not in seen:
                report("FAIL", cable, f"{d}.{p}", "device pin missing from harness table")
print(f"\n{passes} PASS, {fails} FAIL, {opens} OPEN (need data that does not exist yet)")
sys.exit(1 if fails else 0)
