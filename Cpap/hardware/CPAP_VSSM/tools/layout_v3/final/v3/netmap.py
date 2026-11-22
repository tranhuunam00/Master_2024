"""Schematic changes for v3 (applied to the model; written to the SchDoc later).
U1 GPIO remap: each peripheral uses the module side that faces it. J3 pin order follows the U1 pin order."""
U1_REMAP = {            # module pin -> net ('' = no connection)
    "6": "",            # IO6   was SAFETY_TRIP
    "7": "",            # IO7   was BUZZER_CTRL
    "10": "",           # IO17  was ENC_A
    "11": "",           # IO18  was ENC_B
    "12": "",           # IO8   was I2C_SDA
    "15": "",           # IO3   was BLOWER_PWM (strapping pin, now free)
    "17": "SAFETY_TRIP",  # IO9  was I2C_SCL
    "18": "BLOWER_PWM",   # IO10
    "19": "BLOWER_FG",    # IO11
    "20": "BUZZER_CTRL",  # IO12
    "23": "ENC_B",        # IO21 was ENC_SW
    "24": "ENC_A",        # IO47
    "25": "ENC_SW",       # IO48
    "38": "I2C_SDA",      # IO2  was BLOWER_FG
    "39": "I2C_SCL",      # IO1
}
U1_GPIO = {"3": "EN", "4": 4, "5": 5, "6": 6, "7": 7, "8": 15, "9": 16, "10": 17, "11": 18, "12": 8, "13": 19,
           "14": 20, "15": 3, "16": 46, "17": 9, "18": 10, "19": 11, "20": 12, "21": 13, "22": 14, "23": 21,
           "24": 47, "25": 48, "26": 45, "27": 0, "28": 35, "29": 36, "30": 37, "31": 38, "32": 39, "33": 40,
           "34": 41, "35": 42, "36": "RXD0", "37": "TXD0", "38": 2, "39": 1}
# J3, pin 1 at the bottom: 3V3, EN, GND, IO0, RXD, TXD (old: GND, 3V3, TXD, RXD, EN, IO0)
J3_REMAP = {}   # pin order unchanged (GND, 3V3, TXD, RXD, EN, IO0); header turned 180 instead

def apply(pads_local):
    for p in pads_local.get("U1", []):
        if p["name"] in U1_REMAP:
            p["net"] = U1_REMAP[p["name"]]
    for p in pads_local.get("J3", []):
        p["net"] = J3_REMAP.get(p["name"], p["net"])
