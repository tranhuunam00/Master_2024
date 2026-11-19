"""Rev C PCB placements for the new/changed parts (pad nets from the Rev C netlist)."""
import json, os
LIB = r"C:\Users\ADMIN\OneDrive\Máy tính\Master_2024\Cpap\hardware\CPAP_VSSM\SiPAP_Footprints.PcbLib"
UID = {"J1":"LARRENTU","F1":"AXBSQBDD","D1":"WKUUVTBI","D2":"MPOVDULB","C23":"GEQHNXMH","Q1":"SIJKHFGH",
       "U8":"UHZIGGNU","C13":"JCWSNYIZ","C15":"NHFRQBIB","C17":"AWVJUHTA","L1":"EXJPZLTZ","C18":"RONSZRXQ",
       "R19":"QZVWRDXT","C19":"FGONDDBB","R20":"RHNZYECN","U9":"WISSGBUX","C14":"ZBUGWAGN","C16":"DSXPYTQJ",
       "C20":"ROFATXIB","L2":"QHVUEGTG","C21":"GFAPOTZX","R21":"WLDSHLHM","C22":"HIACGPIE","R22":"UFNIUQPH",
       "BZ1":"KQPTAQNN"}
COMMENT = {"J1":"PJ-102AH 12V IN","F1":"MF-R300 3A PTC","D1":"SS54 Schottky 5A","D2":"SMBJ15A TVS","C23":"22uF 25V",
           "Q1":"AO4407A","U8":"TPS54202 (5V)","U9":"TPS54202 (3V3)","C13":"10uF 35V","C14":"10uF 35V",
           "C15":"100nF","C16":"100nF","C17":"100nF","C20":"100nF","L1":"15uH SRN6045TA-150M","L2":"10uH SRN6045TA-100M",
           "C18":"22uF 16V","C21":"22uF 16V","R19":"100k 1%","R21":"100k 1%","C19":"75pF C0G","C22":"56pF C0G",
           "R20":"13.3k 1%","R22":"22.1k 1%","BZ1":"PS1240P02BT"}
FP = {"J1":"CUI_PJ-102AH","F1":"PTC_RADIAL_MF-R300","D1":"DIOM_SMC_DO-214AB","D2":"DIOM_SMB_DO-214AA_BOURNS",
      "Q1":"SOIC-8","U8":"TI_DDC0006A_SOT23-6","U9":"TI_DDC0006A_SOT23-6","L1":"IND_BOURNS_SRN6045TA",
      "L2":"IND_BOURNS_SRN6045TA","BZ1":"BUZZER_TDK_PS1240"}
P = []
def add(d, x, y, rot, nets):
    P.append({"footprint": FP.get(d, "R_0805"), "library_path": LIB, "designator": d, "x": x, "y": y,
              "rotation": rot, "layer": "TopLayer", "comment": COMMENT[d], "unique_id": UID[d], "pad_nets": nets})
def buck(u, cin, cbyp, cboot, l, co, rtop, cff, rbot, ux, uy, sw, bst, fb, vout):
    add(u, ux, uy, 0, {"1": "GND", "2": sw, "3": "VIN12V", "4": fb, "6": bst})
    add(cbyp, ux-190, uy, 90, {"1": "VIN12V", "2": "GND"})
    add(cin, ux-300, uy, 90, {"1": "VIN12V", "2": "GND"})
    add(l, ux-110, uy+260, 180, {"1": sw, "2": vout})
    add(cboot, ux+130, uy+110, 90, {"1": bst, "2": sw})
    add(co, ux+40, uy+470, 0, {"1": vout, "2": "GND"})
    add(rtop, ux+170, uy-120, 90, {"1": vout, "2": fb})
    add(cff, ux+260, uy-120, 90, {"1": vout, "2": fb})
    add(rbot, ux+350, uy-120, 270, {"1": fb, "2": "GND"})
buck("U8", "C13", "C15", "C17", "L1", "C18", "R19", "C19", "R20", 400, 1700, "SW5", "BST5", "FB5", "5V")
buck("U9", "C14", "C16", "C20", "L2", "C21", "R21", "C22", "R22", 1150, 1700, "SW3", "BST3", "FB3", "3V3")
add("J1", 0, 2480, 0, {"1": "VIN_RAW", "2": "GND", "3": "GND"})
add("F1", 850, 2700, 0, {"1": "VIN_RAW", "2": "VIN_FUSED"})
add("D1", 950, 2420, 0, {"1": "VIN_FUSED", "2": "VIN12V"})
add("D2", 1300, 2420, 0, {"1": "VIN12V", "2": "GND"})
add("C23", 1300, 2560, 0, {"1": "VIN12V", "2": "GND"})
add("Q1", 700, 720, 0, {"1": "VIN12V", "2": "VIN12V", "3": "VIN12V", "4": "GATE",
                        "5": "BLOWER_V12", "6": "BLOWER_V12", "7": "BLOWER_V12", "8": "BLOWER_V12"})
add("BZ1", 1720, 280, 0, {"1": "BUZZER_CTRL", "2": "GND"})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "place_revc.json")
json.dump(P, open(out, "w"), indent=1)
print(len(P), "placements ->", out)
