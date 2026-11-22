"""Assemble the SiPAP Rev C layout-v3 fabrication package in Fab_RevC_v3/.

Inputs (made in Altium through the eda-agent bridge):
  - Project Outputs for CPAP_VSSM/v3/Fabrication/  (OutJob SiPAP_Fab.OutJob: Gerber RS-274X mm 4:4,
    Excellon mm 4:3 with G85 slots, IPC-D-356 test-point netlist, ODB++)
  - Project Outputs for CPAP_VSSM/v3docs/Documentation/Schematic Print/Schematic Prints.PDF
  - board dump JSON (pads, vias, components) from obj_query
  - Fab_RevC_v2/BOM_RevC_v2.csv as the base of the BOM (lines edited below)
Usage: python fab_v3.py <board_dump.json>
2026-11-22 review fixes: R10 47k -> 36k plus trimmer RV1 (Bourns 3224W-1-503E) in the COMP_REF top leg, so the
LM393 trip can be set to 25 cmH2O whatever U3's offset (MPX5010 DS p.2: 0-0.425 V); R30 100k pulls BLOWER_VSR
low while the ESP32 is in reset. Board dump: tools/layout_v3/board_dump_v3_rv1.json.
"""
import csv, json, os, shutil, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(PRJ, 'Project Outputs for CPAP_VSSM', 'v3', 'Fabrication')
DOCS = os.path.join(PRJ, 'Project Outputs for CPAP_VSSM', 'v3docs', 'Documentation')
OUT = os.path.join(PRJ, 'Fab_RevC_v3')
NAME = 'SiPAP_RevC_v3'
os.makedirs(OUT, exist_ok=True)
GER = os.path.join(OUT, 'gerber')
os.makedirs(GER, exist_ok=True)

# ---------------- Gerber + drill: copy with a real name, then zip ----------------
# Only when fresh OutJob output exists (the raw folders are git-ignored and may have been cleaned);
# otherwise the package already in Fab_RevC_v3 is kept as it is.
if os.path.isdir(os.path.join(SRC, 'Gerber')):
    fab_files = []
    for ext in ('GTL', 'GBL', 'GTO', 'GBO', 'GTS', 'GBS', 'GTP', 'GBP', 'GM'):
        s = os.path.join(SRC, 'Gerber', 'PCB_PART_NUMBER.' + ext)
        d = os.path.join(GER, f'{NAME}.{ext}')
        shutil.copyfile(s, d); fab_files.append(d)
    for s_name, d_name in (('PCB_PART_NUMBER-RoundHoles.TXT', f'{NAME}-RoundHoles.TXT'),
                           ('PCB_PART_NUMBER-SlotHoles.TXT', f'{NAME}-SlotHoles.TXT'),
                           ('PCB_PART_NUMBER.DRR', f'{NAME}.DRR')):
        d = os.path.join(GER, d_name)
        shutil.copyfile(os.path.join(SRC, 'NC Drill', s_name), d); fab_files.append(d)
    with zipfile.ZipFile(os.path.join(OUT, f'{NAME}_Gerber.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for f in fab_files:
            z.write(f, os.path.basename(f))
    shutil.copyfile(os.path.join(SRC, 'Test Points', 'PCB_PART_NUMBER.ipc'), os.path.join(OUT, f'{NAME}.ipc'))
    shutil.copyfile(os.path.join(SRC, 'ODB', 'PCB_PART_NUMBER.zip'), os.path.join(OUT, f'{NAME}_ODB++.zip'))
    shutil.copyfile(os.path.join(DOCS, 'Schematic Print', 'Schematic Prints.PDF'), os.path.join(OUT, f'{NAME}_Schematic.pdf'))
else:
    print('no raw OutJob output: Gerber/drill/IPC/ODB/PDF in Fab_RevC_v3 kept as they are')

# ---------------- Pick and place (mm, origin = board lower-left, top side) ----------------
dump = json.load(open(sys.argv[1], encoding='utf-8'))
comps = dump['comps']
THT = {'J1', 'J2', 'J3', 'U4', 'U5', 'U6', 'F1', 'BZ1', 'C25'}


def dkey(d):
    return (''.join(c for c in d if c.isalpha()), int(''.join(c for c in d if c.isdigit()) or 0))


def expand(s):
    out = []
    for tok in s.split():
        if '-' in tok:           # MH1-MH4 style
            a, b = tok.split('-'); p = ''.join(c for c in a if c.isalpha())
            out += [f'{p}{i}' for i in range(int(a[len(p):]), int(b[len(p):]) + 1)]
        else:
            out.append(tok)
    return out


with open(os.path.join(OUT, f'{NAME}_CPL.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['Designator', 'Comment', 'Footprint', 'Mid X', 'Mid Y', 'Rotation', 'Layer', 'Mount'])
    for c in sorted(comps, key=lambda c: dkey(c['Designator'])):
        d = c['Designator']
        w.writerow([d, c['Comment'], c['Pattern'], f"{float(c['X']) * 0.0254:.3f}mm", f"{float(c['Y']) * 0.0254:.3f}mm",
                    int(float(c['Rotation'])) % 360, 'Top' if 'Top' in c['Layer'] else 'Bottom',
                    'THT (hand solder, not machine placed)' if d in THT else 'SMD'])

# ---------------- BOM: v2 lines + the v3 changes, checked against the board ----------------
rows = list(csv.DictReader(open(os.path.join(PRJ, 'Fab_RevC_v2', 'BOM_RevC_v2.csv'), encoding='utf-8')))
for r in rows:
    des = r['Designators'].split()
    if 'C1' in des and r['Value'] == '22uF 16V':
        r['Designators'] = 'C1 C2 C18 C21 C27'; r['Qty'] = '5'
        r['Notes'] = ('2 per buck output (C2 C18 = 5V, C1 C21 = 3V3); C27 = 22 uF bulk at U1 3V3 pin '
                      '(Espressif ESP32-S3-WROOM-1 DS v1.8 Fig. 9-1 p.41: 22 uF + 0.1 uF on 3V3)')
    if des == ['BZ1']:
        r['Notes'] = 'Driven from IO12 (layout v3 GPIO map)'
    if des == ['J3']:
        r['Notes'] = ('Cross TX/RX on USB-UART adapter; layout v3: vertical at the right edge, pin 1 (GND) at the top, '
                      'pin names printed on silkscreen')
    if des == ['U1']:
        r['Notes'] = ('Antenna at board edge; keepout under antenna; EPAD vias tented; layout v3 GPIO map '
                      '(see firmware/sipap_v1/config.h); IO47/IO48 are 3.3 V on N8R8 (1.8 V only on N16R16V, DS v1.8 p.41)')
    if des == ['D2']:
        r['Alternate MPN'] = 'SMBJ13A (Littelfuse); SMBJ13CA (bidirectional, same clamp row in Bourns SMBJ DS p.2)'
    if des == ['U6']:
        r['Notes'] = ('31.5 mm tall above PCB: 6.5 mm body + L = 25 mm shaft (Bourns PEC11R DS p.2-3 shaft table); '
                      'STEP model updated to this height')
    if des == ['J1']:
        r['Description'] = 'DC barrel jack THT, 2.0 mm centre pin, 24 V / 5.0 A (Same Sky PJ-102AH DS p.1-2)'
    if des == ['R19', 'R21']:   # R19 (5V divider) goes 0.1 %, R21 (3V3) stays 1 % on its own line 32c
        r['Qty'] = '1'; r['Designators'] = 'R19'
        r['Description'] = 'Resistor 0.1% 1/8W thin film 25 ppm/C (5V FB top)'
        r['Manufacturer'] = 'Yageo'; r['MPN'] = 'RT0805BRD07100KL'
        r['Notes'] = ('0.1 % keeps the 5V rail inside the 4.75-5.25 V allowed for MPX5010 (DS p.2) and SFM3300 (DS p.3): '
                      'VFB 0.581-0.611 V (TI SLVSD26C p.5) gives 4.93-5.23 V with 0.1 % and 25 ppm/C over 60 C; 1 % '
                      'thick film could reach 5.35 V. Yageo RT DS V.16: B = 0.1 %, D = 25 ppm/C (p.2), RT0805 1/8 W, '
                      '150 V, 1 R-1.5 M at 0.1 %/25 ppm (p.5)')
    if des == ['R20']:
        r['Description'] = 'Resistor 0.1% 1/8W thin film 25 ppm/C (5V FB bottom)'
        r['Manufacturer'] = 'Yageo'; r['MPN'] = 'RT0805BRD0713K3L'
        r['Notes'] = ('Vout = 0.596*(1+100k/13.3k) = 5.08V typ; 0.1 % with R19, see R19 note '
                      '(Yageo RT DS V.16 p.2 and p.5)')
    if des == ['R10']:
        r['Value'] = '36k'; r['MPN'] = 'RC0805FR-0736KL'
        r['Notes'] = ('COMP_REF top leg in series with RV1 (5V - RV1 - R10 - COMP_REF, R11 10k to GND): COMP_REF '
                      '0.52-1.09 V over RV1 travel at Vs = 5 V, covering MPX5010 DS p.2 offset 0-0.425 V and span '
                      '4.275-4.725 V (needs 0.69-1.05 V for a 25 cmH2O trip)')
new_rows = {
    '29': {'Item': '29a', 'Qty': '1', 'Designators': 'RV1', 'Value': '50k',
           'Description': 'Trimmer 50k +-10 %, 12-turn cermet, top adjust, SMD 4 mm (Bourns 3224 DS p.1)',
           'Footprint': 'BOURNS_3224W', 'Manufacturer': 'Bourns', 'MPN': '3224W-1-503E', 'Alternate MPN': '',
           'LCSC #': '', 'Digi-Key #': '', 'Mount': 'SMD',
           'Notes': ('Sets the LM393 hardware trip: adjust at test to 25 cmH2O (firmware self-test step 9). '
                     'Pin 1 (CCW end) on 5V, wiper 2 tied to pin 3 (CW end): clockwise = more resistance = lower trip '
                     '(Bourns 3224 DS p.1 schematic). An open element or lifted wiper raises the resistance, so '
                     'COMP_REF falls and the cutoff trips (fail-safe)')},
    '32': {'Item': '32c', 'Qty': '1', 'Designators': 'R21', 'Value': '100k',
           'Description': 'Resistor 1% 1/8W (3V3 FB top)', 'Footprint': 'R_0805', 'Manufacturer': 'Yageo',
           'MPN': 'RC0805FR-07100KL', 'Alternate MPN': '', 'LCSC #': '', 'Digi-Key #': '', 'Mount': 'SMD',
           'Notes': ('3V3 rail with 1 %: 0.581-0.611 V x (1 + 100k/22.1k) incl. 1 % gives 3.16-3.43 V (3.13-3.47 V '
                     'even with 100 ppm/C over 60 C), inside ESP32-S3-WROOM-1 VDD33 3.0-3.6 V (DS v1.8 p.27)')},
    '32a': {'Item': '32b', 'Qty': '1', 'Designators': 'R30', 'Value': '100k',
            'Description': 'Resistor 1% 1/8W (BLOWER_VSR pull-down)', 'Footprint': 'R_0805', 'Manufacturer': 'Yageo',
            'MPN': 'RC0805FR-07100KL', 'Alternate MPN': '', 'LCSC #': '', 'Digi-Key #': '', 'Mount': 'SMD',
            'Notes': ('Holds the blower speed input at 0 V while IO10 is high-Z (reset, boot, flashing via J3); '
                      'with R28 1k it lowers the VSR level by 1 %')},
}
out_rows = []
for r in rows:
    out_rows.append(r)
    if r['Item'] in new_rows:
        out_rows.append(new_rows[r['Item']])
rows = out_rows
# LCSC / JLC parts, looked up 2026-11-22 with the lcsc skill (jlcsearch). (code, JLC library, part MPN if
# it differs from the BOM's MPN). Passives use JLC *basic* parts of the same value, tolerance, voltage
# and dielectric, as the BOM's substitution rule allows. '' = not found in the JLC library index.
LCSC = {
    '1': ('C2913201', 'extended', ''),
    '2': ('C3238865', 'extended', 'MPXV5010GC6T1 = same sensor on tape and reel (MPX5010 DS p.5 ordering table, '
                                  'Motorola/Freescale, now NXP: Gauge, Axial Port, SMT, case 482A); LCSC stock was 3 on '
                                  '2026-11-22 for 2 needed'),
    '3': ('C191884', 'extended', ''),
    '4': ('C7955', 'basic', 'LM393DR2G (onsemi), same SOIC-8 pinout (onsemi LM393 DS p.1: 1 output A, 4 GND, 8 VCC)'),
    '5': ('C16072', 'extended', 'AOS'),
    '6': ('C15127', 'basic', 'AOS'), '6a': ('C8545', 'basic', '2N7002 (JSCJ; JSCJ DS p.1: 1 gate, 2 source, 3 drain)'),
    '7': ('C1973187', 'extended', 'SMBJ13CA-E3/52 (Vishay), bidirectional alternate: Vishay SMBJ DS p.2 SMBJ13A/CA row '
                                  'VWM 13 V, VBR 14.4-15.9 V, VC 21.5 V @ 27.9 A (same values as Bourns SMBJ DS p.2); '
                                  'D2 is on VIN12V after Q4, so it never sees a reversed supply and either orientation is fine'),
    '8': ('C208488', 'extended', ''), '9': ('', '', ''), '10': ('', '', ''), '11': ('C3096093', 'extended', ''),
    '17': ('C76871', 'extended', ''), '18': ('C440198', 'basic', 'GRM21BR61H106KE43L 10uF 50V X5R'),
    '18a': ('C45783', 'basic', 'CL21A226MAQNNNE 22uF 25V X5R'), '18b': ('C178666', 'extended', ''),
    '19': ('C45783', 'basic', 'CL21A226MAQNNNE 22uF 25V X5R'), '20': ('C49678', 'basic', 'CC0805KRX7R9BB104 100nF 50V X7R'),
    '21': ('C28323', 'basic', 'CL21B105KBFNNNE 1uF 50V X7R'), '22': ('C1710', 'basic', ''),
    '23': ('C277525', 'extended', 'CC0805JRNPOYBN471 470pF 250V NP0'), '24': ('C1838', 'extended', '0805CG750J500NT 75pF 50V C0G'),
    '25': ('C1828', 'extended', '0805CG560J500NT 56pF 50V C0G'), '26': ('C17414', 'basic', '0805W8F1002T5E'),
    '27': ('C4328', 'basic', '0805W8F2002T5E'), '28': ('C17673', 'basic', '0805W8F4701T5E'),
    '29': ('C4360', 'extended', '0805W8F3602T5E 36k 1%'), '29a': ('C48484', 'extended', 'Bourns 3224W-1-503E'),
    '30': ('C17514', 'basic', '0805W8F1004T5E'), '32b': ('C149504', 'basic', '0805W8F1003T5E'),
    '31': ('C441993', 'extended', 'ERJP06F3303V'),
    '32': ('C122537', 'extended', 'Yageo RT0805BRD07100KL 100k 0.1% 25ppm'), '32c': ('C149504', 'basic', '0805W8F1003T5E'),
    '32a': ('C149504', 'basic', '0805W8F1003T5E'), '33': ('C865182', 'extended', 'Yageo RT0805BRD0713K3L 13.3k 0.1% 25ppm'),
    '34': ('C17552', 'extended', '0805W8F2212T5E'), '34a': ('C17513', 'basic', '0805W8F1001T5E'),
}
NOT_IN_JLC = {'9': 'not in the JLC library index (2026-11-22): let JLC match the MPN at order time, or hand-solder. '
                    'A substitute must be 15 uH +-20 %, Isat >= 3.8 A, Irms >= 2.8 A, DCR <= 71 mOhm, 6x6 mm '
                    '(SRN6045TA DS p.1)',
              '10': 'not in the JLC library index (2026-11-22): let JLC match the MPN at order time, or hand-solder. '
                    'A substitute must be 10 uH +-20 %, Isat >= 4.6 A, Irms >= 3.2 A, DCR <= 52 mOhm, 6x6 mm '
                    '(SRN6045TA DS p.1)'}
JLC_MPN = {'7': 'SMBJ13CA-E3/52 (alternate for SMBJ13A)', '2': 'MPXV5010GC6T1 (tape and reel of MPXV5010GC6U)'}
JLC_PKG = {'1': 'Module 25.5x18mm (ESP32-S3-WROOM-1)', '2': 'SOP-8 (NXP case 482A)', '3': 'SOT-23-6', '4': 'SOIC-8',
           '5': 'SOIC-8', '6': 'SOT-23', '6a': 'SOT-23', '7': 'SMB (DO-214AA)', '9': 'SMD 6x6mm', '10': 'SMD 6x6mm',
           '29a': 'SMD-3P 4.8x3.5mm (Bourns 3224W)'}
for r in rows:
    code, lib, part = LCSC.get(r['Item'], ('', '', ''))
    if code:
        r['LCSC #'] = code
        extra = f'LCSC {code} ({lib}{", " + part if part else ""})'
        r['Notes'] = (r['Notes'] + '; ' if r['Notes'] else '') + extra
    elif r['Item'] in NOT_IN_JLC:
        r['Notes'] = (r['Notes'] + '; ' if r['Notes'] else '') + NOT_IN_JLC[r['Item']]
# The MPN column names the part JLC actually fits (its datasheet is in Cpap/tools/datasheets); the design-time
# MPN moves to Alternate MPN. Every line gets the datasheet file(s) it was checked against, and the run stops
# if one is missing.
FITTED = {
    '2': ('NXP', 'MPXV5010GC6T1'), '4': ('onsemi', 'LM393DR2G'), '6a': ('JSCJ', '2N7002'),
    '7': ('Vishay', 'SMBJ13CA-E3/52'), '18': ('Murata', 'GRM21BR61H106KE43L'),
    '18a': ('Samsung', 'CL21A226MAQNNNE'), '19': ('Samsung', 'CL21A226MAQNNNE'),
    '20': ('Yageo', 'CC0805KRX7R9BB104'), '21': ('Samsung', 'CL21B105KBFNNNE'), '22': ('Samsung', 'CL21B103KBANNNC'),
    '23': ('Yageo', 'CC0805JRNPOYBN471'), '24': ('FH', '0805CG750J500NT'), '25': ('FH', '0805CG560J500NT'),
    '26': ('UNI-ROYAL', '0805W8F1002T5E'), '27': ('UNI-ROYAL', '0805W8F2002T5E'), '28': ('UNI-ROYAL', '0805W8F4701T5E'),
    '29': ('UNI-ROYAL', '0805W8F3602T5E'), '30': ('UNI-ROYAL', '0805W8F1004T5E'), '31': ('Panasonic', 'ERJP06F3303V'),
    '32a': ('UNI-ROYAL', '0805W8F1003T5E'), '32b': ('UNI-ROYAL', '0805W8F1003T5E'), '32c': ('UNI-ROYAL', '0805W8F1003T5E'),
    '34': ('UNI-ROYAL', '0805W8F2212T5E'), '34a': ('UNI-ROYAL', '0805W8F1001T5E'),
}
DATASHEET = {
    '1': 'esp32-s3-wroom-1_datasheet_v1.8.pdf', '2': 'MPX5010_b.pdf', '3': 'TPS54202.pdf',
    '4': 'onsemi_LM393.pdf; LM393_TI.pdf', '5': 'AO4407A.pdf', '6': 'AO3401A.pdf', '6a': 'JSCJ_2N7002.pdf; 2N7002_Diodes.pdf',
    '7': 'Vishay_SMBJ.pdf; Bourns_SMBJ.pdf', '8': 'Bourns_MF-R.pdf', '9': 'SRN6045TA.pdf', '10': 'SRN6045TA.pdf',
    '11': 'PJ-102AH.pdf', '12': 'Wurth_61300421121.pdf', '13': 'Wurth_61300611121.pdf',
    '14': 'Wurth_61300611121.pdf; SFM3300.pdf', '15': 'Wurth_61300411121.pdf', '16': 'PEC11R.pdf',
    '17': 'TDK_PS_buzzer.pdf', '18': 'Murata_GRM21BR61H106KE43.pdf', '18a': 'Samsung_CL21.pdf', '18b': 'Panasonic_FC.pdf',
    '19': 'Samsung_CL21.pdf', '20': 'Yageo_CC0805_X7R.pdf', '21': 'Samsung_CL21.pdf', '22': 'Samsung_CL21.pdf',
    '23': 'Yageo_CC0805_NP0.pdf', '24': 'FH_0805CG.pdf', '25': 'FH_0805CG.pdf',
    '26': 'UNIROYAL_0805W8F.pdf', '27': 'UNIROYAL_0805W8F.pdf', '28': 'UNIROYAL_0805W8F.pdf', '29': 'UNIROYAL_0805W8F.pdf',
    '29a': 'Bourns_3224.pdf', '30': 'UNIROYAL_0805W8F.pdf', '31': 'Panasonic_ERJP06.pdf', '32': 'Yageo_RT.pdf',
    '32a': 'UNIROYAL_0805W8F.pdf', '32b': 'UNIROYAL_0805W8F.pdf', '32c': 'UNIROYAL_0805W8F.pdf', '33': 'Yageo_RT.pdf',
    '34': 'UNIROYAL_0805W8F.pdf', '34a': 'UNIROYAL_0805W8F.pdf',
}
DS_DIR = os.path.abspath(os.path.join(PRJ, '..', '..', 'tools', 'datasheets'))
missing_ds = []
for r in rows:
    if r['Item'] in FITTED:
        maker, mpn = FITTED[r['Item']]
        if mpn != r['MPN']:
            old = f"{r['Manufacturer']} {r['MPN']} (design-time MPN, datasheet not in repo)"
            r['Alternate MPN'] = old + ('; ' + r['Alternate MPN'] if r['Alternate MPN'] else '')
            r['Manufacturer'], r['MPN'] = maker, mpn
    if r['Item'] in DATASHEET:
        r['Datasheet'] = DATASHEET[r['Item']]
        missing_ds += [f for f in DATASHEET[r['Item']].split('; ') if not os.path.isfile(os.path.join(DS_DIR, f))]
    elif r['Mount'] in ('SMD', 'THT'):
        missing_ds.append(f"item {r['Item']} has no datasheet")
    else:
        r['Datasheet'] = 'n/a (board feature, no part)'
if missing_ds:
    sys.exit(f'datasheet check failed: {missing_ds}')
# Links for later use: official datasheet link per file (tools/datasheets/SOURCES.csv, each one checked against the
# stored file) and the LCSC product page for the LCSC code.
with open(os.path.join(DS_DIR, 'SOURCES.csv'), encoding='utf-8') as fh:
    DS_URL = {s['File']: s['URL'] for s in csv.DictReader(fh)}
for r in rows:
    files = r['Datasheet'].split('; ') if r['Item'] in DATASHEET else []
    unlisted = [f for f in files if f not in DS_URL]
    if unlisted:
        sys.exit(f'datasheet link check failed: {unlisted} not in tools/datasheets/SOURCES.csv')
    r['Datasheet URL'] = '; '.join(DS_URL[f] or f'{f}: no checked link (see tools/datasheets/SOURCES.csv)' for f in files)
    r['LCSC URL'] = f"https://www.lcsc.com/product-detail/{r['LCSC #']}.html" if r['LCSC #'] else ''
with open(os.path.join(OUT, f'BOM_RevC_v3.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# JLC PCBA upload pair: SMD lines only (THT is hand-soldered, see FAB_NOTES), designators comma-separated,
# package names instead of library footprint names (a cap named R_0805 can mis-match), CPL holding exactly
# the BOM's designators (JLC rejects CPL rows that are not in the BOM).
jlc_rows, jlc_des = [], set()
for r in rows:
    if r['Mount'] != 'SMD':
        continue
    des = expand(r['Designators'])
    jlc_des.update(des)
    code, lib, part = LCSC.get(r['Item'], ('', '', ''))
    jlc_rows.append({'Comment': r['Value'], 'Designator': ','.join(des), 'Footprint': JLC_PKG.get(r['Item'], '0805'),
                     'LCSC Part #': code, 'Manufacturer': r['Manufacturer'] if r['Item'] != '7' else 'Vishay',
                     'MPN': JLC_MPN.get(r['Item'], r['MPN']), 'Quantity': len(des),
                     'Notes': (f'JLC {lib}' + (f'; fitted part {part}' if part else '')) if code else NOT_IN_JLC.get(r['Item'], '')})
# one line per LCSC part: 100k (items 32, 32a) and 22 uF (18a 25 V, 19 16 V -> both fitted as the 25 V part)
merged = []
for j in jlc_rows:
    m = next((k for k in merged if j['LCSC Part #'] and k['LCSC Part #'] == j['LCSC Part #']), None)
    if m is None:
        merged.append(dict(j))
        continue
    des = sorted(m['Designator'].split(',') + j['Designator'].split(','), key=dkey)
    m['Designator'], m['Quantity'] = ','.join(des), len(des)
    if j['Comment'] != m['Comment']:
        m['Comment'] = max(m['Comment'], j['Comment'], key=lambda s: s.endswith('25V'))   # name the fitted rating
    if j['MPN'] != m['MPN']:
        m['MPN'] += ' / ' + j['MPN']
jlc_rows = merged
with open(os.path.join(OUT, f'{NAME}_BOM_JLC.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(jlc_rows[0].keys()))
    w.writeheader(); w.writerows(jlc_rows)
with open(os.path.join(OUT, f'{NAME}_CPL_JLC.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['Designator', 'Mid X', 'Mid Y', 'Layer', 'Rotation'])
    for c in sorted(comps, key=lambda c: dkey(c['Designator'])):
        if c['Designator'] in jlc_des:
            w.writerow([c['Designator'], f"{float(c['X']) * 0.0254:.3f}mm", f"{float(c['Y']) * 0.0254:.3f}mm",
                        'Top' if 'Top' in c['Layer'] else 'Bottom', int(float(c['Rotation'])) % 360])
# pin-1 positions of the polarised SMD parts, to compare with JLC's placement preview
pads_by = {}
for p in dump['pads']:
    if p['Component'] and p['Name'] == '1':
        pads_by[p['Component']] = (float(p['X']) * 0.0254, float(p['Y']) * 0.0254)
with open(os.path.join(OUT, f'{NAME}_pin1_check.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['Designator', 'Package', 'Pin 1 X (mm)', 'Pin 1 Y (mm)', 'Pin 1 is', 'Silkscreen mark'])
    marks = {'D2': ('cathode (VIN12V)', 'bar on the right of D2'),
             'U1': ('GND', 'dot at the top-left corner; CPL point = module body centre, 3.62 mm above the pad '
                           'centre, antenna end towards the board edge'),
             'RV1': ('pin 1 = CCW end (5V)', 'none; pin 2 (wiper) is the single 1.6 x 2.0 mm pad, pins 1 and 3 the '
                                            'two 1.6 x 1.3 mm pads (Bourns 3224 DS p.1)')}
    for d in ('U1', 'U2', 'U3', 'U7', 'U8', 'U9', 'Q1', 'Q2', 'Q3', 'Q4', 'D2', 'RV1'):
        x, y = pads_by[d]
        pkg = next(r for r in jlc_rows if d in r['Designator'].split(','))['Footprint']
        w.writerow([d, pkg, f'{x:.2f}', f'{y:.2f}', *marks.get(d, ('pin 1', 'dot next to pin 1' if d not in ('Q2', 'Q3') else 'none (3-pin SOT-23)'))])
print('JLC BOM lines', len(jlc_rows), 'placements', len(jlc_des))

# cross-check: every placed part is on exactly one BOM line, and comments agree
bom_des = {}
for r in rows:
    for d in expand(r['Designators']):
        bom_des.setdefault(d, []).append(r)
board = {c['Designator']: c for c in comps}
problems = []
for d, c in board.items():
    if d not in bom_des:
        problems.append(('not in BOM', d))
    elif len(bom_des[d]) > 1:
        problems.append(('on several BOM lines', d))
    else:
        r = bom_des[d][0]
        if int(r['Qty']) != len(expand(r['Designators'])):
            problems.append(('qty mismatch', r['Item']))
        if not (c['Comment'].startswith(r['Value']) or c['Comment'] == r['MPN']):   # comment may add "1%", "C0G", "(5V)"
            problems.append(('value differs', d, r['Value'], c['Comment']))
extra = [d for d in bom_des if d not in board and not d.startswith(('MH', 'FID', 'TP'))]
print('BOM lines', len(rows), 'parts on board', len(board), 'problems', problems, 'BOM-only', extra)
print('wrote', sorted(os.listdir(OUT)))
