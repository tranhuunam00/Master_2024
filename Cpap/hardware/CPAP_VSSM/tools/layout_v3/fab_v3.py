"""Assemble the SiPAP Rev C layout-v3 fabrication package in Fab_RevC_v3/.

Inputs (made in Altium through the eda-agent bridge):
  - Project Outputs for CPAP_VSSM/v3/Fabrication/  (OutJob SiPAP_Fab.OutJob: Gerber RS-274X mm 4:4,
    Excellon mm 4:3 with G85 slots, IPC-D-356 test-point netlist, ODB++)
  - Project Outputs for CPAP_VSSM/v3docs/Documentation/Schematic Print/Schematic Prints.PDF
  - board dump JSON (pads, vias, components) from obj_query
  - Fab_RevC_v2/BOM_RevC_v2.csv as the base of the BOM (lines edited below)
Usage: python fab_v3.py <board_dump.json>
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
# LCSC / JLC parts, looked up 2026-11-22 with the lcsc skill (jlcsearch). (code, JLC library, part MPN if
# it differs from the BOM's MPN). Passives use JLC *basic* parts of the same value, tolerance, voltage
# and dielectric, as the BOM's substitution rule allows. '' = not found in the JLC library index.
LCSC = {
    '1': ('C2913201', 'extended', ''),
    '2': ('C3238865', 'extended', 'MPXV5010GC6T1 = same sensor on tape and reel (NXP MPX5010 DS p.5 ordering table: '
                                  'Gauge, Axial Port, SMT, case 482A); LCSC stock was 3 on 2026-11-22 for 2 needed'),
    '3': ('C191884', 'extended', ''),
    '4': ('C7955', 'basic', 'LM393DR2G (onsemi), same SOIC-8 pinout'), '5': ('C16072', 'extended', 'AOS'),
    '6': ('C15127', 'basic', 'AOS'), '6a': ('C8545', 'basic', '2N7002 (CJ)'),
    '7': ('C1973187', 'extended', 'SMBJ13CA-E3/52 (Vishay), bidirectional alternate: same row as SMBJ13A in Bourns SMBJ '
                                  'DS p.2 (VRWM 13 V, VBR 14.4-15.9 V, VC 21.5 V @ 28 A 10/1000 us); D2 is on VIN12V after '
                                  'Q4, so it never sees a reversed supply and either orientation is fine'),
    '8': ('C208488', 'extended', ''), '9': ('', '', ''), '10': ('', '', ''), '11': ('C3096093', 'extended', ''),
    '17': ('C76871', 'extended', ''), '18': ('C440198', 'basic', 'GRM21BR61H106KE43L 10uF 50V X5R'),
    '18a': ('C45783', 'basic', 'CL21A226MAQNNNE 22uF 25V X5R'), '18b': ('C178666', 'extended', ''),
    '19': ('C45783', 'basic', 'CL21A226MAQNNNE 22uF 25V X5R'), '20': ('C49678', 'basic', 'CC0805KRX7R9BB104 100nF 50V X7R'),
    '21': ('C28323', 'basic', 'CL21B105KBFNNNE 1uF 50V X7R'), '22': ('C1710', 'basic', ''),
    '23': ('C277525', 'extended', 'CC0805JRNPOYBN471 470pF 50V NP0'), '24': ('C1838', 'extended', '0805CG750J500NT 75pF 50V C0G'),
    '25': ('C1828', 'extended', '0805CG560J500NT 56pF 50V C0G'), '26': ('C17414', 'basic', '0805W8F1002T5E'),
    '27': ('C4328', 'basic', '0805W8F2002T5E'), '28': ('C17673', 'basic', '0805W8F4701T5E'),
    '29': ('C17713', 'basic', '0805W8F4702T5E'), '30': ('C17514', 'basic', '0805W8F1004T5E'),
    '31': ('C441993', 'extended', 'ERJP06F3303V'), '32': ('C149504', 'basic', '0805W8F1003T5E'),
    '32a': ('C149504', 'basic', '0805W8F1003T5E'), '33': ('C17446', 'extended', '0805W8F1332T5E'),
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
           '5': 'SOIC-8', '6': 'SOT-23', '6a': 'SOT-23', '7': 'SMB (DO-214AA)', '9': 'SMD 6x6mm', '10': 'SMD 6x6mm'}
for r in rows:
    code, lib, part = LCSC.get(r['Item'], ('', '', ''))
    if code:
        r['LCSC #'] = code
        extra = f'LCSC {code} ({lib}{", " + part if part else ""})'
        r['Notes'] = (r['Notes'] + '; ' if r['Notes'] else '') + extra
    elif r['Item'] in NOT_IN_JLC:
        r['Notes'] = (r['Notes'] + '; ' if r['Notes'] else '') + NOT_IN_JLC[r['Item']]
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
                           'centre, antenna end towards the board edge')}
    for d in ('U1', 'U2', 'U3', 'U7', 'U8', 'U9', 'Q1', 'Q2', 'Q3', 'Q4', 'D2'):
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
