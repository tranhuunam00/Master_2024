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

# ---------------- Pick and place (mm, origin = board lower-left, top side) ----------------
dump = json.load(open(sys.argv[1], encoding='utf-8'))
comps = dump['comps']
THT = {'J1', 'J2', 'J3', 'U4', 'U5', 'U6', 'F1', 'BZ1', 'C25'}


def dkey(d):
    return (''.join(c for c in d if c.isalpha()), int(''.join(c for c in d if c.isdigit()) or 0))


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
    if des == ['U6']:
        r['Notes'] = ('31.5 mm tall above PCB: 6.5 mm body + L = 25 mm shaft (Bourns PEC11R DS p.2-3 shaft table); '
                      'STEP model updated to this height')
with open(os.path.join(OUT, f'BOM_RevC_v3.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# cross-check: every placed part is on exactly one BOM line, and comments agree
def expand(s):
    out = []
    for tok in s.split():
        if '-' in tok:           # MH1-MH4 style
            a, b = tok.split('-'); p = ''.join(c for c in a if c.isalpha())
            out += [f'{p}{i}' for i in range(int(a[len(p):]), int(b[len(p):]) + 1)]
        else:
            out.append(tok)
    return out


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
