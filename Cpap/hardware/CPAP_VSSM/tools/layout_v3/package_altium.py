"""Pack the SiPAP Rev C v3 Altium project into one zip that opens on another PC.

Only the documents the project needs go in. The project file copy drops the three CAMtastic
viewer documents (absolute D:\\Program Files paths on this PC) and the old v1 generated-output
entries, and turns absolute board paths into relative ones. Originals are not touched.
Usage: python package_altium.py  ->  CPAP_VSSM/SiPAP_RevC_v3_Altium.zip
"""
import os, re, zipfile, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(PRJ, 'SiPAP_RevC_v3_Altium.zip')
ROOT = 'SiPAP_RevC_v3_Altium/'

DOCS = ['SiPAP_Main_RevC.SchDoc', 'SiPAP_PCB.PcbDoc', 'SiPAP_Footprints.PcbLib', 'SiPAP_Parts.SchLib',
        'SiPAP_Parts_v2.SchLib', 'SiPAP_Fab.OutJob', 'SiPAP_Docs.OutJob', 'CPAP_VSSM.PrjPcbStructure']
FOLDERS = ['Fab_RevC_v3', 'sim']
EXTRA = ['3D/height_map.md']


def clean_project(raw: bytes) -> bytes:
    txt = raw.decode('latin-1')
    abs_dir = PRJ.replace('/', '\\') + '\\'
    txt = txt.replace(abs_dir, '')                       # absolute -> relative document paths
    sections = re.split(r'(?m)^(?=\[)', txt)
    keep, n = [], 0
    for sec in sections:
        head = sec.split('\n', 1)[0].strip()
        if re.fullmatch(r'\[GeneratedDocument\d+\]', head):
            continue                                     # old v1 outputs, not shipped
        if re.fullmatch(r'\[Document\d+\]', head):
            m = re.search(r'(?m)^DocumentPath=(.*)$', sec)
            path = m.group(1).strip() if m else ''
            if path.lower().endswith('.cam') or ':' in path:
                continue                                 # CAMtastic viewers with D:\ paths
            n += 1
            sec = re.sub(r'^\[Document\d+\]', '[Document%d]' % n, sec)
        keep.append(sec)
    return ''.join(keep).encode('latin-1')


README = """SiPAP Rev C - layout v3 (2026-11-22) - project Altium

Mo project:
  1. Giai nen ca thu muc nay (giu nguyen cau truc).
  2. Altium Designer: File > Open Project > chon CPAP_VSSM.PrjPcb.
     File duoc luu bang Altium Designer Develop 26.11; nen dung Altium Designer 24 tro len.
  3. Trong panel Projects: SiPAP_Main_RevC.SchDoc (schematic), SiPAP_PCB.PcbDoc (board).
     Ban 2D/3D: phim 2 / phim 3 trong cua so PCB.

Noi dung:
  CPAP_VSSM.PrjPcb           project (da bo cac tai lieu CAMtastic va output cu chi co tren may goc)
  SiPAP_Main_RevC.SchDoc     schematic 1 trang
  SiPAP_PCB.PcbDoc           board 2 lop 101.6 x 76.2 mm
  SiPAP_Footprints.PcbLib    thu vien footprint (doi chieu datasheet)
  SiPAP_Parts*.SchLib        thu vien symbol
  SiPAP_Fab.OutJob           xuat Gerber / khoan / IPC-356 / ODB++
  SiPAP_Docs.OutJob          xuat PDF schematic
  Fab_RevC_v3/               ho so dat in: Gerber.zip, BOM, CPL, STEP, PDF, FAB_NOTES.md, REVIEW_v3.md
  sim/                       mo phong hoat dong binh thuong (netlist + hinh)
  3D/height_map.md           chieu cao linh kien cho thiet ke vo

Footprint va symbol da nam san trong SchDoc/PcbDoc, khong can cai thu vien rieng.
"""

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr(ROOT + 'CPAP_VSSM.PrjPcb', clean_project(open(os.path.join(PRJ, 'CPAP_VSSM.PrjPcb'), 'rb').read()))
    z.writestr(ROOT + 'DOC_TRUOC.txt', README.encode('utf-8'))
    for d in DOCS + EXTRA:
        z.write(os.path.join(PRJ, d), ROOT + d)
    for folder in FOLDERS:
        for base, _, files in os.walk(os.path.join(PRJ, folder)):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, ROOT + os.path.relpath(full, PRJ).replace('\\', '/'))

with zipfile.ZipFile(OUT) as z:
    bad = z.testzip()
    names = z.namelist()
    same = hashlib.md5(z.read(ROOT + 'SiPAP_PCB.PcbDoc')).hexdigest() == hashlib.md5(open(os.path.join(PRJ, 'SiPAP_PCB.PcbDoc'), 'rb').read()).hexdigest()
    prj = z.read(ROOT + 'CPAP_VSSM.PrjPcb').decode('latin-1')
print('zip', OUT, round(os.path.getsize(OUT) / 1e6, 2), 'MB,', len(names), 'files, crc', 'ok' if bad is None else bad, ', PcbDoc identical:', same)
print('project documents:', re.findall(r'(?m)^DocumentPath=(.*)$', prj))
print('absolute paths left in project:', len(re.findall(r'[A-Z]:\\\\', prj)))
