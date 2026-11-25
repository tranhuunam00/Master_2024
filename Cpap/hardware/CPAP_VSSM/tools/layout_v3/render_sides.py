# Realistic top/bottom render of an Altium Gerber set (RS-274X + Excellon).
# Usage: python -I render_sides.py <gerber_dir> <basename> <out_dir> [ppm]
import re, sys, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

G, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PPM = float(sys.argv[4]) if len(sys.argv) > 4 else 24.0  # pixels per mm


def arc_pts(x0, y0, x1, y1, i, j, cw):
    cx, cy = x0 + i, y0 + j
    r = math.hypot(x0 - cx, y0 - cy)
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    if cw:
        while a1 >= a0: a1 -= 2 * math.pi
    else:
        while a1 <= a0: a1 += 2 * math.pi
    if abs(x0 - x1) < 1e-9 and abs(y0 - y1) < 1e-9:
        a1 = a0 - 2 * math.pi if cw else a0 + 2 * math.pi
    n = max(8, int(abs(a1 - a0) * max(r, 0.05) * 40))
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(1, n + 1)]


def parse(path):
    txt = open(path, encoding='latin-1').read()
    dec = int(re.search(r'%FSLAX(\d)(\d)Y', txt).group(2))
    macros = {}
    for m in re.finditer(r'%AM([A-Za-z_0-9]+)\*(.*?)%', txt, re.S):
        prims = []
        for line in m.group(2).split('*'):
            line = line.strip()
            if not line or line.startswith('0 '): continue
            prims.append([float(v) for v in line.split(',')])
        macros[m.group(1)] = prims
    ap = {}
    for m in re.finditer(r'%ADD(\d+)([A-Za-z_0-9]+),?([^*]*)\*%', txt):
        name = m.group(2)
        if name in macros:
            ap[int(m.group(1))] = ('M', macros[name])
        else:
            ap[int(m.group(1))] = (name, [float(v) for v in m.group(3).split('X') if v])
    body = re.sub(r'%AM.*?%', '', txt, flags=re.S)
    toks = re.findall(r'%[^%]*%|[^*%]+\*', body)
    cur = None; x = y = 0.0; reg = None; pol = 'D'; mode = 1; out = []
    for t in toks:
        if t.startswith('%'):
            if t.startswith('%LPC'): pol = 'C'
            elif t.startswith('%LPD'): pol = 'D'
            continue
        c = t[:-1].strip()
        if not c or c.startswith('G04'): continue
        if c == 'G36': reg = []; continue
        if c == 'G37':
            if reg: out.append(('R', pol, reg))
            reg = None; continue
        for g in re.findall(r'G0?([123])(?!\d)', c):
            mode = int(g)
        m = re.search(r'D(\d+)$', c)
        if m and int(m.group(1)) >= 10 and not re.search(r'[XY]', c):
            cur = int(m.group(1)); continue
        m = re.search(r'(?:X(-?\d+))?(?:Y(-?\d+))?(?:I(-?\d+))?(?:J(-?\d+))?D0?([123])$', c)
        if not m: continue
        nx = int(m.group(1)) / 10 ** dec if m.group(1) else x
        ny = int(m.group(2)) / 10 ** dec if m.group(2) else y
        ii = int(m.group(3)) / 10 ** dec if m.group(3) else 0.0
        jj = int(m.group(4)) / 10 ** dec if m.group(4) else 0.0
        d = m.group(5)
        if d == '1':
            pts = [(nx, ny)] if mode == 1 else arc_pts(x, y, nx, ny, ii, jj, mode == 2)
            if reg is not None:
                if not reg: reg.append([(x, y)])
                reg[-1].extend(pts)
            else:
                out.append(('D', pol, [(x, y)] + pts, ap.get(cur)))
        elif d == '2':
            if reg is not None: reg.append([(nx, ny)])
        elif d == '3' and reg is None:
            out.append(('F', pol, nx, ny, ap.get(cur)))
        x, y = nx, ny
    return out


class Canvas:
    def __init__(self, box, ppm):
        self.x0, self.y0, self.x1, self.y1 = box
        self.ppm = ppm
        self.W = int(round((self.x1 - self.x0) * ppm)); self.H = int(round((self.y1 - self.y0) * ppm))

    def T(self, x, y):
        return ((x - self.x0) * self.ppm, (self.y1 - y) * self.ppm)

    def raster(self, ops):
        im = Image.new('L', (self.W, self.H), 0); dr = ImageDraw.Draw(im); P = self.ppm
        for o in ops:
            col = 255 if o[1] == 'D' else 0
            if o[0] == 'R':
                for poly in o[2]:
                    if len(poly) > 2: dr.polygon([self.T(*p) for p in poly], fill=col)
            elif o[0] == 'F':
                _, _, x, y, a = o
                if not a: continue
                k, p = a
                if k == 'M':
                    for pr in p:
                        if pr[0] == 1:  # circle: 1,exp,dia,cx,cy
                            cx, cy = self.T(x + pr[3], y + pr[4]); r = pr[2] / 2 * P
                            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
                        elif pr[0] == 21:  # center rect: 21,exp,w,h,cx,cy,rot
                            w, h, cx0, cy0, rot = pr[2], pr[3], pr[4], pr[5], math.radians(pr[6])
                            pts = []
                            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                                px, py = cx0 + sx * w / 2, cy0 + sy * h / 2
                                rx = px * math.cos(rot) - py * math.sin(rot); ry = px * math.sin(rot) + py * math.cos(rot)
                                pts.append(self.T(x + rx, y + ry))
                            dr.polygon(pts, fill=col)
                    continue
                cx, cy = self.T(x, y)
                if k == 'C':
                    r = p[0] / 2 * P; dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
                elif k == 'R':
                    w = p[0] / 2 * P; h = p[1] / 2 * P; dr.rectangle([cx - w, cy - h, cx + w, cy + h], fill=col)
                elif k == 'O':
                    w = p[0] / 2 * P; h = p[1] / 2 * P; r = min(w, h)
                    dr.rounded_rectangle([cx - w, cy - h, cx + w, cy + h], radius=r, fill=col)
                elif k == 'P':
                    n = int(p[1]); r = p[0] / 2 * P; rot = math.radians(p[2]) if len(p) > 2 else 0
                    dr.polygon([(cx + r * math.cos(rot + 2 * math.pi * q / n), cy - r * math.sin(rot + 2 * math.pi * q / n)) for q in range(n)], fill=col)
            else:
                _, _, pts, a = o
                w = a[1][0] if a and a[1] else 0.1
                tp = [self.T(*q) for q in pts]
                dr.line(tp, fill=col, width=max(1, int(round(w * P))))
                r = w / 2 * P
                for cx, cy in tp:
                    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        return np.array(im) > 127


def drills(path):
    txt = open(path, encoding='latin-1').read()
    tools = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'T(\d+)F\d+S\d+C([\d.]+)', txt)}
    fmt = re.search(r'FILE_FORMAT=(\d):(\d)', txt); ndig, dec = int(fmt.group(1)), int(fmt.group(2))
    lz = bool(re.search(r'^METRIC,LZ|^INCH,LZ', txt, re.M))  # LZ: leading zeros kept, trailing suppressed
    body = txt.split('%', 1)[1] if '%' in txt else txt
    cur = None; x = y = 0.0; holes = []
    for line in body.splitlines():
        line = line.strip()
        m = re.match(r'^T(\d+)$', line)
        if m: cur = int(m.group(1)); continue
        m = re.match(r'^(?:X(-?\d+))?(?:Y(-?\d+))?(?:G85(?:X(-?\d+))?(?:Y(-?\d+))?)?$', line)
        if m and (m.group(1) or m.group(2)):
            def v(s, old):
                if not s: return old
                neg = s.startswith('-'); s = s.lstrip('-')
                if lz: s = s.ljust(ndig + dec, '0')
                return (-1 if neg else 1) * int(s) / 10 ** dec
            x = v(m.group(1), x); y = v(m.group(2), y)
            if 'G85' in line:
                x2 = v(m.group(3), x); y2 = v(m.group(4), y)
                holes.append((x, y, x2, y2, tools.get(cur, 0.3)))
                x, y = x2, y2
            else:
                holes.append((x, y, x, y, tools.get(cur, 0.3)))
    return holes


f = lambda e: os.path.join(G, f'{BASE}.{e}')
outline = parse(f('GM'))
xs = [p[0] for o in outline if o[0] == 'D' for p in o[2]]; ys = [p[1] for o in outline if o[0] == 'D' for p in o[2]]
bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
M = 3.0
cv = Canvas((bx0 - M, by0 - M, bx1 + M, by1 + M), PPM)
board = np.zeros((cv.H, cv.W), bool)
a, b = cv.T(bx0, by1), cv.T(bx1, by0)
board[int(a[1]):int(b[1]), int(a[0]):int(b[0])] = True

hole_im = Image.new('L', (cv.W, cv.H), 0); hd = ImageDraw.Draw(hole_im)
for name in (f'{BASE}-RoundHoles.TXT', f'{BASE}-SlotHoles.TXT'):
    p = os.path.join(G, name)
    if not os.path.exists(p): continue
    for x1, y1, x2, y2, d in drills(p):
        A, B = cv.T(x1, y1), cv.T(x2, y2); r = d / 2 * PPM
        hd.line([A, B], fill=255, width=max(1, int(round(d * PPM))))
        for cx, cy in (A, B): hd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
holes = np.array(hole_im) > 127


def side(cu, mask, silk, mirror, title):
    C = cv.raster(parse(f(cu))); S = cv.raster(parse(f(mask))); K = cv.raster(parse(f(silk)))
    img = np.zeros((cv.H, cv.W, 3), np.float32)
    img[:] = (34, 34, 38)                                   # background
    sub = np.array((22, 92, 48), np.float32)                # FR4 under green mask
    cu_mask = np.array((60, 150, 80), np.float32)           # copper under mask
    gold = np.array((212, 175, 85), np.float32)             # ENIG exposed copper
    fr4 = np.array((170, 150, 95), np.float32)              # bare laminate in mask opening
    img[board] = sub
    img[board & C] = cu_mask
    img[board & S & ~C] = fr4
    img[board & S & C] = gold
    img[board & K & ~S] = (245, 245, 240)
    img[holes] = (18, 18, 20)
    # thin board edge
    edge = board & ~np.array(Image.fromarray(board.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))).astype(bool)
    img[edge] = (12, 60, 30)
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    if mirror: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    pad = 44
    canvas = Image.new('RGB', (im.width, im.height + pad), (34, 34, 38)); canvas.paste(im, (0, pad))
    d = ImageDraw.Draw(canvas)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype('arialbd.ttf', 26)
    except Exception:
        font = None
    d.text((14, 8), title, fill=(235, 235, 235), font=font)
    return canvas


os.makedirs(OUT, exist_ok=True)
w, h = bx1 - bx0, by1 - by0
side('GTL', 'GTS', 'GTO', False, f'SiPAP Rev C v3 - TOP  ({w:.1f} x {h:.1f} mm)').save(os.path.join(OUT, 'SiPAP_v3_TOP_render.png'))
side('GBL', 'GBS', 'GBO', True, f'SiPAP Rev C v3 - BOTTOM (nhin tu duoi, lat guong)').save(os.path.join(OUT, 'SiPAP_v3_BOTTOM_render.png'))
print('board', bx0, by0, bx1, by1, 'px', cv.W, cv.H)
