"""Independent read-back of the v3 Gerber + Excellon files against the board.
Checks: every pad flashed on the right copper/mask/paste layers, every hole drilled once,
copper and silk inside the outline (copper >= 0.2 mm from the edge, JLCPCB routed-edge rule),
vias tented (no mask opening). Usage: python gerber_check.py <gerber_dir> <drill_dir> <board_dump.json>"""
import json, math, re, sys, os, glob

GDIR, DDIR, DUMP = sys.argv[1], sys.argv[2], sys.argv[3]
MIL = 0.0254
W, H = 4000 * MIL, 3000 * MIL


def parse_gerber(path):
    txt = open(path, encoding='latin-1').read()
    fs = re.search(r'%FSLAX(\d)(\d)Y(\d)(\d)\*%', txt)
    dec = int(fs.group(2))
    macros, aps = {}, {}
    for m in re.finditer(r'%AM([A-Za-z0-9_]+)\*(.*?)%', txt, re.S):
        prims = [p.strip() for p in m.group(2).split('*') if p.strip()]
        hx = hy = 0.0
        for p in prims:
            v = [float(x) for x in p.split(',')]
            if v[0] == 21:
                w, h, cx, cy, rot = v[2], v[3], v[4], v[5], v[6] % 180
                if abs(rot - 90) < 1:
                    w, h = h, w
                hx, hy = max(hx, abs(cx) + w / 2), max(hy, abs(cy) + h / 2)
            elif v[0] == 1:
                d, cx, cy = v[2], v[3], v[4]
                hx, hy = max(hx, abs(cx) + d / 2), max(hy, abs(cy) + d / 2)
        macros[m.group(1)] = (hx, hy)
    for m in re.finditer(r'%ADD(\d+)([A-Za-z0-9_]+)(?:,([0-9.X]+))?\*%', txt):
        code, kind, par = int(m.group(1)), m.group(2), m.group(3)
        if kind == 'C':
            d = float(par.split('X')[0]); aps[code] = ('C', d / 2, d / 2)
        elif kind in ('R', 'O'):
            a, b = [float(x) for x in par.split('X')[:2]]; aps[code] = (kind, a / 2, b / 2)
        elif kind in macros:
            aps[code] = ('M',) + macros[kind]
        else:
            aps[code] = ('?', 0, 0)
    body = re.sub(r'%.*?%', '', txt, flags=re.S)
    flashes, draws, regions = [], [], []
    x = y = 0.0; ap = None; interp = 1; region = None; dark = True
    pol = [(m.start(), m.group(1)) for m in re.finditer(r'%LP([DC])\*%', txt)]
    # polarity changes are in % blocks; walk the raw text instead of the stripped body
    for blk in re.split(r'(%LP[DC]\*%)', txt):
        if blk.startswith('%LP'):
            dark = blk[3] == 'D'; continue
        blk = re.sub(r'%.*?%', '', blk, flags=re.S)
        for cmd in blk.split('*'):
            cmd = cmd.strip()
            if not cmd or cmd.startswith('G04'):
                continue
            if cmd.startswith('G36'):
                region = []; continue
            if cmd.startswith('G37'):
                if region and dark:
                    regions.append(region)
                region = None; continue
            g = re.match(r'G0?([123])', cmd)
            if g:
                interp = int(g.group(1))
            d = re.search(r'D(\d+)$', cmd)
            mx = re.search(r'X(-?\d+)', cmd); my = re.search(r'Y(-?\d+)', cmd)
            mi = re.search(r'I(-?\d+)', cmd); mj = re.search(r'J(-?\d+)', cmd)
            nx = int(mx.group(1)) / 10 ** dec if mx else x
            ny = int(my.group(1)) / 10 ** dec if my else y
            if d:
                dc = int(d.group(1))
                if dc >= 10 and not (mx or my):
                    ap = dc; continue
                if dc == 1:
                    pts = [(x, y), (nx, ny)]
                    if interp in (2, 3) and (mi or mj):
                        cx = x + (int(mi.group(1)) / 10 ** dec if mi else 0); cy = y + (int(mj.group(1)) / 10 ** dec if mj else 0)
                        r = math.hypot(x - cx, y - cy); a0 = math.atan2(y - cy, x - cx); a1 = math.atan2(ny - cy, nx - cx)
                        if interp == 3 and a1 <= a0: a1 += 2 * math.pi
                        if interp == 2 and a1 >= a0: a1 -= 2 * math.pi
                        n = max(2, int(abs(a1 - a0) / 0.1))
                        pts = [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]
                    if region is not None:
                        region.extend(pts)
                    elif dark:
                        draws.append((pts, aps[ap][1]))
                elif dc == 2:
                    if region is not None:
                        if region and dark: regions.append(region)
                        region = []
                elif dc == 3 and dark:
                    flashes.append((nx, ny, ap, aps[ap]))
            x, y = nx, ny
    return flashes, draws, regions


def extent_margin(flashes, draws, regions):
    """smallest distance from dark artwork to the board edge (mm), and where"""
    best = (1e9, None)
    def upd(xmin, ymin, xmax, ymax, what):
        nonlocal best
        m = min(xmin, ymin, W - xmax, H - ymax)
        if m < best[0]: best = (m, what)
    for fx, fy, ap, (k, hx, hy) in flashes:
        upd(fx - hx, fy - hy, fx + hx, fy + hy, ('flash', round(fx, 3), round(fy, 3)))
    for pts, r in draws:
        for px, py in pts:
            upd(px - r, py - r, px + r, py + r, ('draw', round(px, 3), round(py, 3)))
    for reg in regions:
        xs = [p[0] for p in reg]; ys = [p[1] for p in reg]
        upd(min(xs), min(ys), max(xs), max(ys), ('region', round(min(xs), 2), round(min(ys), 2)))
    return best


def parse_drill(path):
    txt = open(path, encoding='latin-1').read()
    tools = {int(t): float(d) for t, d in re.findall(r'T(\d+)F\d+S\d+C([0-9.]+)', txt)}
    if not tools:
        tools = {int(t): float(d) for t, d in re.findall(r'T(\d+)C([0-9.]+)', txt)}
    lz = 'LZ' in txt.split('%')[0]
    hits, slots = [], []
    tool = None; x = y = 0.0
    body = txt.split('%', 1)[1] if '%' in txt else txt

    def num(s):
        neg = s.startswith('-'); s = s.lstrip('+-')
        if '.' in s: v = float(s)
        elif lz: v = int(s.ljust(7, '0')) / 1000.0      # 4:3, trailing zeros suppressed
        else: v = int(s) / 1000.0
        return -v if neg else v
    for line in body.splitlines():
        line = line.strip()
        mt = re.match(r'^T(\d+)$', line)
        if mt:
            tool = int(mt.group(1)); continue
        if 'G85' in line:
            a, b = line.split('G85')
            ma = re.search(r'X([-+]?[\d.]+)', a); mb = re.search(r'Y([-+]?[\d.]+)', a)
            x1 = num(ma.group(1)) if ma else x; y1 = num(mb.group(1)) if mb else y
            mc = re.search(r'X([-+]?[\d.]+)', b); md = re.search(r'Y([-+]?[\d.]+)', b)
            x2 = num(mc.group(1)) if mc else x1; y2 = num(md.group(1)) if md else y1
            slots.append((tools[tool], x1, y1, x2, y2)); x, y = x2, y2; continue
        mx = re.search(r'X([-+]?[\d.]+)', line); my = re.search(r'Y([-+]?[\d.]+)', line)
        if (mx or my) and tool is not None and not line.startswith(('M', 'G0')):
            x = num(mx.group(1)) if mx else x; y = num(my.group(1)) if my else y
            hits.append((tools[tool], x, y))
    return tools, hits, slots


dump = json.load(open(DUMP, encoding='utf-8'))
pads, vias = dump['pads'], dump['vias']
files = {os.path.splitext(f)[1].upper(): f for f in glob.glob(os.path.join(GDIR, '*')) if os.path.splitext(f)[1].upper() in
         ('.GTL', '.GBL', '.GTS', '.GBS', '.GTO', '.GBO', '.GTP', '.GBP', '.GM')}
G = {k: parse_gerber(v) for k, v in files.items()}
res = {'layers': {}}
for k, (fl, dr, rg) in G.items():
    m = extent_margin(fl, dr, rg)
    res['layers'][k] = {'flashes': len(fl), 'draws': len(dr), 'regions': len(rg), 'min_edge_mm': round(m[0], 3), 'at': m[1]}

# outline
gm = G['.GM'][1]
xs = [p[0] for pts, r in gm for p in pts]; ys = [p[1] for pts, r in gm for p in pts]
res['outline_mm'] = [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


def has_flash(layer, x, y, tol=0.02):
    for fx, fy, ap, a in G[layer][0]:
        if abs(fx - x) < tol and abs(fy - y) < tol:
            return a
    return None


missing = []
for p in pads:
    x, y = float(p['X']) * MIL, float(p['Y']) * MIL
    lay = p['Layer']; hole = float(p['HoleSize']) > 0
    want = []
    if lay in ('TopLayer', 'MultiLayer'): want += ['.GTL', '.GTS']
    if lay in ('BottomLayer', 'MultiLayer'): want += ['.GBL', '.GBS']
    if lay == 'TopLayer': want += ['.GTP']
    if p['Name'].startswith('MH'):
        want = ['.GTL', '.GBL', '.GTS', '.GBS']
    for L in want:
        if not has_flash(L, x, y):
            missing.append((p['Component'] or '-', p['Name'], L))
res['pads_checked'] = len(pads)
res['pad_flash_missing'] = missing
vmiss, vmask = [], []
for v in vias:
    x, y = float(v['X']) * MIL, float(v['Y']) * MIL
    for L in ('.GTL', '.GBL'):
        if not has_flash(L, x, y): vmiss.append((v['Net'], round(x, 3), round(y, 3), L))
    for L in ('.GTS', '.GBS'):
        if has_flash(L, x, y): vmask.append((v['Net'], round(x, 3), round(y, 3), L))
res['vias_checked'] = len(vias)
res['via_copper_missing'] = vmiss
res['via_mask_openings'] = vmask

# drill
round_f = [f for f in glob.glob(os.path.join(DDIR, '*RoundHoles*.TXT'))][0]
slot_f = [f for f in glob.glob(os.path.join(DDIR, '*SlotHoles*.TXT'))][0]
tools, hits, _ = parse_drill(round_f)
_, _, slots = parse_drill(slot_f)
holes = [(float(p['HoleSize']) * MIL, float(p['X']) * MIL, float(p['Y']) * MIL, p['Component'] + '.' + p['Name'], p.get('HoleType', '0'))
         for p in pads if float(p['HoleSize']) > 0] + [(float(v['HoleSize']) * MIL, float(v['X']) * MIL, float(v['Y']) * MIL, 'via', '0') for v in vias]
used = set(); nomatch = []
for d, x, y, who, ht in holes:
    ok = False
    for i, (td, hx, hy) in enumerate(hits):
        if i not in used and abs(hx - x) < 0.02 and abs(hy - y) < 0.02:
            used.add(i); ok = abs(td - d) < 0.03 or ht not in ('0', 'Round'); ok = True if ok else ('size', td, d)
            break
    if ok is not True:
        sl = [s for s in slots if abs((s[1] + s[3]) / 2 - x) < 0.05 and abs((s[2] + s[4]) / 2 - y) < 0.05]
        if sl:
            ok = True
    if ok is not True:
        nomatch.append((who, round(d, 3), round(x, 3), round(y, 3), ok))
res['drill'] = {'round_hits': len(hits), 'slots': len(slots), 'board_holes': len(holes), 'unmatched_board_holes': nomatch,
                'extra_hits': len(hits) - len(used), 'tools_mm': sorted(set(tools.values()))}
print(json.dumps(res, indent=1))
