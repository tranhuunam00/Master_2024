"""Add the polarity / pin-1 marks the footprints lack on the top silkscreen:
- pin-1 dot (14 mil round dot) for SO-8, SOT-23-6, MPXV, headers;
- cathode bar for D2 (beside pad 1 = cathode, outside the pad);
- '+' beside pad 1 of C25 (electrolytic) and BZ1 (buzzer).
Each mark goes to the free spot (pads + mask + 6.5 mil, silk + 6.5, texts + 6.5, board edge 20)
that is nearest pad 1 and clearly farther from every other pad of the same part."""
import json, math
from shapely.geometry import box, LineString, Point
from edacall import call, BOARD

GAP, EDGE, MASK = 6.5, 20, 4
call('app_set_active_document', {'file_path': BOARD})
pads = call('obj_query', {'object_type': 'ePadObject', 'properties': 'Component,Name,X,Y,TopXSize,TopYSize,Rotation', 'limit': 0})['objects']
silk = call('obj_query', {'object_type': 'eTrackObject', 'properties': 'X1,Y1,X2,Y2,Width', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']
arcs = call('obj_query', {'object_type': 'eArcObject', 'properties': 'XCenter,YCenter,Radius,Width', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']
texts = call('obj_query', {'object_type': 'eTextObject', 'properties': 'Text,IsComment,BoundsLeft,BoundsBottom,BoundsRight,BoundsTop', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']


def padbox(p, extra=0.0):
    w, h = float(p['TopXSize']), float(p['TopYSize'])
    if p['Rotation'] in ('90', '270'):
        w, h = h, w
    x, y = float(p['X']), float(p['Y'])
    return box(x - w / 2 - extra, y - h / 2 - extra, x + w / 2 + extra, y + h / 2 + extra)


obs = [padbox(p, MASK + GAP) for p in pads]
obs += [LineString([(float(s['X1']), float(s['Y1'])), (float(s['X2']), float(s['Y2']))]).buffer(float(s['Width']) / 2 + GAP) for s in silk]
obs += [Point(float(a['XCenter']), float(a['YCenter'])).buffer(float(a['Radius']) + float(a['Width']) / 2 + GAP) for a in arcs]
obs += [box(float(t['BoundsLeft']), float(t['BoundsBottom']), float(t['BoundsRight']), float(t['BoundsTop'])).buffer(GAP)
        for t in texts if t['IsComment'] not in ('true', 'True')]


def free(g):
    b = g.bounds
    if b[0] < EDGE or b[1] < EDGE or b[2] > 4000 - EDGE or b[3] > 3000 - EDGE:
        return False
    return not any(g.intersects(o) for o in obs)


def best_spot(ref, shape_fn, rmin, rmax):
    own = [p for p in pads if p['Component'] == ref]
    p1 = [p for p in own if p['Name'] == '1'][0]
    others = [padbox(p) for p in own if p['Name'] != '1']
    b1 = padbox(p1)
    x1, y1 = float(p1['X']), float(p1['Y'])
    best = None
    for r in range(rmin, rmax + 1, 3):
        for a in range(0, 360, 10):
            x = x1 + r * math.cos(math.radians(a)); y = y1 + r * math.sin(math.radians(a))
            g = shape_fn(x, y)
            if not free(g):
                continue
            d1 = g.distance(b1)
            do = min((g.distance(o) for o in others), default=1e9)
            score = do - 2.0 * d1          # close to pad 1, clearly away from the other pads
            if do < d1 + 25:
                continue
            if best is None or score > best[0]:
                best = (score, x, y, round(d1, 1), round(do, 1))
        if best and r > rmin + 30:
            break
    return best


tracks, report = [], []
DOT = 14
for ref in ('Q1', 'Q4', 'U7', 'U8', 'U9', 'U2', 'U3', 'J2', 'J3', 'U4', 'U5', 'U1'):
    b = best_spot(ref, lambda x, y: Point(x, y).buffer(DOT / 2), 20, 140)
    if not b:
        report.append((ref, 'NO SPOT')); continue
    _, x, y, d1, do = b
    tracks.append({'x1': x - 0.5, 'y1': y, 'x2': x + 0.5, 'y2': y, 'width': DOT, 'layer': 'TopOverlay'})
    obs.append(Point(x, y).buffer(DOT / 2 + GAP))
    report.append((ref, 'dot', round(x, 1), round(y, 1), 'to pad1', d1, 'to others', do))

# D2 cathode bar: parallel to the pad's long side, outside pad 1
d2 = [p for p in pads if p['Component'] == 'D2']
p1 = [p for p in d2 if p['Name'] == '1'][0]; p2 = [p for p in d2 if p['Name'] != '1'][0]
bx1 = padbox(p1)
dx = float(p1['X']) - float(p2['X'])
if abs(dx) > 1:  # pads along x: vertical bar beyond pad 1
    side = 1 if dx > 0 else -1
    xb = (bx1.bounds[2] if side > 0 else bx1.bounds[0]) + side * (MASK + GAP + 5)
    y0, y1_ = bx1.bounds[1], bx1.bounds[3]
    g = LineString([(xb, y0), (xb, y1_)]).buffer(5)
    ok = free(g)
    report.append(('D2', 'cathode bar', round(xb, 1), round(y0, 1), round(y1_, 1), 'free' if ok else 'BLOCKED'))
    if ok:
        tracks.append({'x1': xb, 'y1': y0, 'x2': xb, 'y2': y1_, 'width': 10, 'layer': 'TopOverlay'})
        obs.append(g.buffer(GAP))

# '+' marks for C25 and BZ1 (two strokes, 40 mil, 6 mil wide)
PL = 20
for ref in ('C25', 'BZ1'):
    b = best_spot(ref, lambda x, y: box(x - PL - 3, y - PL - 3, x + PL + 3, y + PL + 3), 30, 160)
    if not b:
        report.append((ref, 'NO SPOT for +')); continue
    _, x, y, d1, do = b
    tracks.append({'x1': x - PL, 'y1': y, 'x2': x + PL, 'y2': y, 'width': 6, 'layer': 'TopOverlay'})
    tracks.append({'x1': x, 'y1': y - PL, 'x2': x, 'y2': y + PL, 'width': 6, 'layer': 'TopOverlay'})
    obs.append(box(x - PL - 3, y - PL - 3, x + PL + 3, y + PL + 3).buffer(GAP))
    report.append((ref, '+', round(x, 1), round(y, 1), 'to pad1', d1, 'to others', do))

for r in report:
    print(r)
res = call('pcb_place_tracks', {'tracks': tracks, 'board_path': BOARD})
print('placed', json.dumps(res)[:300])
json.dump({'tracks': tracks, 'report': report}, open('polarity_marks.json', 'w'), indent=1)
