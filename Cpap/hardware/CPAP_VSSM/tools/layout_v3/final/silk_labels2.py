"""Second pass: place the labels that failed, searching outward from a preferred anchor.
Hidden comment texts are ignored (they are not printed and DRC ignores them)."""
import json, math
from shapely.geometry import box, LineString, Point
from edacall import call, BOARD

GAP, EDGE, W = 6.5, 20, 6
call('app_set_active_document', {'file_path': BOARD})
pads = call('obj_query', {'object_type': 'ePadObject', 'properties': 'X,Y,TopXSize,TopYSize,Rotation', 'limit': 0})['objects']
silk = call('obj_query', {'object_type': 'eTrackObject', 'properties': 'X1,Y1,X2,Y2,Width', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']
arcs = call('obj_query', {'object_type': 'eArcObject', 'properties': 'XCenter,YCenter,Radius,Width', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']
obs = []
for p in pads:
    w, h = float(p['TopXSize']), float(p['TopYSize'])
    if p['Rotation'] in ('90', '270'):
        w, h = h, w
    x, y = float(p['X']), float(p['Y'])
    obs.append(box(x - w / 2 - 4 - GAP, y - h / 2 - 4 - GAP, x + w / 2 + 4 + GAP, y + h / 2 + 4 + GAP))
for s in silk:
    obs.append(LineString([(float(s['X1']), float(s['Y1'])), (float(s['X2']), float(s['Y2']))]).buffer(float(s['Width']) / 2 + GAP))
for a in arcs:
    obs.append(Point(float(a['XCenter']), float(a['YCenter'])).buffer(float(a['Radius']) + float(a['Width']) / 2 + GAP))


def visible_texts():
    t = call('obj_query', {'object_type': 'eTextObject', 'properties': 'Text,X,Y,IsComment,BoundsLeft,BoundsBottom,BoundsRight,BoundsTop', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']
    return [box(float(x['BoundsLeft']), float(x['BoundsBottom']), float(x['BoundsRight']), float(x['BoundsTop'])).buffer(GAP)
            for x in t if x['IsComment'] not in ('true', 'True')]


def bad(g, texts):
    b = g.bounds
    if b[0] < EDGE or b[1] < EDGE or b[2] > 4000 - EDGE or b[3] > 3000 - EDGE:
        return True
    return any(g.intersects(o) for o in obs) or any(g.intersects(t) for t in texts)


JOBS = [  # text, height, rotation, anchor, max search radius
    ("SiPAP Rev C v3 2026-11", 50, 0, (2620, 1430), 700),
    ("BLOWER", 40, 0, (350, 340), 250),
    ("12V", 40, 0, (1272, 2200), 160),
    ("5V", 40, 0, (1170, 1190), 160),
    ("GND", 40, 0, (3700, 2650), 40),
    ("UART", 40, 90, (3885, 2400), 120),
]
out = []
for text, h, rot, (ax, ay), rmax in JOBS:
    r0 = call('pcb_place_text', {'text': text, 'x': ax, 'y': ay, 'layer': 'TopOverlay', 'height': h, 'rotation': rot,
                                 'stroke': W, 'kind': 'stroke', 'justification': 'center'})
    bl = r0['bounds']; bw, bh = bl[2] - bl[0], bl[3] - bl[1]
    cx0, cy0 = (bl[0] + bl[2]) / 2, (bl[1] + bl[3]) / 2
    call('obj_delete', {'object_type': 'eTextObject', 'filter': 'Layer=TopOverlay|Text=%s|X=%s|Y=%s' % (text, r0['location'][0], r0['location'][1])}) \
        if False else None
    # delete the probe: find it by bounds
    q = call('obj_query', {'object_type': 'eTextObject', 'properties': 'Text,X,Y,BoundsLeft,BoundsBottom', 'filter': 'Layer=TopOverlay|Text=' + text, 'limit': 0})['objects']
    for t in q:
        if abs(float(t['BoundsLeft']) - bl[0]) < 0.6 and abs(float(t['BoundsBottom']) - bl[1]) < 0.6:
            call('obj_delete', {'object_type': 'eTextObject', 'filter': 'Layer=TopOverlay|Text=%s|X=%s|Y=%s' % (text, t['X'], t['Y'])})
    texts = visible_texts()
    best = None
    for r in range(0, rmax + 1, 5):
        for a in range(0, 360, 10 if r else 360):
            x = ax + r * math.cos(math.radians(a)); y = ay + r * math.sin(math.radians(a))
            g = box(x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)
            if not bad(g, texts):
                best = (round(x, 1), round(y, 1)); break
        if best:
            break
    if not best:
        print('NO SPOT', text); out.append((text, None)); continue
    # place so that the bounds centre lands on best
    px, py = ax + (best[0] - cx0), ay + (best[1] - cy0)
    r1 = call('pcb_place_text', {'text': text, 'x': px, 'y': py, 'layer': 'TopOverlay', 'height': h, 'rotation': rot,
                                 'stroke': W, 'kind': 'stroke', 'justification': 'center'})
    print(text, 'anchor', (ax, ay), '->', best, 'bounds', [round(v, 1) for v in r1['bounds']])
    out.append((text, best))
json.dump(out, open('silk_labels2_result.json', 'w'), indent=1)
