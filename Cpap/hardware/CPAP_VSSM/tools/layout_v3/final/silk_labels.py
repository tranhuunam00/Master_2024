"""Place v3 silkscreen labels (board name, connector and test-point labels) without touching
pads (mask + 6 mil), other silk (6 mil) or the board edge. Each label lists candidate anchors in
order of preference; the first clean one is kept (place, read back bounds, check, else delete)."""
import json, sys
from shapely.geometry import box, LineString
from edacall import call, BOARD

H, W = 40, 6          # text height / stroke (mil): 1.0 mm / 0.15 mm (JLC min line 0.15 mm)
GAP = 6.5             # silk-to-mask and silk-to-silk clearance target (rule = 6 mil)
EDGE = 20
call('app_set_active_document', {'file_path': BOARD})

pads = call('obj_query', {'object_type': 'ePadObject', 'properties': 'X,Y,TopXSize,TopYSize,Rotation,HoleSize,Layer', 'limit': 0})['objects']
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
    from shapely.geometry import Point
    r = float(a['Radius']); w = float(a['Width'])
    obs.append(Point(float(a['XCenter']), float(a['YCenter'])).buffer(r + w / 2 + GAP))


def text_boxes():
    t = call('obj_query', {'object_type': 'eTextObject', 'properties': 'Text,BoundsLeft,BoundsBottom,BoundsRight,BoundsTop,Layer', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']
    return [(x['Text'], box(float(x['BoundsLeft']), float(x['BoundsBottom']), float(x['BoundsRight']), float(x['BoundsTop']))) for x in t]


def clean(g, own):
    b = g.bounds
    if b[0] < EDGE or b[1] < EDGE or b[2] > 4000 - EDGE or b[3] > 3000 - EDGE:
        return 'edge'
    for o in obs:
        if g.intersects(o):
            return 'pad/silk'
    for name, tb in text_boxes():
        if name == own:
            continue
        if g.intersects(tb.buffer(GAP)):
            return 'text ' + name
    return ''


LABELS = [
    # text, height, [(x, y, rot, just)...]
    ("SiPAP Rev C v3 2026-11", 50, [(2620, 1430, 0, 'center'), (2650, 1300, 0, 'center'), (2620, 1550, 0, 'center')]),
    ("12V IN", H, [(430, 2650, 0, 'center'), (300, 2660, 0, 'center'), (520, 2620, 0, 'center')]),
    ("BLOWER", H, [(350, 330, 0, 'center'), (350, 660, 0, 'center'), (200, 330, 0, 'center')]),
    ("12V", H, [(1272, 2205, 0, 'center'), (1272, 2335, 0, 'center'), (1200, 2270, 0, 'center_right')]),
    ("5V", H, [(1170, 1195, 0, 'center'), (1170, 1325, 0, 'center'), (1240, 1260, 0, 'center_left')]),
    ("3V3", H, [(1400, 2125, 0, 'center'), (1475, 2190, 0, 'center_left'), (1400, 2255, 0, 'center')]),
    ("GND", H, [(450, 1265, 0, 'center'), (450, 1395, 0, 'center'), (525, 1330, 0, 'center_left')]),
    ("GND", H, [(3700, 2650, 0, 'center_right'), (3690, 2650, 0, 'center_right')]),
    ("3V3", H, [(3700, 2550, 0, 'center_right'), (3690, 2550, 0, 'center_right')]),
    ("TX", H, [(3700, 2450, 0, 'center_right'), (3690, 2450, 0, 'center_right')]),
    ("RX", H, [(3700, 2350, 0, 'center_right'), (3690, 2350, 0, 'center_right')]),
    ("EN", H, [(3700, 2250, 0, 'center_right'), (3690, 2250, 0, 'center_right')]),
    ("IO0", H, [(3700, 2150, 0, 'center_right'), (3690, 2150, 0, 'center_right')]),
    ("UART", H, [(3800, 2050, 0, 'center'), (3800, 2040, 0, 'center')]),
]

placed, failed = [], []
for text, h, cands in LABELS:
    ok = False
    for (x, y, rot, just) in cands:
        r = call('pcb_place_text', {'text': text, 'x': x, 'y': y, 'layer': 'TopOverlay', 'height': h, 'rotation': rot,
                                    'stroke': W, 'kind': 'stroke', 'justification': just})
        bl = r.get('bounds') or []
        b = {'x1': bl[0], 'y1': bl[1], 'x2': bl[2], 'y2': bl[3]} if len(bl) == 4 else {}
        g = box(b['x1'], b['y1'], b['x2'], b['y2']) if b else None
        why = clean(g, text) if g is not None else 'no bounds'
        if not why:
            placed.append((text, x, y, just, b)); ok = True
            break
        # remove the try: match by exact text and location
        q = call('obj_query', {'object_type': 'eTextObject', 'properties': 'Text,X,Y,BoundsLeft,BoundsBottom', 'filter': 'Layer=TopOverlay|Text=' + text, 'limit': 0})['objects']
        for t in q:
            if abs(float(t['BoundsLeft']) - b['x1']) < 0.6 and abs(float(t['BoundsBottom']) - b['y1']) < 0.6:
                call('obj_delete', {'object_type': 'eTextObject', 'filter': 'Layer=TopOverlay|Text=%s|X=%s|Y=%s' % (text, t['X'], t['Y'])})
        print('  reject', text, (x, y), why)
    if not ok:
        failed.append(text)
print('placed', len(placed))
for p in placed:
    print(' ', p[0], p[1:4])
print('failed', failed)
json.dump({'placed': placed, 'failed': failed}, open('silk_labels_result.json', 'w'), indent=1)
