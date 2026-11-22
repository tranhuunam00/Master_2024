"""Readable designators: no upside-down text (180 -> 0, 270 -> 90, same centre), nothing within
20 mil of the board edge, module/sensor/encoder designators moved off their bodies so they stay
visible after assembly, and the 5V test-point label put right under TP2."""
import json, math
from shapely.geometry import box, LineString, Point
from edacall import call, BOARD

GAP, EDGE = 6.5, 20
call('app_set_active_document', {'file_path': BOARD})


def texts():
    return call('obj_query', {'object_type': 'eTextObject', 'properties': 'Text,X,Y,Rotation,IsDesignator,IsComment,BoundsLeft,BoundsBottom,BoundsRight,BoundsTop',
                              'filter': 'Layer=TopOverlay', 'limit': 0})['objects']


def bb(t):
    return [float(t[k]) for k in ('BoundsLeft', 'BoundsBottom', 'BoundsRight', 'BoundsTop')]


def setxy(filt, x, y, extra=''):
    s = 'X=%.4f|Y=%.4f' % (x, y) + (('|' + extra) if extra else '')
    r = call('obj_modify', {'items': [{'object_type': 'eTextObject', 'filter': filt, 'set': s}]})
    assert r['total_matched'] == 1, (filt, r)


# 1) flip upside-down designators about their centre
flipped = []
for t in texts():
    if t['IsComment'] in ('true', 'True') or t['Rotation'] not in ('180', '270'):
        continue
    if t['IsDesignator'] not in ('true', 'True'):
        continue
    b0 = bb(t); c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
    filt = 'Layer=TopOverlay|IsDesignator=true|Text=' + t['Text']
    newrot = '0' if t['Rotation'] == '180' else '90'
    call('obj_modify', {'items': [{'object_type': 'eTextObject', 'filter': filt, 'set': 'Rotation=' + newrot}]})
    t1 = [x for x in texts() if x['Text'] == t['Text'] and x['IsDesignator'] in ('true', 'True')][0]
    b1 = bb(t1); c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    setxy(filt, float(t1['X']) + c0[0] - c1[0], float(t1['Y']) + c0[1] - c1[1])
    flipped.append(t['Text'])
print('flipped', len(flipped), flipped)

# 2) obstacles for moving texts
pads = call('obj_query', {'object_type': 'ePadObject', 'properties': 'X,Y,TopXSize,TopYSize,Rotation', 'limit': 0})['objects']
silk = call('obj_query', {'object_type': 'eTrackObject', 'properties': 'X1,Y1,X2,Y2,Width', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']
arcs = call('obj_query', {'object_type': 'eArcObject', 'properties': 'XCenter,YCenter,Radius,Width', 'filter': 'Layer=TopOverlay', 'limit': 0})['objects']
comps = {c['designator']: c for c in call('pcb_get_components', {})['components']}
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
BODY = {}
for d in ('U1', 'U2', 'U3', 'U6', 'J1', 'F1', 'C25', 'BZ1', 'L1', 'L2'):
    b = comps[d]['bbox_no_name_comment']
    BODY[d] = box(b['x1'], b['y1'], b['x2'], b['y2'])


TCACHE = None


def free(g, own, avoid=()):
    b = g.bounds
    if b[0] < EDGE or b[1] < EDGE or b[2] > 4000 - EDGE or b[3] > 3000 - EDGE:
        return False
    if any(g.intersects(o) for o in obs) or any(g.intersects(o) for o in BODY.values()) or any(g.intersects(o) for o in avoid):
        return False
    for name, tb in TCACHE:
        if g.intersects(tb):
            return False
    return True


def move_near(text, anchor, rmax, desig=True, avoid=()):
    tt = [x for x in texts() if x['Text'] == text and (x['IsDesignator'] in ('true', 'True')) == desig]
    assert len(tt) == 1, (text, len(tt))
    t = tt[0]; b = bb(t); w, h = b[2] - b[0], b[3] - b[1]; c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    global TCACHE
    TCACHE = [(x['Text'], box(*bb(x)).buffer(GAP)) for x in texts() if x['IsComment'] not in ('true', 'True')
              and not (x['Text'] == text and x['X'] == t['X'] and x['Y'] == t['Y'])]
    ax, ay = anchor
    for r in range(0, rmax + 1, 5):
        for a in range(0, 360, 10 if r else 360):
            x = ax + r * math.cos(math.radians(a)); y = ay + r * math.sin(math.radians(a))
            g = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if free(g, text, avoid):
                filt = 'Layer=TopOverlay|Text=%s|IsDesignator=%s' % (text, 'true' if desig else 'false')
                if not desig:
                    filt += '|X=%s|Y=%s' % (t['X'], t['Y'])
                setxy(filt, float(t['X']) + x - c[0], float(t['Y']) + y - c[1])
                print('moved', text, 'centre', (round(c[0]), round(c[1])), '->', (round(x), round(y)))
                return True
    print('NO SPOT for', text)
    return False


# designators hidden under bodies or at the edge
for d in ('U1', 'U2', 'U3', 'U6'):
    b = comps[d]['bbox_no_name_comment']
    # prefer the spot left of the body, mid-height (reads with the part)
    move_near(d, (b['x1'] - 45, (b['y1'] + b['y2']) / 2), 400)
move_near('J1', (120, 2150), 250)
# 5V label straight under TP2 (1170, 1260); move R15's designator out of the way first
tp2_zone = box(1110, 1150, 1230, 1232)
move_near('R15', (1295, 1215), 250, avoid=(tp2_zone,))
move_near('5V', (1170, 1190), 120, desig=False)
