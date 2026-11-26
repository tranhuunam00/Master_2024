# Minimal STEP AP214 reader for planar B-rep assemblies (Altium/OCC export): returns per-part world polygons.
import re
import numpy as np


def split_args(s):
    out, depth, cur, inq = [], 0, '', False
    for ch in s:
        if inq:
            cur += ch
            if ch == "'": inq = False
            continue
        if ch == "'": inq = True; cur += ch; continue
        if ch == '(':
            depth += 1
            if depth == 1: continue
        elif ch == ')':
            depth -= 1
            if depth == 0:
                out.append(cur.strip()); cur = ''; continue
        if ch == ',' and depth == 1:
            out.append(cur.strip()); cur = ''; continue
        if depth >= 1: cur += ch
    return out


def parse_list(s):
    s = s.strip()
    assert s.startswith('(') and s.endswith(')'), s
    return [a for a in split_args(s)]


def load(path):
    txt = open(path, encoding='latin-1').read()
    data = txt.split('DATA;', 1)[1].split('ENDSEC;', 1)[0]
    E = {}
    for m in re.finditer(r'#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|$)', data, re.S):
        eid = int(m.group(1)); body = ' '.join(m.group(2).split())
        if body.startswith('('):  # complex entity
            parts = {}
            for mm in re.finditer(r'([A-Z_0-9]+)\s*(\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\))', body[1:-1]):
                parts[mm.group(1)] = split_args(mm.group(2))
            E[eid] = ('COMPLEX', parts)
        else:
            mm = re.match(r'([A-Z_0-9]+)\s*(\(.*\))$', body)
            E[eid] = (mm.group(1), split_args(mm.group(2)))
    return E


def ref(s): return int(s.strip().lstrip('#'))


def build(path):
    E = load(path)
    pt = lambda i: np.array([float(v) for v in parse_list(E[i][1][1])])
    def dirv(i): return np.array([float(v) for v in parse_list(E[i][1][1])])
    def axis(i):
        a = E[i][1]
        o = pt(ref(a[1])); z = dirv(ref(a[2])) if a[2] != '$' else np.array([0, 0, 1.]); x = dirv(ref(a[3])) if a[3] != '$' else np.array([1, 0, 0.])
        z = z / np.linalg.norm(z); x = x - z * np.dot(x, z); x = x / np.linalg.norm(x); y = np.cross(z, x)
        M = np.eye(4); M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z; M[:3, 3] = o
        return M
    # face polygons per brep
    def vert(i): return pt(ref(E[i][1][1]))
    def face_poly(fid):
        a = E[fid][1]; bounds = [ref(b) for b in parse_list(a[1])]; same = a[3].strip() == '.T.'
        loops = []
        for b in bounds:
            ba = E[b][1]; loop = ref(ba[1]); ori = ba[2].strip() == '.T.'
            pts = []
            for oe in parse_list(E[loop][1][1]):
                oa = E[ref(oe)][1]; ec = ref(oa[3]); fwd = oa[4].strip() == '.T.'
                ea = E[ec][1]; v1, v2 = ref(ea[1]), ref(ea[2])
                pts.append(vert(v1) if fwd else vert(v2))
            if not ori: pts = pts[::-1]
            loops.append(np.array(pts))
        return loops, same
    breps = {}
    for i, (t, a) in E.items():
        if t == 'MANIFOLD_SOLID_BREP':
            shell = ref(a[1]); faces = [ref(f) for f in parse_list(E[shell][1][1])]
            breps[i] = [face_poly(f) for f in faces]
    # representation -> breps
    rep_items = {}
    for i, (t, a) in E.items():
        if t in ('ADVANCED_BREP_SHAPE_REPRESENTATION', 'SHAPE_REPRESENTATION'):
            rep_items[i] = [ref(x) for x in parse_list(a[1])]
    # product definitions
    pd_name = {}
    for i, (t, a) in E.items():
        if t == 'PRODUCT_DEFINITION':
            pdf = ref(a[2]); prod = ref(E[pdf][1][2]); pd_name[i] = E[prod][1][0].strip("'")
    pds_target = {}
    for i, (t, a) in E.items():
        if t == 'PRODUCT_DEFINITION_SHAPE': pds_target[i] = ref(a[2])
    pd_rep = {}
    for i, (t, a) in E.items():
        if t == 'SHAPE_DEFINITION_REPRESENTATION':
            pds = ref(a[0]); tgt = pds_target[pds]
            if tgt in pd_name: pd_rep[tgt] = ref(a[1])
    # links
    rep_pd = {v: k for k, v in pd_rep.items()}
    children = {}
    for i, (t, a) in E.items():
        if t == 'CONTEXT_DEPENDENT_SHAPE_REPRESENTATION':
            rel = E[ref(a[0])][1]; nauo = pds_target[ref(a[1])]
            na = E[nauo][1]; parent, child = ref(na[3]), ref(na[4])
            rr = rel['REPRESENTATION_RELATIONSHIP']; r1, r2 = ref(rr[2]), ref(rr[3])
            idt = E[ref(rel['REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION'][0])][1]
            A1, A2 = axis(ref(idt[2])), axis(ref(idt[3]))
            # child rep must be r1 or r2
            crep = pd_rep[child]
            if crep == r1: T = A2 @ np.linalg.inv(A1)
            else: T = A1 @ np.linalg.inv(A2)
            children.setdefault(parent, []).append((child, T))
    allchild = {c for v in children.values() for c, _ in v}
    roots = [p for p in pd_name if p not in allchild]
    parts = []  # (top-level name, sub index, list of world polygons)
    def walk(pd, M, top, path):
        rep = pd_rep.get(pd)
        if rep is not None:
            for it in rep_items.get(rep, []):
                if it in breps:
                    polys = []
                    for loops, same in breps[it]:
                        L = [(M[:3, :3] @ l.T).T + M[:3, 3] for l in loops]
                        polys.append(L)
                    parts.append((top, path, polys))
        for k, (c, T) in enumerate(children.get(pd, [])):
            nm = pd_name[c]
            walk(c, M @ T, nm if top is None else top, path + (k,))
    for r in roots: walk(r, np.eye(4), None, ())
    return parts
