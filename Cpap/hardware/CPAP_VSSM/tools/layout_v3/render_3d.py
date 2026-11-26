# 3D views of the board: component bodies from the Altium STEP (planar B-rep), board slab
# textured with the Gerber layers. Software z-buffer, orthographic camera.
# Usage: python render_3d.py <board.step> <gerber_dir> <gerber_basename> <out_dir>
import os, re, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import step_planar, render_sides

THICK = 1.6     # FAB_NOTES: 1.6 mm FR-4
SS = 2          # supersampling
TEX_PPM = 32.0

METAL, BLACK, GOLD = (190, 190, 195), (40, 40, 44), (218, 178, 72)


def part_color(name, sub, zmin, zmax):
    p = re.match(r'[A-Z]+', name).group(0)
    if name == 'U1': return (205, 207, 212) if zmin > 0.5 else (32, 32, 36)          # shield / module PCB
    if name == 'U6': return [(205, 205, 210), (165, 165, 170), (62, 62, 66)][min(sub, 2)]  # shaft / bushing / body
    if name in ('U4', 'U5', 'J3', 'J2'): return GOLD if zmin >= 2.4 else BLACK           # pins / header plastic
    if name in ('U2', 'U3'): return [(62, 62, 68), (226, 226, 220), METAL][min(sub, 2)]  # body / port / leads
    if name == 'RV1': return (70, 70, 76) if sub == 0 else METAL
    if p in ('Q', 'U', 'D') and zmax <= 0.45: return METAL                            # SO-8 lead block
    if name == 'F1': return (212, 158, 42)
    if name == 'C25': return (44, 52, 84)
    if name in ('BZ1', 'J1'): return (30, 30, 33)
    if p == 'L': return (72, 72, 78)
    if p == 'C': return (196, 162, 112)
    if p == 'R': return (30, 30, 32)
    return BLACK


def newell(P):
    n = np.zeros(3)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    L = np.linalg.norm(n)
    return n / L if L > 0 else np.array([0, 0, 1.])


def triangulate(P3):
    P3 = [p for i, p in enumerate(P3) if i == 0 or np.linalg.norm(p - P3[i - 1]) > 1e-9]
    if len(P3) > 1 and np.linalg.norm(P3[0] - P3[-1]) < 1e-9: P3 = P3[:-1]
    P3 = np.array(P3)
    if len(P3) < 3: return P3, []
    n = newell(P3); ax = int(np.argmax(abs(n))); k = [i for i in range(3) if i != ax]
    P = P3[:, k]
    area = sum(P[i, 0] * P[(i + 1) % len(P), 1] - P[(i + 1) % len(P), 0] * P[i, 1] for i in range(len(P)))
    V = list(range(len(P))) if area > 0 else list(range(len(P)))[::-1]
    cross = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    tris, guard = [], 0
    while len(V) > 3 and guard < 10000:
        guard += 1; ear = False
        for i in range(len(V)):
            a, b, c = V[i - 1], V[i], V[(i + 1) % len(V)]
            if cross(P[a], P[b], P[c]) <= 1e-12: continue
            if any(cross(P[a], P[b], P[q]) >= 0 and cross(P[b], P[c], P[q]) >= 0 and cross(P[c], P[a], P[q]) >= 0
                   for q in V if q not in (a, b, c)):
                continue
            tris.append((a, b, c)); V.pop(i); ear = True; break
        if not ear: break
    if len(V) == 3: tris.append(tuple(V))
    elif len(V) > 3: tris += [(V[0], V[i], V[i + 1]) for i in range(1, len(V) - 1)]
    return P3, tris


class ZBuffer:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.z = np.full((H, W), np.inf, np.float32)
        self.rgb = np.zeros((H, W, 3), np.float32)
        self.fid = np.full((H, W), -1, np.int32)

    def tri(self, s, w, fid, color, tex=None, shade=1.0):
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = s
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9: return
        xa, xb = max(0, int(math.floor(min(x0, x1, x2)))), min(self.W - 1, int(math.ceil(max(x0, x1, x2))))
        ya, yb = max(0, int(math.floor(min(y0, y1, y2)))), min(self.H - 1, int(math.ceil(max(y0, y1, y2))))
        if xa > xb or ya > yb: return
        xs = np.arange(xa, xb + 1, dtype=np.float32) + 0.5
        for r0 in range(ya, yb + 1, 256):
            r1 = min(yb + 1, r0 + 256)
            ys = np.arange(r0, r1, dtype=np.float32)[:, None] + 0.5
            l0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / den
            l1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / den
            l2 = 1 - l0 - l1
            e = -1e-4
            inside = (l0 >= e) & (l1 >= e) & (l2 >= e)
            if not inside.any(): continue
            z = l0 * z0 + l1 * z1 + l2 * z2
            zb = self.z[r0:r1, xa:xb + 1]
            m = inside & (z < zb)
            if not m.any(): continue
            zb[m] = z[m]
            self.fid[r0:r1, xa:xb + 1][m] = fid
            if tex is None:
                self.rgb[r0:r1, xa:xb + 1][m] = np.array(color, np.float32) * shade
            else:
                img, fx = tex
                wx = (l0 * w[0][0] + l1 * w[1][0] + l2 * w[2][0])[m]
                wy = (l0 * w[0][1] + l1 * w[1][1] + l2 * w[2][1])[m]
                u, v = fx(wx, wy)
                u = np.clip(u.astype(int), 0, img.shape[1] - 1); v = np.clip(v.astype(int), 0, img.shape[0] - 1)
                self.rgb[r0:r1, xa:xb + 1][m] = img[v, u].astype(np.float32) * shade


def render(faces, cam_from, up, size, title, out):
    """faces: list of (poly Nx3, color, tex or None, fid)."""
    f = -np.asarray(cam_from, float); f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    light = 0.55 * (-f) + 0.45 * u + 0.25 * r; light /= np.linalg.norm(light)
    allp = np.vstack([p for p, *_ in faces])
    sx, sy = allp @ r, -(allp @ u)
    W, H = size[0] * SS, size[1] * SS
    pad = 0.07
    scale = min(W * (1 - 2 * pad) / (sx.max() - sx.min()), H * (1 - 2 * pad) / (sy.max() - sy.min()))
    ox = W / 2 - scale * (sx.max() + sx.min()) / 2; oy = H / 2 - scale * (sy.max() + sy.min()) / 2 + 0.02 * H
    zb = ZBuffer(W, H)
    for P, color, tex, fid in faces:
        n = newell(P)
        if np.dot(n, -f) < 0: n = -n
        shade = 0.42 + 0.58 * max(0.0, float(np.dot(n, light)))
        P3, tris = triangulate(list(P))
        S = np.stack([P3 @ r * scale + ox, -(P3 @ u) * scale + oy, P3 @ f], 1)
        for a, b, c in tris:
            zb.tri((S[a], S[b], S[c]), (P3[a], P3[b], P3[c]), fid, color, tex, shade)
    # background gradient
    g = np.linspace(238, 198, H, dtype=np.float32)[:, None, None]
    bg = np.repeat(np.repeat(g, W, 1), 3, 2) * np.array([1.0, 1.0, 1.02], np.float32)
    img = np.where((zb.fid >= 0)[..., None], zb.rgb, bg)
    # outline edges between different faces
    fid = zb.fid
    e = np.zeros_like(fid, bool)
    e[:, 1:] |= fid[:, 1:] != fid[:, :-1]; e[:, :-1] |= fid[:, 1:] != fid[:, :-1]
    e[1:, :] |= fid[1:, :] != fid[:-1, :]; e[:-1, :] |= fid[1:, :] != fid[:-1, :]
    img[e] *= 0.45
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype('arialbd.ttf', 28)
    except Exception: font = None
    d.text((20, 14), title, fill=(40, 40, 46), font=font)
    im.save(out); print(out)


def main():
    step, gdir, base, out = sys.argv[1:5]
    parts = step_planar.build(step)
    bd = render_sides.Board(gdir, base, TEX_PPM, M=0.0)
    top, bot = bd.side_images()
    x0, y0, x1, y1 = bd.bx0, bd.by0, bd.bx1, bd.by1
    sxp = top.shape[1] / (x1 - x0); syp = top.shape[0] / (y1 - y0)
    fx = lambda wx, wy: ((wx - x0) * sxp, (y1 - wy) * syp)
    faces, fid = [], 0
    zt, zbm = 0.0, -THICK
    c = lambda x, y, z: np.array([x, y, z], float)
    faces.append((np.array([c(x0, y0, zt), c(x1, y0, zt), c(x1, y1, zt), c(x0, y1, zt)]), None, (top, fx), fid)); fid += 1
    faces.append((np.array([c(x0, y0, zbm), c(x1, y0, zbm), c(x1, y1, zbm), c(x0, y1, zbm)]), None, (bot, fx), fid)); fid += 1
    edge = (178, 168, 118)
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        faces.append((np.array([c(ax, ay, zbm), c(bx, by, zbm), c(bx, by, zt), c(ax, ay, zt)]), edge, None, fid)); fid += 1
    n_parts = 0
    for name, path, polys in parts:
        if name in (None, 'Board'): continue
        pts = np.vstack([l for fl in polys for l in fl])
        col = part_color(name, path[1] if len(path) > 1 else 0, pts[:, 2].min(), pts[:, 2].max())
        n_parts += 1
        for loops in polys:
            faces.append((loops[0], col, None, fid)); fid += 1
    print('bodies', n_parts, 'faces', len(faces))
    os.makedirs(out, exist_ok=True)
    size = (2400, 1600)
    render(faces, (-0.55, -1.0, 0.95), (0, 0, 1), size, 'SiPAP Rev C v3 - 3D, mặt trên', os.path.join(out, 'SiPAP_v3_3D_top_front.png'))
    render(faces, (0.9, 0.75, 0.85), (0, 0, 1), size, 'SiPAP Rev C v3 - 3D, mặt trên (góc sau-phải)', os.path.join(out, 'SiPAP_v3_3D_top_back.png'))
    render(faces, (-0.55, -1.0, -0.95), (0, 0, -1), size, 'SiPAP Rev C v3 - 3D, mặt dưới', os.path.join(out, 'SiPAP_v3_3D_bottom.png'))


if __name__ == '__main__':
    main()
