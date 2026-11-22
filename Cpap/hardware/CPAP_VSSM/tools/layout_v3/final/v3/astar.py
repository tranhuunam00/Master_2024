"""numba A* on a 2-layer octilinear grid with direction + run-length state (only 45-degree turns, min run between turns)."""
import numpy as np
from numba import njit

DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], np.int64)
DY = np.array([0, 1, 1, 1, 0, -1, -1, -1], np.int64)
RUNS = 3   # run state 0,1,2 (2 = may turn)

@njit(cache=True)
def _push(hk, hv, n, key, val):
    i = n
    hk[i] = key; hv[i] = val
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]; hv[p], hv[i] = hv[i], hv[p]
        i = p
    return n + 1

@njit(cache=True)
def _pop(hk, hv, n):
    key = hk[0]; val = hv[0]
    n -= 1
    hk[0] = hk[n]; hv[0] = hv[n]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= n:
            break
        r = l + 1
        c = l
        if r < n and hk[r] < hk[l]:
            c = r
        if hk[i] <= hk[c]:
            break
        hk[c], hk[i] = hk[i], hk[c]; hv[c], hv[i] = hv[i], hv[c]
        i = c
    return key, val, n

@njit(cache=True)
def astar(blk, vok, cost, src, tgt, tx, ty, G, lcost, t45, via_cost, maxexp, start_any):
    """blk[L,j,i] bool blocked; vok[j,i] via allowed; cost[L,j,i] extra per-cell cost factor (>=1);
    src: (k,3) array of (L,j,i) sources; tgt[L,j,i] bool targets; (tx,ty) target centre in cells for heuristic.
    Returns path array (m,3) of (L,j,i) or empty."""
    NL, NY, NX = blk.shape
    D = 9 * RUNS
    N = NL * NY * NX * D
    INF = 1e18
    g = np.full(N, INF)
    par = np.full(N, -1, np.int64)
    cap = 4000000
    hk = np.empty(cap); hv = np.empty(cap, np.int64)
    n = 0
    sq2 = np.sqrt(2.0)
    for k in range(src.shape[0]):
        L = src[k, 0]; j = src[k, 1]; i = src[k, 2]
        s = (((L * NY + j) * NX + i) * 9 + 8) * RUNS + (RUNS - 1)
        g[s] = 0.0
        dx = abs(i - tx); dy = abs(j - ty)
        h = (max(dx, dy) - min(dx, dy) + sq2 * min(dx, dy)) * G * lcost[0]
        n = _push(hk, hv, n, h, s)
    goal = -1
    exp = 0
    while n > 0:
        f, s, n = _pop(hk, hv, n)
        r = s % RUNS; t = s // RUNS
        d = t % 9; t //= 9
        i = t % NX; t //= NX
        j = t % NY; L = t // NY
        gs = g[s]
        dxh = abs(i - tx); dyh = abs(j - ty)
        if f - (max(dxh, dyh) - min(dxh, dyh) + sq2 * min(dxh, dyh)) * G * lcost[0] > gs + 1e-6:
            continue
        if tgt[L, j, i]:
            goal = s
            break
        exp += 1
        if exp > maxexp:
            break
        for nd in range(8):
            if d != 8:
                diff = abs(nd - d)
                if diff > 4:
                    diff = 8 - diff
                if diff > 1:
                    continue
                if diff == 1 and r < RUNS - 1:
                    continue
                tc = t45 if diff == 1 else 0.0
                nr = 0 if diff == 1 else min(r + 1, RUNS - 1)
            else:
                if DX[nd] != 0 and DY[nd] != 0:
                    continue          # leave the tree orthogonally (no acute T-junctions)
                tc = 0.0
                nr = 0
            ni = i + DX[nd]; nj = j + DY[nd]
            if ni < 0 or nj < 0 or ni >= NX or nj >= NY:
                continue
            if blk[L, nj, ni] and not tgt[L, nj, ni]:
                continue
            if DX[nd] != 0 and DY[nd] != 0:
                # no corner cutting through blocked cells
                if blk[L, j, ni] and blk[L, nj, i]:
                    continue
                step = sq2 * G * 1.65
            else:
                step = G
            ns = (((L * NY + nj) * NX + ni) * 9 + nd) * RUNS + nr
            ng = gs + step * lcost[L] * cost[L, nj, ni] + tc
            if ng < g[ns]:
                g[ns] = ng; par[ns] = s
                dx = abs(ni - tx); dy = abs(nj - ty)
                hh = (max(dx, dy) - min(dx, dy) + sq2 * min(dx, dy)) * G * lcost[0]
                if n >= cap:
                    return np.zeros((0, 3), np.int64), exp
                n = _push(hk, hv, n, ng + hh, ns)
        # via: change layer, keep direction, needs run state settled
        if d != 8 and r == RUNS - 1 and vok[j, i]:
            L2 = 1 - L
            if not blk[L2, j, i] or tgt[L2, j, i]:
                ns = (((L2 * NY + j) * NX + i) * 9 + d) * RUNS + 0
                ng = gs + via_cost
                if ng < g[ns]:
                    g[ns] = ng; par[ns] = s
                    n = _push(hk, hv, n, ng + (max(dxh, dyh) - min(dxh, dyh) + sq2 * min(dxh, dyh)) * G * lcost[0], ns)
    if goal < 0:
        return np.zeros((0, 3), np.int64), exp
    cnt = 0
    s = goal
    while s >= 0:
        cnt += 1; s = par[s]
    out = np.zeros((cnt, 3), np.int64)
    s = goal; k = cnt - 1
    while s >= 0:
        t = s // RUNS // 9
        i = t % NX; t //= NX
        j = t % NY; L = t // NY
        out[k, 0] = L; out[k, 1] = j; out[k, 2] = i
        k -= 1; s = par[s]
    return out, exp
