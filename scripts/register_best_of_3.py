#!/usr/bin/env python3
"""
register_best_of_3.py -- register each surface to ALL group templates and keep
the one that yields the most balanced (most plausible) cut.

For each <name>.byu in INPUT_GLOB it registers every *_template.vtk in
TEMPLATE_DIR onto it, transfers each template's boundary, evaluates the
resulting cut's balance (smaller-side fraction), and writes the contour IDs of
the WINNING template to OUTPUT_DIR/<name>_contour_ids.txt. It also logs which
group won and the balance achieved.

This auto-handles "which shape group is this surface?" (the best-fitting
template wins) and rejects degenerate registrations (low balance).

Run on the cluster (env with py-lddmm). Cost ~ (#templates) x normal time
per surface, so ~3x slower than single-template.

Usage:
    python3 register_best_of_3.py TEMPLATE_DIR INPUT_GLOB OUTPUT_DIR
Example:
    python3 register_best_of_3.py templates \
        '/cis/home/jli401/contour_project/byu_tests/CON_LH/cut/*.byu' best_contours
"""
from sys import path as sys_path
sys_path.insert(0, '/cis/project/adni_thickness/registration_old/py-lddmm')
sys_path.insert(0, '/cis/project/adni_thickness/registration_old/py-lddmm/base')

import sys, os, glob, time, heapq
import numpy as np
from collections import defaultdict, deque, Counter
from scipy.spatial import cKDTree
from base.surfaces import Surface
from base.kernelFunctions import Kernel
from base.surfaceMatching import SurfaceMatching
from base.affineRegistration import rigidRegistration
from base import loggingUtils

MAXITER = 100
K1 = Kernel(name='laplacian', sigma=5.0)
K2 = Kernel(name='gauss', sigma=5.0)
MATCH_OPTIONS = {                 # same as the validated single-template run
    'mode': 'normal', 'timeStep': 0.1, 'KparDiff': K1, 'KparDist': K2,
    'internalCost': [['elastic', 50]], 'internalWeight': 1.,
    'sigmaError': .25, 'errorType': 'varifold', 'maxIter': MAXITER,
    'affine': 'none', 'rotWeight': 10., 'saveRate': 1000, 'transWeight': 1.,
    'algorithm': 'bfgs', 'scaleWeight': 10., 'affineWeight': 100.,
    'verb': False, 'pk_dtype': 'float32',
}


def read_byu(path):
    t = open(path).read().split(); it = iter(t)
    npart = int(next(it)); nv = int(next(it)); nf = int(next(it)); ne = int(next(it))
    for _ in range(npart):
        next(it); next(it)
    V = np.array([float(next(it)) for _ in range(nv * 3)]).reshape(nv, 3)
    conn = [int(next(it)) for _ in range(ne)]; F = []; cur = []
    for c in conn:
        if c < 0:
            cur.append(-c - 1); F.append(cur); cur = []
        else:
            cur.append(c - 1)
    return V, np.array(F, int)


def load_surface(path):
    try:
        return Surface(path)
    except Exception:
        V, F = read_byu(path)
        return Surface(surf=(F, V))


def ordered_boundary_loop(faces):
    ec = Counter()
    for tri in faces:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            ec[(min(a, b), max(a, b))] += 1
    be = [e for e, c in ec.items() if c == 1]
    if not be:
        return np.array([], dtype=int)
    adj = defaultdict(list)
    for a, b in be:
        adj[a].append(b); adj[b].append(a)
    start = be[0][0]; loop = [start]; prev = None; cur = start
    while True:
        nxts = [v for v in adj[cur] if v != prev]
        if not nxts or nxts[0] == start:
            break
        loop.append(nxts[0]); prev, cur = cur, nxts[0]
        if len(loop) > len(adj) + 5:
            break
    return np.array(loop)


def balance_of(V, F, ids):
    """Route contour ids on the mesh, flood-fill, return smaller-side fraction (0 if no split)."""
    adj = defaultdict(dict)
    for tri in F:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            d = float(np.linalg.norm(V[a] - V[b])); adj[a][b] = d; adj[b][a] = d

    def dij(s, t):
        dist = {s: 0.0}; prev = {}; pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == t:
                break
            if d > dist.get(u, 1e18):
                continue
            for v, w in adj[u].items():
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        p = [t]
        while p[-1] != s:
            if p[-1] not in prev:
                return [s, t]
            p.append(prev[p[-1]])
        return p[::-1]

    barrier = set()
    for i in range(len(ids)):
        barrier.update(dij(int(ids[i]), int(ids[(i + 1) % len(ids)])))

    def flood(seed):
        seen = {seed}; q = deque([seed])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in barrier and v not in seen:
                    seen.add(v); q.append(v)
        return seen
    nonb = [v for v in range(len(V)) if v not in barrier]
    if not nonb:
        return 0.0
    A = flood(nonb[0]); rem = set(nonb) - A
    if not rem:
        return 0.0
    B = flood(next(iter(rem)))
    return min(len(A), len(B)) / float(len(A) + len(B))


def main():
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(1)
    tdir, in_glob, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    loggingUtils.setup_default_logging(outdir, fileName='bestof_log.txt', stdOutput=True)

    tfiles = sorted(glob.glob(os.path.join(tdir, '*_template.vtk')))
    templates = []
    for tf in tfiles:
        t = load_surface(tf)
        loop = ordered_boundary_loop(t.faces)
        templates.append((os.path.basename(tf).replace('_template.vtk', ''), t, loop))
    print("loaded %d templates: %s" % (len(templates), [g for g, _, _ in templates]))

    files = sorted(glob.glob(in_glob))
    print("processing %d surfaces" % len(files))
    for ffile in files:
        name = os.path.splitext(os.path.basename(ffile))[0]
        t0 = time.time()
        Vorig, Forig = read_byu(ffile) if ffile.endswith('.byu') else (None, None)
        if Vorig is None:
            s_tmp = load_surface(ffile); Vorig = s_tmp.vertices.copy(); Forig = s_tmp.faces
        best = None
        for g, t, loop in templates:
            if len(loop) == 0:
                continue
            surf = load_surface(ffile)
            R0, T0 = rigidRegistration(surfaces=(surf.vertices, t.vertices),
                                       rotWeight=0., verb=False, temperature=10., annealing=True)
            surf.updateVertices(surf.vertices @ R0.T + T0)
            opt = dict(MATCH_OPTIONS); opt['outputDir'] = os.path.join(outdir, '_work', name, g)
            f = SurfaceMatching(Template=t, Target=surf, options=opt)
            f.optimizeMatching()
            pred = f.fvDef.vertices[loop]
            ids = cKDTree(surf.vertices).query(pred)[1]
            bal = balance_of(Vorig, Forig, ids)
            if best is None or bal > best[2]:
                best = (g, ids, bal)
        if best is None:
            print("SKIP %s: no usable template" % name); continue
        np.savetxt(os.path.join(outdir, name + '_contour_ids.txt'), best[1], fmt='%d')
        print("OK   %s  best=%s  balance=%.0f%%  (%.1f min)"
              % (name, best[0], best[2] * 100, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
