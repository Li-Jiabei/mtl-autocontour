#!/usr/bin/env python3
"""
finalize_cuts.py -- clean, organized ASCII-VTK output from registered contours.

For every <name>.byu in SURF_DIR (searched recursively) that has a matching
<name>_contour_ids.txt in CONTOUR_DIR, it writes THREE ASCII .vtk files into

    OUTPUT_DIR/<subject>/
        <name>.vtk          (the original surface, byu -> ascii vtk)
        <name>_top.vtk      (top region)
        <name>_bottom.vtk   (bottom region)

where <subject> is the part of <name> before the first underscore
(e.g. BEIALE_150428_7  ->  subject "BEIALE"). All timepoints of one subject
land in that subject's folder.

- Output is VTK ASCII (not binary).
- Intermediates (contour ids, logs, lddmm dirs) are NOT touched — OUTPUT_DIR
  contains only the 3 vtk files per surface, nothing else.
- Surfaces with holes / non-manifold edges are skipped and logged.

Usage:
    python3 finalize_cuts.py SURF_DIR CONTOUR_DIR OUTPUT_DIR
Example (one group/hemisphere at a time so filenames don't collide):
    python3 finalize_cuts.py byu_tests/CON_LH auto_contours final_vtk
"""
import sys, os, glob, heapq
import numpy as np
from collections import defaultdict, deque, Counter


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


def write_vtk_ascii(path, V, F):
    with open(path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(os.path.basename(path) + "\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS %d float\n" % len(V))
        for p in V:
            f.write("%f %f %f\n" % (p[0], p[1], p[2]))
        f.write("POLYGONS %d %d\n" % (len(F), len(F) * 4))
        for tri in F:
            f.write("3 %d %d %d\n" % (tri[0], tri[1], tri[2]))


def health_problem(F):
    ec = Counter()
    for tri in F:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            ec[(min(a, b), max(a, b))] += 1
    if any(c > 2 for c in ec.values()):
        return "non-manifold edges"
    if any(c == 1 for c in ec.values()):
        return "open boundary / hole"
    return None


def build_adj(V, F):
    adj = defaultdict(dict)
    for tri in F:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            d = float(np.linalg.norm(V[a] - V[b])); adj[a][b] = d; adj[b][a] = d
    return adj


def dij(adj, s, t):
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


def process(byu, ids_path, outdir):
    name = os.path.splitext(os.path.basename(byu))[0]
    subject = name.split('_')[0]
    V, F = read_byu(byu)
    prob = health_problem(F)
    if prob:
        print("SKIP %s: %s" % (name, prob)); return False
    ids = np.loadtxt(ids_path, dtype=int).ravel()
    adj = build_adj(V, F)
    barrier = set()
    for i in range(len(ids)):
        barrier.update(dij(adj, int(ids[i]), int(ids[(i + 1) % len(ids)])))

    def flood(seed):
        seen = {seed}; q = deque([seed])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in barrier and v not in seen:
                    seen.add(v); q.append(v)
        return seen

    nonb = [v for v in range(len(V)) if v not in barrier]
    A = flood(nonb[0]); rem = set(nonb) - A
    if not rem:
        print("SKIP %s: contour did not separate the surface" % name); return False
    B = flood(next(iter(rem)))
    lab = np.full(len(V), 2)
    for v in A: lab[v] = 0
    for v in B: lab[v] = 1

    def submesh(region):
        keep = [fi for fi, tri in enumerate(F)
                if any(lab[v] == region for v in tri) and all(lab[v] != (1 - region) for v in tri)]
        vids = sorted({v for fi in keep for v in F[fi]})
        rm = {v: i for i, v in enumerate(vids)}
        return V[vids], np.array([[rm[v] for v in F[fi]] for fi in keep])

    # degenerate-cut guard. Real hand cuts have smaller side 35-50% (median ~43%,
    # measured over 198 manual cuts), so anything below ~35% is a failed registration.
    SUSPECT_MIN_FRACTION = 0.35
    small = min(len(A), len(B)) / float(len(A) + len(B))

    # Deterministic top/bottom by GEOMETRY (flood-fill order is arbitrary).
    # The region whose centroid is higher along the axis where the two regions
    # differ most is called "top". If your convention is opposite, set False.
    TOP_IS_HIGHER = True
    cA = V[lab == 0].mean(0); cB = V[lab == 1].mean(0)
    ax = int(np.argmax(np.abs(cA - cB)))
    a_is_top = (cA[ax] >= cB[ax]) if TOP_IS_HIGHER else (cA[ax] < cB[ax])
    top_region, bot_region = (0, 1) if a_is_top else (1, 0)
    subdir = os.path.join(outdir, subject)
    os.makedirs(subdir, exist_ok=True)
    write_vtk_ascii(os.path.join(subdir, name + ".vtk"), V, F)          # original
    Vt, Ft = submesh(top_region); write_vtk_ascii(os.path.join(subdir, name + "_top.vtk"), Vt, Ft)
    Vb, Fb = submesh(bot_region); write_vtk_ascii(os.path.join(subdir, name + "_bottom.vtk"), Vb, Fb)
    if small < SUSPECT_MIN_FRACTION:
        print("SUSPECT %s/%s  lopsided cut (smaller side %.0f%% < %.0f%%) -- likely bad registration"
              % (subject, name, small * 100, SUSPECT_MIN_FRACTION * 100))
        return "suspect"
    print("OK   %s/%s  (top %d / bottom %d faces, smaller side %.0f%%)"
          % (subject, name, len(Ft), len(Fb), small * 100))
    return True


def main():
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(1)
    surf_dir, cont_dir, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    n_ok = n_skip = 0; suspects = []
    for byu in sorted(glob.glob(os.path.join(surf_dir, '**', '*.byu'), recursive=True)):
        name = os.path.splitext(os.path.basename(byu))[0]
        ids = os.path.join(cont_dir, name + '_contour_ids.txt')
        if not os.path.isfile(ids):
            print("SKIP %s: no contour ids" % name); n_skip += 1; continue
        r = process(byu, ids, outdir)
        if r == "suspect":
            n_ok += 1; suspects.append(name)
        elif r:
            n_ok += 1
        else:
            n_skip += 1
    print("\nDone: %d surfaces -> %s/<subject>/  | %d skipped" % (n_ok, outdir, n_skip))
    if suspects:
        print("\n%d SUSPECT (lopsided) cuts to review/redo:" % len(suspects))
        for s in suspects:
            print("   " + s)


if __name__ == "__main__":
    main()
