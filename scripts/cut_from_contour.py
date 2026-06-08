#!/usr/bin/env python3
"""
cut_from_contour.py  -- cut a surface into top/bottom from a transferred contour.
No py-lddmm needed; runs anywhere (numpy + scipy).

Given a .byu surface and its contour vertex-id list (the *_contour_ids.txt that
ContourMappings.py writes), it:
  1. routes the contour along the mesh edges (Dijkstra) into a CLOSED on-surface
     barrier (so a loose/floating contour is fixed onto the surface),
  2. flood-fills into two regions,
  3. writes <name>_top.byu and <name>_bottom.byu.

Includes a guard: skips surfaces with holes / non-manifold edges (logs them).

Usage:
  one file:   python3 cut_from_contour.py surface.byu contour_ids.txt outdir
  batch:      python3 cut_from_contour.py --batch SURF_DIR CONTOUR_DIR OUTDIR
              (matches <name>.byu in SURF_DIR with <name>_contour_ids.txt in CONTOUR_DIR)
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


def write_byu(path, V, F):
    with open(path, 'w') as f:
        f.write("1 %d %d %d\n1 %d\n" % (len(V), len(F), len(F) * 3, len(F)))
        for p in V:
            f.write("%f %f %f\n" % (p[0], p[1], p[2]))
        for tri in F:
            f.write("%d %d -%d\n" % (tri[0] + 1, tri[1] + 1, tri[2] + 1))


def check_health(F):
    """Return None if OK, else a string describing the problem (holes/non-manifold)."""
    ec = Counter()
    for tri in F:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            ec[(min(a, b), max(a, b))] += 1
    boundary = sum(1 for c in ec.values() if c == 1)
    nonman = sum(1 for c in ec.values() if c > 2)
    if nonman:
        return "non-manifold edges (%d)" % nonman
    if boundary:
        return "open boundary / hole (%d boundary edges)" % boundary
    return None


def build_adj(V, F):
    adj = defaultdict(dict)
    for tri in F:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            d = float(np.linalg.norm(V[a] - V[b])); adj[a][b] = d; adj[b][a] = d
    return adj


def dijkstra_path(adj, s, t):
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


def cut(byu_path, ids_path, outdir):
    name = os.path.splitext(os.path.basename(byu_path))[0]
    V, F = read_byu(byu_path)
    bad = check_health(F)
    if bad:
        print("SKIP %s: %s" % (name, bad))
        return False
    ids = np.loadtxt(ids_path, dtype=int).ravel()
    adj = build_adj(V, F)
    barrier = set()
    for i in range(len(ids)):
        barrier.update(dijkstra_path(adj, int(ids[i]), int(ids[(i + 1) % len(ids)])))

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
        print("SKIP %s: barrier covers whole surface" % name); return False
    A = flood(nonb[0]); rem = set(nonb) - A
    if not rem:
        print("SKIP %s: contour did not separate the surface (open loop)" % name); return False
    B = flood(next(iter(rem)))
    lab = np.full(len(V), 2)
    for v in A: lab[v] = 0
    for v in B: lab[v] = 1

    def submesh(region):
        keep = [fi for fi, tri in enumerate(F)
                if any(lab[v] == region for v in tri) and all(lab[v] != (1 - region) for v in tri)]
        vids = sorted({v for fi in keep for v in F[fi]})
        remap = {v: i for i, v in enumerate(vids)}
        return V[vids], np.array([[remap[v] for v in F[fi]] for fi in keep])

    os.makedirs(outdir, exist_ok=True)
    for r, nm in [(0, 'top'), (1, 'bottom')]:
        Vs, Fs = submesh(r)
        write_byu(os.path.join(outdir, "%s_%s.byu" % (name, nm)), Vs, Fs)
    print("OK   %s: top=%d / bottom=%d verts (barrier %d)" % (name, len(A), len(B), len(barrier)))
    return True


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == '--batch':
        surf_dir, cont_dir, outdir = sys.argv[2], sys.argv[3], sys.argv[4]
        n_ok = n_skip = 0
        for byu in sorted(glob.glob(os.path.join(surf_dir, '**', '*.byu'), recursive=True)):
            name = os.path.splitext(os.path.basename(byu))[0]
            ids = os.path.join(cont_dir, name + '_contour_ids.txt')
            if not os.path.isfile(ids):
                print("SKIP %s: no contour ids" % name); n_skip += 1; continue
            (n_ok := n_ok + 1) if cut(byu, ids, outdir) else (n_skip := n_skip + 1)
        print("\nbatch done: %d cut, %d skipped" % (n_ok, n_skip))
    elif len(sys.argv) == 4:
        cut(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(__doc__); sys.exit(1)


if __name__ == "__main__":
    main()
