#!/usr/bin/env python3
"""
contour_diagnostics.py
======================
Reusable diagnostics for the "can we auto-find the cut contour?" question,
run on the BIOCARD ERC+TEC surfaces. Two independent experiments:

  (A) feature_auc(thickness_dir)
      For thickness surfaces (single ERC+TEC patch with a per-point
      'displacement' = thickness scalar and an open boundary = the contour),
      measure how well each candidate feature separates CONTOUR vertices from
      interior vertices. AUC: 0.5 = no signal, ->1.0 = perfect separator.
      Finding on our data: thickness ~0.94 (low at rim), curvature ~0.59,
      i.e. the rim sits on LOW (relative, not zero) thickness; curvature is weak.

  (B) atlas_transfer(pieces_dir)
      For cut pieces (each case has BASE.vtk + BASE_top.vtk + BASE_bottom.vtk;
      the contour = open boundary of the _top piece), test whether a contour
      can be transferred between subjects by registration. Rigid (ICP) is used
      here as a lower bound; the lab's LDDMM should do better. Reports the
      mean nearest-distance between a transferred contour and the true contour.
      Finding on our data: single rigid atlas ~2.1 mm median (~1 vertex),
      5-atlas best-match ~1.4 mm. -> registration/atlas transfer is viable.

Requires: pyvista, numpy, scipy   (pip install pyvista numpy scipy)
Usage:
    python3 contour_diagnostics.py auc    "/path/to/RH_CON_thickness"
    python3 contour_diagnostics.py atlas  "/path/to/Already Cut pieces"
"""
import sys, glob, os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def boundary_points(path):
    """Return Nx3 coords of the open-boundary (contour) vertices of a mesh."""
    m = pv.read(path)
    e = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                manifold_edges=False, non_manifold_edges=False)
    return np.asarray(e.points)


def boundary_mask(m):
    """Boolean per-vertex mask: True if the vertex lies on the open boundary."""
    from collections import Counter
    F = m.faces.reshape(-1, 4)[:, 1:]
    ec = Counter()
    for tri in F:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            ec[(min(a, b), max(a, b))] += 1
    mask = np.zeros(m.n_points, bool)
    for (a, b), c in ec.items():
        if c == 1:
            mask[a] = True; mask[b] = True
    return mask


def _auc(feat, label):
    r = rankdata(feat); npos = label.sum(); nneg = (~label).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    return (r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def feature_auc(thickness_dir):
    files = sorted(glob.glob(os.path.join(thickness_dir, "*", "*.vtk"))) \
        or sorted(glob.glob(os.path.join(thickness_dir, "*.vtk")))
    print("feature-separation AUC over %d files (0.5=no signal)\n" % len(files))
    acc = {}
    for f in files:
        m = pv.read(f)
        t = np.asarray(m.point_data.get('displacement',
                       m.point_data.get('weights', np.zeros(m.n_points)))).astype(float)
        bnd = boundary_mask(m)
        adj = [[] for _ in range(m.n_points)]
        F = m.faces.reshape(-1, 4)[:, 1:]
        for tri in F:
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                adj[a].append(b); adj[b].append(a)
        gm = np.array([np.mean([abs(t[i]-t[j]) for j in adj[i]]) if adj[i] else 0
                       for i in range(m.n_points)])
        cu = np.abs(np.asarray(m.curvature('mean')))
        feats = {'thickness': t, '|grad thickness|': gm, '|mean curvature|': cu}
        for k, v in feats.items():
            a = _auc(v, bnd)
            acc.setdefault(k, []).append(a if k == 'thickness' else max(a, 1 - a))
    for k, v in acc.items():
        print("  %-18s AUC %.2f" % (k, np.nanmean(v)))
    print("\n(thickness AUC near 0 = strongly LOW at the rim; report as %.2f discriminative)"
          % (1 - np.nanmean(acc['thickness'])))


def atlas_transfer(pieces_dir, n_atlas=5):
    cases = [t[:-8] for t in sorted(glob.glob(os.path.join(pieces_dir, "*_top.vtk")))
             if os.path.exists(t[:-8] + ".vtk")]
    print("atlas-transfer feasibility over %d cases\n" % len(cases))
    sizes = sorted([(c, pv.read(c + ".vtk").n_points) for c in cases], key=lambda x: x[1])
    atlases = [sizes[int(f * (len(sizes) - 1))][0]
               for f in np.linspace(0.1, 0.9, n_atlas)]
    A = [(a, pv.read(a + ".vtk"), boundary_points(a + "_top.vtk")) for a in atlases]

    def one(As, Ac, target):
        B = pv.read(target + ".vtk")
        _, T = As.align(B, return_matrix=True)
        Act = (np.c_[Ac, np.ones(len(Ac))] @ np.array(T).T)[:, :3]
        Bc = boundary_points(target + "_top.vtk")
        if len(Bc) == 0:
            return None
        return 0.5 * (cKDTree(Bc).query(Act)[0].mean() + cKDTree(Act).query(Bc)[0].mean())

    best = []
    for c in cases:
        if c in atlases:
            continue
        es = [one(As, Ac, c) for (_, As, Ac) in A]
        es = [e for e in es if e is not None]
        if es:
            best.append(min(es))
    best = np.array(best)
    print("  rigid (ICP) multi-atlas best-match transfer error:")
    print("    median %.2f mm | mean %.2f mm | under 2mm %.0f%%"
          % (np.median(best), best.mean(), (best < 2).mean() * 100))
    print("\n  NOTE: this is a RIGID lower bound. Diffeomorphic (LDDMM) registration,")
    print("  which absorbs narrow/wide/bent shape variation, should do better.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "auc":
        feature_auc(path)
    elif mode == "atlas":
        atlas_transfer(path)
    else:
        print("mode must be 'auc' or 'atlas'"); sys.exit(1)
