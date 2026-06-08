"""
ContourMappings.py  -- automatic contour by template-boundary transfer.
Reuses the SAME py-lddmm calls as AdniMappings.py.

PATHS ARE PRE-FILLED for the cluster. You do NOT need to edit anything to run
the first test. Just run (see instructions Carol was given).

What it does per surface:
  rigid align -> SurfaceMatching(Template, Target) -> f.fvDef (template on subject)
  -> carry template boundary loop onto subject -> save contour as .vtk polyline.

TEST_ONE = True processes ONE surface so you can eyeball it in ParaView first.
Set TEST_ONE = False (last resort: edit the one line below) to batch all.
"""
from sys import path as sys_path
# Absolute paths to the OLD py-lddmm (works with: pyfftw 0.13.1 + pykeops 2.1.2 + meshpy).
# Absolute so this script runs from YOUR OWN project directory, not just py-lddmm-scripts.
sys_path.insert(0, '/cis/project/adni_thickness/registration_old/py-lddmm')
sys_path.insert(0, '/cis/project/adni_thickness/registration_old/py-lddmm/base')

import glob, os
import numpy as np
from collections import Counter, defaultdict
from scipy.spatial import KDTree
import h5py
from base.surfaces import Surface
from base.kernelFunctions import Kernel
from base.surfaceMatching import SurfaceMatching
from base.affineRegistration import rigidRegistration
from base import loggingUtils

# ============================ CONFIG (pre-filled) ============================
# Your own project workspace. Your inputs/outputs live here; the shared template
# and py-lddmm code are referenced by absolute path above / below.
PROJECT_DIR = '/cis/home/jli401/contour_project'
TEST_ONE    = True
MAXITER     = 100     # this + sigmaError 0.25 gave the preferred result; fast (~8 min/surface).
INPUT_GLOB  = PROJECT_DIR + '/byu_tests/*/cut/*.byu'
OUTPUT_DIR  = PROJECT_DIR + '/auto_contours'
TEMPLATE_H5 = '/cis/project/adni_thickness/Results/ADNIthickness.h5'   # shared
REFERENCE_SURFACE = None   # set to a hand-cut CLOSED .byu if the patch template fails
# ============================================================================

K1 = Kernel(name='laplacian', sigma=5.0)
K2 = Kernel(name='gauss', sigma=5.0)
options = {
    'mode': 'normal', 'timeStep': 0.1, 'KparDiff': K1, 'KparDist': K2,
    'internalCost': [['elastic', 50]], 'internalWeight': 1.,
    'sigmaError': .25, 'errorType': 'varifold', 'maxIter': MAXITER,
    'affine': 'none', 'rotWeight': 10., 'saveRate': 50, 'transWeight': 1.,
    'algorithm': 'bfgs', 'scaleWeight': 10., 'affineWeight': 100.,
    'verb': True, 'pk_dtype': 'float32',
}
import time


def read_byu_arrays(path):
    """Parse a BYU file into (vertices Nx3, faces Mx3)."""
    toks = open(path).read().split()
    it = iter(toks)
    npart = int(next(it)); nv = int(next(it)); nf = int(next(it)); ne = int(next(it))
    for _ in range(npart):
        next(it); next(it)
    verts = np.array([float(next(it)) for _ in range(nv * 3)]).reshape(nv, 3)
    conn = [int(next(it)) for _ in range(ne)]
    faces = []; cur = []
    for c in conn:
        if c < 0:
            cur.append(-c - 1); faces.append(cur); cur = []
        else:
            cur.append(c - 1)
    return verts, np.array(faces, dtype=int)


def load_surface(path):
    """Load .byu or .vtk into a py-lddmm Surface (robust to format)."""
    try:
        return Surface(path)
    except Exception:
        v, fc = read_byu_arrays(path)
        return Surface(surf=(fc, v))


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


def write_polyline_vtk(points, path):
    n = len(points)
    with open(path, 'w') as fh:
        fh.write("# vtk DataFile Version 3.0\nauto contour\nASCII\nDATASET POLYDATA\n")
        fh.write("POINTS %d float\n" % n)
        for p in points:
            fh.write("%f %f %f\n" % (p[0], p[1], p[2]))
        fh.write("LINES 1 %d\n" % (n + 2))
        fh.write("%d " % (n + 1) + " ".join(str(k) for k in range(n)) + " 0\n")


# ---- reference (template) ----
if REFERENCE_SURFACE:
    template = load_surface(REFERENCE_SURFACE)
else:
    with h5py.File(TEMPLATE_H5, 'r') as f5:
        template = Surface(surf=(np.array(f5['template']['faces']),
                                 np.array(f5['template']['vertices'])))
loop_idx = ordered_boundary_loop(template.faces)
print("template boundary loop: %d vertices" % len(loop_idx))
if len(loop_idx) == 0:
    raise SystemExit("Reference has no open boundary; use the patch template or a "
                     "closed surface with a marked contour.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Route py-lddmm's per-iteration logs to the screen (and to auto_contours/info.txt)
loggingUtils.setup_default_logging(OUTPUT_DIR, fileName='info.txt', stdOutput=True)
files = sorted(glob.glob(INPUT_GLOB))
if not files:
    raise SystemExit("No surfaces found at: %s" % INPUT_GLOB)
if TEST_ONE:
    files = files[:1]
print("processing %d surface(s)" % len(files))

for nproc, ffile in enumerate(files, 1):
    t0 = time.time()
    name = os.path.splitext(os.path.basename(ffile))[0]
    print("[%d/%d] registering %s ..." % (nproc, len(files), name), flush=True)
    surf = load_surface(ffile)
    orig = surf.vertices.copy()
    R0, T0 = rigidRegistration(surfaces=(surf.vertices, template.vertices),
                               rotWeight=0., verb=False, temperature=10., annealing=True)
    surf.updateVertices(surf.vertices @ R0.T + T0)
    options['outputDir'] = os.path.join(OUTPUT_DIR, name)
    f = SurfaceMatching(Template=template, Target=surf, options=options)
    f.optimizeMatching()
    pred_aligned = f.fvDef.vertices[loop_idx]
    _, ids = KDTree(surf.vertices).query(pred_aligned)
    contour = orig[ids]
    out_vtk = os.path.join(OUTPUT_DIR, name + '_contour.vtk')
    write_polyline_vtk(contour, out_vtk)
    np.savetxt(os.path.join(OUTPUT_DIR, name + '_contour_ids.txt'), ids, fmt='%d')
    print("  -> %s (%d pts, %.1f min)" % (out_vtk, len(contour), (time.time()-t0)/60), flush=True)

print("DONE. Open each .byu with its *_contour.vtk in ParaView to verify.")
