#!/usr/bin/env python3
"""
build_group_templates.py -- estimate one LDDMM AVERAGE template per shape group,
using ALL surfaces in the group.

Avoids pygalmesh: (1) stubs the pygalmesh import, (2) initialises the template
from one of your own surfaces (median vertex count) instead of an auto-generated
pygalmesh shape. computeTemplate then averages every surface in the group.

Folders sharing a basename merge; 'tri:weird' -> 'tri_weird'.

NOTE ON SPEED: averaging registers the template against all group surfaces every
iteration -- ~15-20 min/iteration on CPU. With TEMPLATE_MAXITER below at 100
that's ~a day+ per group. A GPU (keops uses it automatically) cuts this ~100x.

Usage (cluster, py-lddmm env, in screen):
    python3 build_group_templates.py TEMPLATE_OUTDIR GROUP_DIR [GROUP_DIR ...]
"""
import sys, os, glob, re, types, time
import numpy as np

sys.path.insert(0, '/cis/project/adni_thickness/registration_old/py-lddmm')
sys.path.insert(0, '/cis/project/adni_thickness/registration_old/py-lddmm/base')

try:
    import pygalmesh  # noqa: F401
except Exception:
    _stub = types.ModuleType('pygalmesh'); _stub.DomainBase = object
    sys.modules['pygalmesh'] = _stub

from base.surfaces import Surface
from base.kernelFunctions import Kernel
from base.surfaceTemplate import SurfaceTemplate
from base.affineRegistration import rigidRegistration
from base import loggingUtils

TEMPLATE_MAXITER = 100   # the average is essentially converged by ~80-100 iters;
                         # 100 keeps quality while ~halving the 200-iter runtime.
K1 = Kernel(name='laplacian', sigma=5.0)
K2 = Kernel(name='gauss', sigma=5.0)
TEMPLATE_OPTIONS = {
    'mode': 'normal', 'timeStep': 0.1, 'KparDiff': K1, 'KparDist': K2,
    'sigmaError': 1., 'errorType': 'current', 'testGradient': False,
    'lambdaPrior': 1., 'maxIter': TEMPLATE_MAXITER, 'affine': 'none', 'rotWeight': 10.,
    'sgd': None, 'transWeight': 1., 'scaleWeight': 10., 'affineWeight': 100.,
    'updateTemplate': True, 'pk_dtype': 'float32', 'verb': True,
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


def write_vtk_ascii(path, V, F):
    with open(path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n" + os.path.basename(path) +
                "\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS %d float\n" % len(V))
        for p in V:
            f.write("%f %f %f\n" % (p[0], p[1], p[2]))
        f.write("POLYGONS %d %d\n" % (len(F), len(F) * 4))
        for tri in F:
            f.write("3 %d %d %d\n" % (tri[0], tri[1], tri[2]))


def build_one(name, files, outdir):
    if not files:
        print("GROUP %s: no surfaces, skipping" % name); return
    print("GROUP %s: averaging %d surfaces" % (name, len(files)))
    surfs = [load_surface(f) for f in files]
    order = sorted(range(len(surfs)), key=lambda i: surfs[i].vertices.shape[0])
    init = surfs[order[len(order) // 2]]
    aligned = []
    for s in surfs:
        R0, T0 = rigidRegistration(surfaces=(s.vertices, init.vertices),
                                   rotWeight=0., verb=False, temperature=10., annealing=True)
        s.updateVertices(s.vertices @ R0.T + T0)
        aligned.append(s)
    opt = dict(TEMPLATE_OPTIONS)
    opt['outputDir'] = os.path.join(outdir, '_work_' + name)
    t0 = time.time()
    f = SurfaceTemplate(Template=init, Target=aligned, options=opt)
    f.computeTemplate()
    tmpl = Surface(f.fvTmpl)
    out = os.path.join(outdir, name + '_template.vtk')
    write_vtk_ascii(out, tmpl.vertices, tmpl.faces)
    print("  -> %s  (%d verts, %.1f min)" % (out, tmpl.vertices.shape[0], (time.time() - t0) / 60))


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    loggingUtils.setup_default_logging(outdir, fileName='template_log.txt', stdOutput=True)
    groups = {}
    for gd in sys.argv[2:]:
        raw = os.path.basename(os.path.normpath(gd)).lower()
        nm = re.sub(r'[^a-z0-9]+', '_', raw).strip('_')
        groups.setdefault(nm, []).extend(
            sorted(glob.glob(os.path.join(gd, '*.vtk'))) +
            sorted(glob.glob(os.path.join(gd, '*.byu'))))
    for nm, files in groups.items():
        build_one(nm, files, outdir)
    print("\nDone. Averaged templates in %s/" % outdir)


if __name__ == "__main__":
    main()
