# Environment setup (Stage 1: py-lddmm registration)

Stage 1 (`ContourMappings.py`) needs `py-lddmm` plus several compiled
dependencies. This is the exact, tested recipe — written down because the
defaults do **not** work (newer package versions broke the API, and one package
fails to compile against the system).

Stage 2 (`finalize_cuts.py` / `cut_from_contour.py`) needs only `numpy` and
`scipy` — no special setup.

## Which py-lddmm

Use the **older** copy of py-lddmm (in our project this was
`registration_old/py-lddmm`). The newer copy imports `ngsolve`, a heavy FEM
package this pipeline does not need. The scripts add the path explicitly:

```python
sys_path.insert(0, '<...>/registration_old/py-lddmm')
sys_path.insert(0, '<...>/registration_old/py-lddmm/base')
```

## Packages (install into a conda env)

Most dependencies come with anaconda. The three that need care:

```bash
# pyfftw: install an OLDER version that ships a prebuilt wheel.
# The newest (0.14) has no wheel for py3.9 and tries to compile against
# fftw3.h, which isn't present on the shared cluster (no sudo).
pip install "pyfftw==0.13.1"

# pykeops: MUST be 2.1.2. The code calls keopscore.config.config.use_cuda,
# an API that keops 2.3 removed. 2.1.2 has it.
pip install "pykeops==2.1.2"

# meshpy: imported by pointSets.py
pip install meshpy
```

Other requirements (numpy, scipy, vtk, numba, matplotlib, pillow,
scikit-image, nibabel, imageio, h5py, tqdm, pandas) are typically already in
anaconda `base`. `pygalmesh` is optional — a "could not import Pygalmesh
functions" warning is harmless.

## Verify it imports

From the scripts directory:

```bash
python3 -c "import sys; \
  sys.path.insert(0,'<...>/registration_old/py-lddmm'); \
  sys.path.insert(0,'<...>/registration_old/py-lddmm/base'); \
  from base.surfaces import Surface; \
  from base.surfaceMatching import SurfaceMatching; \
  from base.kernelFunctions import Kernel; \
  from base.affineRegistration import rigidRegistration; \
  print('ALL IMPORTS OK')"
```

`[KeOps] ... CUDA libraries not found ... CPU only` is fine (just slower).

## GPU

keops uses CUDA automatically when available — roughly 100x faster than CPU.
Run Stage 1 on a GPU node if you can; otherwise CPU works at minutes/surface.
