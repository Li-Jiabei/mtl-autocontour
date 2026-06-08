# MTL Surface Auto-Contour Pipeline

Automatically place the cutting contour on medial-temporal-lobe surfaces
(entorhinal / ERC+TEC) and split each surface into **top** and **bottom**
regions — replacing the manual contour tracing previously done by hand in
ParaView.

The method: an **LDDMM population template** is registered onto each surface
(using `py-lddmm`), the template's boundary loop is **transferred** onto the
surface, and the surface is **cut along that transferred contour, routed on the
mesh**. On validation data the transferred contour matched manual tracings to
sub-millimetre accuracy.

---

## How it works (two stages)

```
  surface (.byu)
        │
        │  [Stage 1 — needs py-lddmm, run on the cluster]
        │     register the LDDMM template onto the surface,
        │     transfer the template boundary -> contour vertex IDs
        ▼
  *_contour_ids.txt
        │
        │  [Stage 2 — pure numpy/scipy, runs anywhere]
        │     route the contour along mesh edges into a closed barrier,
        │     flood-fill into two regions, write surfaces
        ▼
  <name>.vtk   <name>_top.vtk   <name>_bottom.vtk   (ASCII, per subject)
```

Stage 2 is deliberately independent of the registration: the contour IDs are
snapped onto the surface and re-routed along mesh edges, so a slightly loose
registration still yields a clean, watertight cut.

---

## Repository contents

```
scripts/
  ContourMappings.py        Stage 1: register template -> surfaces, write contour IDs (py-lddmm)
  finalize_cuts.py          Stage 2: cut into top/bottom, output clean ASCII VTK by subject
  cut_from_contour.py       Stage 2 (simpler variant): cut one or many, .byu output
  contour_diagnostics.py    QC: feature-separation AUC + atlas-transfer feasibility checks
docs/
  ENVIRONMENT.md            Exact environment recipe (the dependency setup that works)
```

---

## Environment (Stage 1 only)

Stage 1 needs `py-lddmm` and a specific set of dependencies. The full, tested
recipe is in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md). Short version, in a
conda env on the cluster:

```bash
pip install "pyfftw==0.13.1"   # prebuilt wheel (avoids compiling against system FFTW)
pip install "pykeops==2.1.2"   # MUST be 2.1.2 — newer keops changed an API the code uses
pip install meshpy
```
and use the **`registration_old`** copy of py-lddmm (the newer one needs
`ngsolve`, which this pipeline does not). Stage 2 needs only `numpy` + `scipy`.

---

## Stage 1 — registration & contour transfer

`scripts/ContourMappings.py`. Edit the CONFIG block at the top (project dir,
input glob, output dir, template `.h5`), then run on the cluster:

```bash
cd <your project dir>
nohup python3 ContourMappings.py > batch.log 2>&1 &   # survives disconnects
tail -f batch.log
```

- `TEST_ONE = True` processes a single surface (use first, to sanity-check).
  Set `False` to batch the whole input folder.
- `MAXITER = 100` and `sigmaError = 0.25` are the validated settings
  (good quality, ~8 min/surface on CPU). Lower `sigmaError` / raise `MAXITER`
  for a tighter fit if ever needed (diminishing returns).
- Output: one `*_contour_ids.txt` (and a `*_contour.vtk` preview) per surface.

GPU note: keops uses CUDA automatically. On a GPU node this is ~100x faster;
on CPU expect minutes per surface.

## Stage 2 — cut & finalize

`scripts/finalize_cuts.py`. Runs anywhere (no py-lddmm):

```bash
python3 finalize_cuts.py SURF_DIR CONTOUR_DIR OUTPUT_DIR
# e.g.
python3 finalize_cuts.py byu_tests/CON_LH auto_contours final_vtk
```

For each `<name>.byu` that has a matching `<name>_contour_ids.txt`, writes three
**ASCII** VTK files into `OUTPUT_DIR/<subject>/`:

```
final_vtk/
└── BEIALE/
    ├── BEIALE_150428_7.vtk          (original surface, byu -> vtk)
    ├── BEIALE_150428_7_top.vtk
    ├── BEIALE_150428_7_bottom.vtk
    └── ... (more timepoints)
```

`<subject>` is the text before the first underscore. Surfaces with holes /
non-manifold edges are skipped and logged; contours that fail to close are
skipped and logged. Intermediate files (contour IDs, logs) are left in place —
the output tree contains only the three surfaces per timepoint.

---

## QC

`scripts/contour_diagnostics.py` — sanity checks used during development:
`auc` mode measures how well candidate features separate contour vs interior
vertices; `atlas` mode estimates contour-transfer feasibility between subjects.

---

## Notes & caveats

- **Template type vs target type.** The bundled template is an open ERC *patch*;
  the test surfaces are *closed*. Registering a patch onto a closed surface
  localises well but the contour line can sit slightly off — Stage 2's on-mesh
  routing absorbs this, but for best placement consider re-estimating a template
  from closed surfaces, or a closed→closed reference.
- **The template is fixed.** Processing surfaces does NOT update the template;
  each surface is registered independently for reproducibility.
- **Tuning.** Placement is governed by the registration (`sigmaError`,
  `MAXITER`), not the cut. The cut itself is robust to a loose contour.

---

## Credits

LDDMM registration: `py-lddmm` by Laurent Younes (JHU). Pipeline, contour
transfer, and cutting tools by Jiabei Li. Surface-cutting concept adapted from
the Center for Imaging Science workflow.
