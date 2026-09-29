# Gas-jet density post-processing (`fludat_proc`)

Tools for turning gas-jet fluid-simulation exports into the regular density cubes that
FBPIC reads, plus convergence and shock diagnostics for the simulations themselves.

## Pipeline

```
ANSYS lineout exports            CGNS field exports
data/density_lineouts_raw/       data/density_field_raw/
        │                                │
        ▼ lineouts_to_density_hdf5              ▼ cgns_to_density_hdf5
        └──────────► HDF5 density cube ◄─┘
                     density[z_m, x_mm, pressure_bar]
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        interpolate_density               plot_density
        (library; FBPIC callable)         (maps, lineouts, PNG export)

convergence      – grid-resolution study on lineout or CGNS exports
shock_evaluator  – interactive |grad rho| heatmap with oblique-shock check
```

The CGNS route is the production path: FBPIC reads the uniform-grid cube produced by
`cgns_to_density_hdf5` through `interpolate_density.build_density_callable`.

## Installation and running

The package lives in [`fludat_proc/`](fludat_proc/). No installation is needed: the
command-line modules can be run directly as scripts, or with `python -m` from this
directory. Installing (`pip install -e .`) additionally registers the console scripts below.

```bash
conda run -n inv-fbpic python fludat_proc/plot_density.py data/density_lineouts/400_um.h5
conda run -n inv-fbpic python -m fludat_proc.plot_density data/density_lineouts/400_um.h5
```

| Module | Console script | Purpose |
|---|---|---|
| `fludat_proc.lineouts_to_density_hdf5` | `lineouts-to-density-hdf5` | ANSYS lineouts → density cube |
| `fludat_proc.cgns_to_density_hdf5` | `cgns-to-density-hdf5` | CGNS fields → density cube |
| `fludat_proc.plot_density` | `plot-density` | Plot a density cube |
| `fludat_proc.convergence` | `density-convergence` | Grid-convergence study |
| `fludat_proc.shock_evaluator` | `shock-evaluator` | Interactive shock inspection |

Library modules: `ansys_lineouts` (raw lineout parsing), `cgns_io` (CGNS reading),
`density_cube` (the HDF5 format), `interpolate_density`, `filenames`, `common`.

Requirements: Python ≥ 3.10, NumPy ≥ 2.0, SciPy, Matplotlib, h5py. Tests: `pytest`.

### Conventions

- **Axes.** `z` is the laser/propagation axis, `x` the transverse position. In the
  Fluent CGNS exports these are `CoordinateY` and `CoordinateX` respectively.
- **Units on the command line.** Every `--x-bounds` / `--z-bounds` option takes
  **millimetres**. Inside the HDF5 cube `z_m` is in metres and `x_mm` in millimetres.
- **Filenames.** Backing pressure is encoded as `<pressure>_bar.<ext>` (`5_bar.txt`,
  `12.5_bar.cgns`). Grid size for convergence studies is encoded in mm with `_` as
  the decimal point: `0_075.cgns` means 0.075 mm.
- **Symmetry.** Every cube is mirrored across `z = 0`; only `z ≥ 0` (and, for CGNS,
  `x ≥ 0`) source data is used.

## The density-cube format

One HDF5 file per nozzle:

| Dataset | Shape | Units |
|---|---|---|
| `z_m` | `(n_z,)` | m |
| `x_mm` | `(n_x,)` | mm |
| `pressure_bar` | `(n_pressure,)` | bar |
| `density` | `(n_z, n_x, n_pressure)` | `density.attrs["units"]` |

Root attributes: `nozzle`, `density_scale`, `density_axis_order="z_m,x_mm,pressure_bar"`
plus provenance attributes written by the producing tool (`source_format`,
`source_files`, interpolation settings, ...). Each `density` axis carries an HDF5
dimension-scale link and label to its coordinate dataset, so `load_density_cube`
can verify the axis mapping rather than infer it from array lengths.

---

## `cgns_to_density_hdf5`

Interpolate unstructured CGNS density fields onto a regular `(z, x)` grid and write a
density cube. A directory of `<pressure>_bar.cgns` files becomes the pressure axis; a
single file needs `--pressure`.

Only the `CoordinateX ≥ 0, CoordinateY ≥ 0` quadrant of each field is used. The output
z grid is symmetric and every grid point is sampled at `|z|`, so the cube is exactly
mirrored. Grid points outside the simulated domain are zero by default.

```bash
python -m fludat_proc.cgns_to_density_hdf5 data/density_field_raw/htu_fields_7_0 \
    --output data/density_field/htu_dens_7_0.h5 \
    --density-units kg/m^3 --z-bounds -7.5 7.5 --x-bounds 0 15
```

| Argument | Description |
|---|---|
| `input` | Directory of `<pressure>_bar.cgns` files, or one CGNS file with `--pressure`. |
| `-o`, `--output` | Required HDF5 output path. |
| `--density-units` | Required units of the (scaled) output density. |
| `--density-scale` | Factor applied to the source density; default `1`. |
| `--pressure` | Backing pressure [bar] for single-file input. |
| `--nozzle` | Nozzle name stored in the file; default input name. |
| `--z-points`, `--x-points` | Output grid resolution; defaults `1000` and `500`. |
| `--z-bounds`, `--x-bounds` | Output bounds [mm]; default symmetric common z extent and common x extent. |
| `--interpolation` | `linear` (default), `nearest`, or `cubic`. |
| `--outside-fill` | `zero` (default), `nearest`, or `raise` for grid points outside the source domain. |

---

## `lineouts_to_density_hdf5`

Build a density cube from ANSYS lineout exports: one text file per backing pressure,
each holding one `(z, density)` section per transverse position:

```
((xy/key/label "x-0-5")
0.002675  0.00097875
0.0027    0.00087528
...
)
```

Section labels `x-<int>[-<tenths>]` give the x position in mm (`x-0-5` → 0.5 mm).
Each lineout is mirrored across `z = 0`, multiplied by `density_scale`, and resampled
onto the z grid of the first lineout.

```bash
python -m fludat_proc.lineouts_to_density_hdf5 1.0e20 data/density_lineouts_raw/400_um \
    --output data/density_lineouts/400_um.h5 --density-units cm^-3
```

| Argument | Description |
|---|---|
| `density_scale` | Required factor applied to raw densities. |
| `input` | Nozzle directory of `<pressure>_bar.txt` files (or one such file). |
| `-o`, `--output` | Required HDF5 output path. |
| `--density-units` | Required units of the scaled density. |
| `--nozzle` | Nozzle name stored in the file; default directory name. |

---

## `interpolate_density` (library)

```python
from fludat_proc.interpolate_density import (
    build_density_callable, build_density_interpolation,
)

field = build_density_interpolation("data/density_field/htu_dens_7_0.h5", method="linear")
rho = field.interpolate(z_value=0.0, x_value=1.0, pressure_value=12.5)
profile = field.interpolate_along_z(field.z, x_value=1.0, pressure_value=12.5)
grid = field.interpolate_xz_grid(x_values, z_values, pressure_value=12.5)  # (n_x, n_z)

# FBPIC: density(z, r) at fixed x and backing pressure; r is ignored.
density = build_density_callable(
    "data/density_field/htu_dens_7_0.h5", backing_pressure=12.5, x_position=1.0
)
```

`method` selects the `(x, pressure)` interpolation: `linear` (default), `cubic`,
`quintic`, or `pchip` (the spline methods need ≥ 4 or ≥ 6 samples per axis). Along
`z` interpolation is always linear. Queries outside the tabulated `z`, `x`, or
pressure range evaluate to zero density rather than raising, so an FBPIC window that
extends beyond the simulated jet sees vacuum there.

---

## `plot_density`

```bash
# Interactive (x, z) map with a pressure slider
python -m fludat_proc.plot_density data/density_lineouts/400_um.h5

# Interactive lineout along z with pressure and x sliders
python -m fludat_proc.plot_density data/density_lineouts/400_um.h5 --plot lineout

# Same, evaluated through build_density_callable (checks the FBPIC path)
python -m fludat_proc.plot_density data/density_lineouts/400_um.h5 --plot callable_lineout

# Static figure
python -m fludat_proc.plot_density data/density_lineouts/400_um.h5 \
    --plot xz_map --pressure 12.5 --z-bounds -3 3 -o density_map.png
```

| Argument | Default | Description |
|---|---|---|
| `hdf5_path` | — | Density cube. |
| `--plot` | `xz_map` | `xz_map`, `lineout`, or `callable_lineout`. |
| `--method` | `linear` | `(x, pressure)` interpolation method. |
| `--pressure` | lowest | Backing pressure [bar]. |
| `--x` | lowest | x position [mm] for lineouts. |
| `--x-bounds`, `--z-bounds` | full extent | Plot bounds [mm]. |
| `-o`, `--output` | — | Save at 150 dpi instead of showing; disables sliders. |
| `-t`, `--title` | auto | Figure title. |

Density axes are labelled with the units stored in the cube.

---

## `convergence`

Compare each grid resolution against a reference, either the finest grid (default,
treated as ground truth) or the next finer grid (`--reference adjacent`). Input is a
directory of resolution-tagged files: `0_1.txt` / `0_1.cgns` for 0.1 mm.

**Lineout input** (`.txt`): for each matching section label the two profiles are
interpolated onto their shared z samples and the relative integrated error

```
∫ |ρ_coarse − ρ_ref| dz  /  ∫ |ρ_ref| dz
```

is computed over the raw (non-mirrored) z range. Duplicate z samples keep the last
value.

**CGNS input**: every field is clipped to `--x-bounds` / `--z-bounds`, interpolated once
onto one shared regular grid, and compared with the same area-integrated metric. The
summary plot and CSV also report two pointwise error distributions over the grid: the
local relative error `|ρ − ρ_ref| / |ρ_ref|` (cells with `ρ_ref = 0` excluded) and the
peak-normalised error `|ρ − ρ_ref| / max|ρ_ref|`. `--field-error` chooses which appear
in the plot.

```bash
python -m fludat_proc.convergence data/density_lineouts_raw/htu_conv_test
python -m fludat_proc.convergence data/density_field_raw/htu_conv_7_0 \
    --x-bounds 0 15 --output figures/htu_cgns_convergence.png
```

| Argument | Description |
|---|---|
| `input_dir` | Directory of resolution-tagged `.txt` lineouts or `.cgns` fields. |
| `--input-format` | `auto` (default: CGNS if any `.cgns` present), `lineout`, `cgns`. |
| `--reference` | `finest` (default) or `adjacent`. |
| `--view` | Lineout plots: `summary`, `lineouts`, or `both` (default). CGNS always `summary`. |
| `--field-error` | CGNS summary curves: `local-relative`, `peak-normalized`, `both` (default). |
| `--x-bounds`, `--z-bounds` | CGNS comparison region [mm]. |
| `--x-points`, `--z-points` | CGNS common-grid resolution; default `200` each. |
| `--interpolation` | CGNS: `linear` (default) or `cubic` Clough-Tocher. |
| `-o`, `--output` | Save the plot instead of showing it. |
| `--metrics-output` | CSV path; defaults to `<output>.csv` when `--output` is given. No CSV is written for interactive runs unless requested. |
| `-t`, `--title` | Plot title. |

---

## `shock_evaluator`

Interactive inspection of `|∇ρ| = sqrt(dp-dX² + dp-dY²)` from one CGNS file that
contains the point-aligned `dp-dX` and `dp-dY` fields.

```bash
python -m fludat_proc.shock_evaluator data/generic_field_raw/htu_all_fields.cgns
```

Drag with the primary mouse button across the heatmap to define a segment. The lineout
panel samples the checked scalar fields along it (distance in mm from the drag start;
gaps where the segment leaves the mesh). One field is shown in source units; several
are each scaled to unit magnitude for shape comparison.

When `Mach`, `Axial_Velocity`, `Radial_Velocity`, `Pressure`, `Density`, and
`Temperature` are available, the report panel compares the observed downstream/upstream
ratios `P₂/P₁`, `ρ₂/ρ₁`, `T₂/T₁` with oblique-shock predictions for the upstream Mach
number `M₁` and shock angle `β`. In every mode the endpoint with the larger Mach number
is upstream (state 1), so the drag direction does not matter; equal endpoint Mach
numbers are rejected as ambiguous. The angle comes from:

- `--shock-angle auto` (default): the velocity change `v₁ − v₂` between the endpoints.
- `--shock-angle perp`: the normal to the dragged segment.
- `--shock-angle manual`: after dragging, click a third point; the ray from the segment
  midpoint to it is the shock direction.

The shock direction is drawn as a dashed lime line from the segment midpoint.

| Argument | Description |
|---|---|
| `cgns_path` | Input unstructured CGNS field. |
| `--line-samples N` | Samples per segment; default `400`. |
| `--gamma` | Heat-capacity ratio; default `1.4`. |
| `--shock-angle` | `auto`, `perp`, or `manual`. |
| `--show-color-limit-boxes` | Show editable log colour limits (positive, increasing). |
| `--flow-solution`, `--x-coordinate`, `--z-coordinate` | CGNS node names. |
| `--x-bounds`, `--z-bounds` | Clip the loaded field [mm]. |

---

## Data layout

`data/` is not tracked by git (`*.cgns`, `*.h5`, `*.txt`, `*.csv`, `*.png` are ignored
repository-wide). The expected layout:

```
data/
├── density_lineouts_raw/<nozzle>/<pressure>_bar.txt     ANSYS lineout exports
├── density_lineouts/<nozzle>.h5                         cubes from lineouts_to_density_hdf5
├── density_field_raw/<suite>/<pressure>_bar.cgns        CGNS pressure scans
├── density_field_raw/<suite>/<grid>.cgns                CGNS resolution scans
├── density_field/<suite>.h5                             cubes from cgns_to_density_hdf5
└── generic_field_raw/*.cgns                             all-fields exports for shock_evaluator
```

## Development

```bash
conda run -n inv-fbpic python -m pytest      # from this directory
ruff check . && ruff format .                # lint/format (config in pyproject.toml)
```
