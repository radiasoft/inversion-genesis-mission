# Fluid HDF5 Interpolation Demo

This demo plots longitudinal density lineouts from an N-dimensional HDF5
density grid using `InterpolateFromH5Profile`. It samples the stored pressure
values and their midpoints, then marks the cutoff-derived profile bounds and
centroid.

## Files

| File | Purpose |
|------|---------|
| `plot_profiles.py` | Loads HDF5 density grids and writes a lineout plot. |
| `<profile>.h5` | User-supplied density profile in the format below. Not included in the repository. |

## HDF5 format

Each input profile must contain a root-level density dataset and one root-level
coordinate dataset for each density dimension:

```text
<profile>.h5
|
+-- density_name                         # N-dimensional density values
|   +-- shape: (N_axis_0, N_axis_1, ..., N_axis_M)
|   `-- attrs["DIMENSION_LABELS"]:
|         ["axis_0_name", "axis_1_name", ..., "axis_M_name"]
|                                           # Labels in dataset-dimension order
|
+-- axis_0_name                          # Coordinates for density_name axis 0
|   `-- shape: (N_axis_0,)
|
+-- axis_1_name                          # Coordinates for density_name axis 1
|   `-- shape: (N_axis_1,)
...
|
`-- axis_M_name                          # Coordinates for density_name axis M
    `-- shape: (N_axis_M,)
```

The density values must be finite and non-negative. Every coordinate array must
be finite, strictly increasing, contain at least two entries, and match its
corresponding density dimension. The default script expects a dataset named
`density` and uses `pressure_bar`, `z_m`, and `x_mm` as coordinate labels. While the `InterpolateFromH5Profile` class is generalized to handle any set of density and coordinate names, this script is designed for the above conventions.

## Run the plot

Activate an environment containing `inversion_fbpic`, `h5py`, `numpy`, and
`matplotlib`. Run from this directory and pass paths to your profile files:

```bash
cd lpa/Simulation_FBPIC/demos/miscellaneous/demo_fluid_interp
python plot_profiles.py --files /path/to/profile.h5 --output density_profiles.png
```

Use `--x-mm` to choose the transverse position and `--angle` to select the
lineout angle relative to the `z_m` axis. For example:

```bash
python plot_profiles.py --files /path/to/profile.h5 --x-mm 2.0 --angle 45
```

You can pass multiple `--files` to be plotted side-by-side:

```bash
python plot_profiles.py --files /path/to/profile1.h5 /path/to/profile2.h5
```