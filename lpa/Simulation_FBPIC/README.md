# Genesis FBPIC

A Python package for Genesis laser-plasma accelerator simulations with FBPIC, focused on ionization injection and electron beams at a few hundred MeV.

## Prerequisites

- Python 3.11 or higher
- Anaconda or Miniconda
- For Apple Silicon (M1/M2) Macs: Additional setup required (see Installation section)

## Installation

### Basic Installation

1. Create and activate a new conda environment:
   ```bash
   conda create --name FBPIC
   conda activate FBPIC
   ```

2. Install FBPIC following the [official documentation](https://fbpic.github.io/install/install_local.html)

   Note: For Apple Silicon (M1/M2) Macs, the `mkl` module is not available. Instead, install fftw:
   ```bash
   conda install -c conda-forge pyfftw
   ```

3. Install this package in editable mode:
   ```bash
   pip install -e .
   ```

### Development Setup

To set up the development environment with testing capabilities:

1. Activate your conda environment:
   ```bash
   conda activate FBPIC
   ```

2. Install pytest and setuptools in your conda environment:
   ```bash
   conda install pytest setuptools
   ```

3. Install the package in development mode with testing dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run the tests:
   ```bash
   pytest tests -v
   ```

The `-v` flag provides verbose output showing each test case.

## Usage

### Density Profiles

The package provides several density profile functions:

- `build_gaussian_profile`: Creates a Gaussian density profile
- `build_gaussian_plus_triangle_z_density_function`: Creates a combined Gaussian and triangular density profile

Example usage:
```python
from inversion_fbpic.density_profiles.downramp_injection import build_gaussian_profile

# Create a Gaussian profile
density = build_gaussian_profile(sigma=3e-6, center_location=10e-6)
```

### Running FBPIC

FBPIC can be run from Python once the environment is set up. Example simulation workflows are in `demos/demo_ionization_simulation/` and `demos/demo_downramp_simulation/`.


### Visualization with openPMD

For visualizing simulation outputs, use the openPMD notebook tool:

1. Install following the instructions at the bottom of the [FBPIC How to Run page](https://fbpic.github.io/how_to_run.html#visualizing-the-simulation-results)

2. Launch the visualization tool:
   ```bash
   openPMD_notebook
   ```

3. If the default diagnostic folder name was changed in the FBPIC launch script, edit the following cell in the generated notebook accordingly:
   ```python
   # Replace the string below, to point to your data
   ts = OpenPMDTimeSeries('./diags/hdf5/')
   ```
   *Note*: The root directory here will be the directory where the openPMD_notebook was launched.


### Visualization with CLI

The package includes analysis and visualization commands implemented in `inversion_fbpic/scripts/`, backed by helpers in `inversion_fbpic/utils/`.

After installing the package with `pip install -e .`, the commands below can be called from anywhere. They support flexible command-line arguments with both long and short flag options.

   **Command-Line Scripts:**
   
   After installation, scripts can be called directly by name. Common shorthand flags include:
   - `-d, --diag-folder`: Path to diagnostics directory
   - `-i, --iteration-number`: Iteration number to load
   - `-s, --species`: Particle species name
   - `-b, --bins`: Number of bins for calculations
   
   **Example usage:**
   ```bash
   plot-ebeam-analysis -d sim0077/lab_diags/hdf5 -s n_elec -i -1
   ```
   
   This command analyzes the electron beam from the specified diagnostics folder, using species `n_elec` and loading the final iteration (`-1`).
   
   **Available scripts include:**
   - `plot-ebeam-analysis`: Outputs beam analysis for a single simulation
   - `visualize-ebeamparams-vs-scan`: Analyzes beam parameter measurements in a 1D parameter scan
   - `energy-at-peak-current`: Calculates energy at peak current location
   - `slideshow-from-npy`: Creates PNG images from .npy files for charge density movies
   - `extract-hdf5-field` and `extract-hdf5-particles`: Extract field and particle data from OpenPMD HDF5 diagnostics
   - `fbpic-calc-nr`: Calculates FBPIC resolution requirements
   - `analyze-laser-evolution`: Laser energy and `a0` tracking from OpenPMD field diagnostics
   - `lasy-propagation`: Propagates and analyzes a LASY laser pulse
   
   For detailed usage information and all available options for each script, see the documentation at the top of each script file or use the `--help` flag:
   ```bash
   plot-ebeam-analysis --help
   ```

      #### Beam Dataset and Phase-Space Analysis

      `build-dataset` converts a collection of FBPIC runs into a JSON dataset for
      downstream analysis. Its required positional argument is a directory that
      contains `sim_*` subdirectories. Each run must contain `input.ini` with a
      `[PhysicalParameters]` section and particle diagnostics in
      `lab_diags/hdf5/`.

      ```bash
      build-dataset /path/to/raw_runs --output /path/to/dataset.json \
         --species nitrogen_electrons --uz-min 30 --central-fraction 0.95
      ```

      The output records each run's physical input parameters and a weighted
      six-dimensional electron-beam descriptor derived from the final diagnostic.
      The `runs/initial_sample/` directory contains an example run configuration
      and a seven-run sample dataset: one nominal case plus six single-parameter
      variations.

      `plot-phase-space-moments` creates a phase-space figure from a single
      openPMD particle diagnostic. It compares weighted particle projections with
      moment-based density models; `--all` renders the full triangular set of 1D
      and 2D projections.

      ```bash
      plot-phase-space-moments /path/to/lab_diags/hdf5/data00000049.h5 \
         --species nitrogen_electrons --uz-min 30 --all \
         --output phase_space_moments.png
      ```

   #### Laser evolution analysis (`analyze-laser-evolution`)

   Tracks laser energy and engineering ``a0`` along the propagation axis by reading
   **E** and **B** fields from an OpenPMD HDF5 time series (laser-centred bandpass,
   cylindrical energy integration, analytic-signal envelope). Writes a JSON results
   file and a summary ``energy`` / ``a0`` vs. ``z`` plot.

   ```bash
   # Default: ./diags/hdf5, saves laser_evolution_results.json and laser_evolution_plot.png
   analyze-laser-evolution

   # Custom diagnostics directory and output paths
   analyze-laser-evolution -d /path/to/sim/lab_diags/hdf5 \
       --results-file my_results.json --output-plot my_plot.png

   # Per-iteration imshow panels (optional log colour scale)
   analyze-laser-evolution -d /path/to/sim/lab_diags/hdf5 --show-plots --log

   # Re-plot summary from a saved JSON without re-running the analysis
   analyze-laser-evolution -j my_results.json --output-plot my_plot.png
   ```

   Useful options: ``-d`` series path, ``-w`` laser wavelength (default 0.8 µm),
   ``--band-half-width-frac``, ``--rmax-window``, ``--on-axis-rmax``,
   ``--energy-threshold``, ``--start-iteration`` / ``--end-iteration``,
   ``--show-plots``, ``--log``, ``-j`` / ``--from-json``.

### Visualization with Python

The modules `inversion_fbpic.utils.plotting` and `inversion_fbpic.utils.make_movie` let you render every timestep to PNG stills and stitch them into an MP4. `plot_from_hdf5_series` reads OpenPMD diagnostics directly. Pre-extracted `.npy` field arrays (e.g. from `extract-hdf5-field`) can also be plotted with `plot_from_npy`, which accepts the same color-scale and layout options as the HDF5 workflow below, though with limited quantities available for plotting. `make_movie` calls the system `ffmpeg`, which must be on your `PATH`; importing from `make_movie` raises an `RuntimeError` if `ffmpeg` is not installed.

Key options on `plot_from_hdf5_series`:
- `field_name` / `component`: scalar fields (e.g. `"rho"`), vector components (`"E"` + `"z"`), or `"eme"` for electromagnetic energy density. For `"eme"`, set `field_name=None`; the plotter loads all **E** and **B** components and applies a cosine-squared bandpass (`em_bandpass_filter`) to produce six sub-panels per frame. `rho` is automatically scaled to -e/cm^3 and `eme` has units of J/um^3.
- `vminmax`: fixed `(vmin, vmax)`, `"even"` for a symmetric color scale around zero, or `None` to auto-scale each frame.
- `scale` / `cmap`: `"linear"` or `"log"` color norms (a log-linear distribution is used for signed data in the case of `log`); any Matplotlib colormap name.
- `rmax`: optional radial crop in meters (display limits only).

Example: build movies for charge density, one **E** component, and band-filtered EM energy:

```python
from pathlib import Path

from inversion_fbpic.utils.make_movie import make_movie
from inversion_fbpic.utils.plotting import plot_from_hdf5_series

DIAGS_DIR = Path("lab_diags")   # parent of the hdf5/ folder (or the hdf5 folder itself)
PLOTS_DIR = Path("plots")
RMAX = 50e-6                    # display |r| <= RMAX [m]; use None for full grid

# (field_name, component, extra kwargs for plot_from_hdf5_series)
specs = [
      ("rho",  None,    dict(vminmax=(1e14, 2e18),    cmap="magma",  scale="log")),
      ("E",    "z",     dict(vminmax="even",          cmap="bwr",    scale="linear")),
      (None,   "eme",   dict(vminmax=(1e-15, 1e-5),   cmap="magma",  scale="log")),
]

for field_name, component, kwargs in specs:
      stills_dir, prefix = plot_from_hdf5_series(
         series_path=DIAGS_DIR / "hdf5",
         save_path=PLOTS_DIR / "stills",
         field_name=field_name,
         component=component,
         rmax=RMAX,
         font_size=16,
         **kwargs,
      )
      movie_path = make_movie(
         images_dir=stills_dir,
         image_prefix=prefix,
         filename=prefix,
         framerate=10,
      )
      print(f"Wrote {movie_path}")
```

To render additional vector components (e.g. all **E**/**B** axes), add more `(field_name, component, kwargs)` tuples—`plot_from_hdf5_series` returns `(save_dir, file_prefix)` so `make_movie` can glob `file_prefix_*.png` in that directory. After the movie is created, you can delete the intermediate PNGs if you only need the MP4.

### Troubleshooting

If you are having issues with getting openPMD to successfully read data, can try to enforce that openpmd is of the following version:
```shell
conda install -c conda-forge "openpmd-api=0.16.1" "openpmd-viewer=1.11.0”
```


