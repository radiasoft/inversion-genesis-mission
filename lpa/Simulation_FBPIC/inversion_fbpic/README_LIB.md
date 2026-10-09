# `inversion_fbpic.lib` Guide

`inversion_fbpic.lib` is a configuration-oriented wrapper around FBPIC. It turns a laser-plasma interaction into validated, serializable components, then constructs and runs the corresponding FBPIC simulation. It centralizes the physical and operational choices that must remain consistent through three main classes:

- `Simulation` (paired with `SimulationHyperparameters`)
  - Uses constituent `_DensityProfile`s and `_LaserPulse`s, as well as the hyperparameters, to drive the base FBPIC solver
  - Many of the checks/helpers perform tasks that users of base FBPIC typically do manually
- `_DensityProfile` (with concrete implementations)
  - Contributes a collection of methods and fields that are common among implementations
  - Retains information, such as longitudinal extent and nominal plasma density, which `Simulation` uses to prepare an FBPIC run with minimal manual calculation
- `_LaserPulse` (also with concrete implementations)
  - Similar contributions as `_DensityProfile`
  - Makes circular/elliptical polarization trivial without LASY
  - Takes in total energy or amplitude $a_0$, derives the other for ease-of-use

## `lib` Code Map

- `serializable_config.py`: configuration protocol and registry management.
- `density_core.py`, `density_profiles.py`, and `density_modifiers.py`: common plasma behavior, public shapes, and composition.
- `laser.py`: laser validation and FBPIC profile construction. `GaussianLaserPulse` builds analytic FBPIC profiles; `LasyLaserPulse` wraps `utils.laser.HighOrderLasyLaser` (super-Gaussian transverse profile with Zernike aberrations), writes a LASY HDF5 file on rank zero during `Simulation.setup_simulation()`, and emits it through FBPIC's laser antenna.
- `simulation.py`: FBPIC assembly, execution, diagnostics, and completion hashing.
- `commented_yaml.py`: comment-aware YAML I/O.
- `_density_implementations/`, `_laser_implementations/`, and `_doc_management/`: density profile internals, laser pulse implementations, and generated editor-facing documentation.

# Usage

## Assemble And Run A Simulation

Build one `SimulationHyperparameters`, one or more density profiles, and one or more laser pulses, then pass them to `Simulation`. The wrapper validates this assembly before it constructs FBPIC.

```python
from pathlib import Path

from inversion_fbpic.lib.density_profiles import ExampleDensityProfile
from inversion_fbpic.lib.laser import GaussianLaserPulse
from inversion_fbpic.lib.simulation import Simulation, SimulationHyperparameters

hyperparameters = SimulationHyperparameters(
    zmin=-200e-6,
    zmax=0.0,
    rmax=200e-6,
    nz=1024,
    nr=300,
    nm=2,
    use_mpi=False,
    number_dumps=2,
)
density = ExampleDensityProfile(
    nominal_density=1.0e24,
    p_nz=2,
    p_nr=2,
    p_nt=4,
    length=1.0e-3,
)
laser = GaussianLaserPulse(
    energy=5.0,
    z0=-50e-6,
    wavelength=8.0e-7,
    tau_fwhm=3.8e-14,
    cep=0.0,
    waist=2.8e-5,
    focal_position=3.0e-3,
    polarization=0.0,
)

simulation = Simulation(elements=[hyperparameters, density, laser])
simulation.setup_simulation(working_directory=Path("runs/example"))
simulation.run_simulation()
```

`setup_simulation()` resolves runtime paths, creates the FBPIC object, adds particle species and lasers, configures the moving window, diagnostics, checkpoints, and restart behavior. `run_simulation()` advances the derived interaction time. Setup and execution have filesystem side effects; the wrapper writes only from rank zero when MPI is enabled.

## Choose Physical And Runtime Parameters

- `SimulationHyperparameters` controls the FBPIC grid, timestep, boundaries, boosted-frame settings, moving window, diagnostics, restart behavior, and random seed. Values use SI units. `nz` is checked for FFT-friendly factorization; `save_directory` is relative to `working_directory` unless absolute.
- Density profiles represent relative $n(z, r)$; `nominal_density` supplies the physical scale in $m^{-3}$. `get_z_extent()` contributes to the interaction length, and the `Simulation` instance will use this to determine the full interaction length. `get_r_extent()` or `p_rmax` bounds radial plasma loading.
- `species` and `ionization` select the ion model. Ionization `0` starts neutral; a negative value means fully ionized. Electron and ion diagnostic names and selection maps determine recorded particle populations. In laboratory-frame runs, diagnostic names must be unique across density components.
- Laser pulses require exactly one of energy or $a_0$; the wrapper derives and records the other. Concrete pulse types specify the envelope and optical parameters, then construct the FBPIC laser profile. `LasyLaserPulse` accepts energy only: its $a_0$ is measured numerically at focus when the LASY file is built and reported as `out_a0`, so it is `null` in configurations written before setup.
- Use density `plot()`, `plot_z_profile()`, and `plot_r_profile()` to inspect or record the plasma profile before launching a run. The standard plots report electron density by default.

## Use Configuration Files

Objects can be assembled in Python and serialized to record a run, or loaded from mappings, YAML/JSON text, individual files, or a directory of component files. Directory loading considers `.yaml`, `.json`, and compatible component types. Nested references resolve relative to their containing configuration file, keeping a run directory portable.

All configurations use a tagged representation:

```yaml
config_type: density_profile
subclass: sine_squared_bump
parameters:
  nominal_density: 1.0e24
  length: 5.0e-6
  p_nz: 1
  p_nr: 1
  p_nt: 1
```

`config_type` selects the component domain, `subclass` selects its model, and `parameters` contains user inputs rather than derived state. `commented_yaml.py` preserves user comments before PyYAML parsing and restores them, or injects docstring-derived descriptions, on output. Files therefore remain both machine-loadable and practical to inspect between runs.

### LASY Laser Pulses

`LasyLaserPulse` mirrors the inputs of `utils.laser.HighOrderLasyLaser` with the same defaults. It can only be emitted by a stationary antenna, so `z0_antenna` is required and `method`/`v_antenna` are fixed. The expensive LASY build runs once, inside `Simulation.setup_simulation()`, on MPI rank zero; the other ranks block until rank zero broadcasts the written path, or its error, so a bad configuration raises on every rank instead of hanging. FBPIC resets the LASY time axis to zero, so the peak intensity leaves the antenna at `t_start` plus the peak's delay from the start of the LASY time window; set `peak_delay_from_file_start` to fix that delay explicitly (it is `3 * tau_fwhm` for the default transform-limited pulse). Spectral-phase controls (`spectral_bandwidth`, `cep`, `gdd`/`tod`/`fod` or their duration-relative forms) are passed straight through to `HighOrderLasyLaser`, which validates them at build time.

```yaml
config_type: laser_pulse
subclass: lasy
parameters:
  energy: 2.5
  z0: -1.0e-4            # informational: nominal centroid at t = 0
  z0_antenna: 0.0
  t_start: 0.0
  wavelength: 8.0e-7
  tau_fwhm: 3.0e-14
  waist: 2.4e-5
  focal_position: 3.5e-3
  super_gaussian_order: 3.0
  zernike_coefficients:  # missing names default to 0.0
    coma_x: -0.25
  lasy_file: diags/lasy_laser   # relative to working_directory; written as diags/lasy_laser_00000.h5
```

## Demos

The [demos](../demos) directory contains runnable examples. Start with the core wrapper demos; the `miscellaneous/` entries are related density-modeling workflows rather than minimal simulation templates.

- [Building blocks](../demos/demo_building_blocks/README.md): generates a library of YAML components, selects active components in a directory, and runs `Simulation(elements=cfg_active)`.
- [Density profiles](../demos/demo_densities): generates and plots YAML, JSON, or HDF5 configurations for some concrete density-profile types.
- [Laser pulses](../demos/demo_lasers): generates and plots Gaussian-laser YAML, JSON, or HDF5 configurations for supported polarization variants.
- [Downramp simulation](../demos/demo_downramp_simulation/README.md): script-assembled hydrogen flattop/downramp LPA simulation, with recorded configuration, diagnostics, and density movie output.
- [Ionization simulation](../demos/demo_ionization_simulation/README.md): helium plasma with nitrogen doping, showing species-specific macroparticle settings and ionization injection.
- [LASY laser simulation](../demos/demo_lasy_laser_simulation/README.md): hydrogen flattop driven by an aberrated super-Gaussian `LasyLaserPulse`, showing the rank-0 LASY build, antenna emission, and the numerically measured `a0`.

The density demo uses [create_density_configs.py](../demos/demo_densities/create_density_configs.py)
and [plot_density_configs.py](../demos/demo_densities/plot_density_configs.py);
the laser demo uses [create_laser_configs.py](../demos/demo_lasers/create_laser_configs.py)
and [plot_laser_configs.py](../demos/demo_lasers/plot_laser_configs.py).
All four scripts accept `--format json` or `--format hdf5` to write or read
JSON or native HDF5 configs; omit the option for the existing YAML workflow.
All formats can coexist in each demo's `cfg` directory. JSON plots go to
`plots/json` and HDF5 plots to `plots/hdf5`, leaving YAML plots unchanged.
Re-running a generator explicitly
replaces its existing HDF5 configurations while preserving unrelated data.

## Skipping Runs if Complete

The wrapper hashes normalized hyperparameters, density profiles, and laser pulses. If the diagnostics directory contains the same hash, `setup_simulation()` and `run_simulation()` skip the matching completed configuration; component order does not affect the hash. The marker records configuration identity, not simulation-output validation.

# Architecture

## Component Model

The public configuration classes inherit shared domain behavior rather than repeating FBPIC setup logic:

- `SerializableConfig` provides JSON/YAML/HDF5 I/O, relative-path handling, example generation, and tagged type dispatch.
- `_DensityProfile` supplies particle-loading settings, species and ionization handling, diagnostic selections, plotting, plasma-wavelength calculation, and FBPIC species creation. Concrete profiles only define spatial shape and extent.
- `_DensityModifier` transforms a density function. `ModifiedDensityProfile` applies modifiers in order while inheriting loading and species settings from its base profile.
- `_LaserPulse` enforces the energy/$a_0$ contract, derives the companion value, and defines the interface for physical extents and FBPIC profile construction.
- `Simulation` owns component assembly and translates the configuration model into FBPIC calls.

## Policies Embedded In `Simulation`

This wrapper incorporates simulation policy as well as object wiring, things that are typically done manually using base FBPIC. It derives interaction length from the union of density-profile extents plus `right_buffer`; when `beta_window` is unset, it estimates the moving-window velocity from the first laser wavelength and summed on-axis plasma density; and it configures a boosted Galilean frame for all relevant components when `gamma_boost` is active.

## Registry And Serialization

`SerializableConfig` maintains a two-level registry: `config_type` maps to a domain base class, then `subclass` maps to a concrete class in that domain. Importing `inversion_fbpic.lib` imports the public domains and registers their concrete types, so `from_dict()`, `from_yaml()`, and `from_file()` reconstruct components without caller-specific dispatch code.

The type tags and serialized parameter names are part of the persisted run interface. Unknown, missing, or duplicate tags fail early; changing them can make existing configurations unloadable. Derived `attrs` fields are excluded from serialization, while input fields are retained for reconstruction.

### Native HDF5 configurations

`config.to_hdf5_file(path)` writes a native configuration into `/config` in an
HDF5 file. `SerializableConfig.from_file(path)` and `from_any(path)` recognize
`.h5` and `.hdf5` extensions, case-insensitively. To select a different group,
use `to_hdf5_file(path, group_path="/metadata/config")` and
`from_hdf5_file(path, group_path="/metadata/config")`.

For embedding in an already-open file, pass an empty `h5py.Group` to
`config.to_hdf5(group)`. Read it with `SerializableConfig.from_hdf5(group)` or
`from_any(group)`. These methods never close caller-owned handles. File writers
open in append mode and preserve unrelated datasets and metadata. Replacing a
previously written configuration requires `overwrite=True`; unrelated occupied
groups cannot be replaced, even with that flag. Encoding is staged before any
existing configuration is replaced, so unsupported values leave it intact.

The versioned schema retains `config_type`, `subclass`, and `parameters` as native
datasets/groups, rather than an opaque JSON document. Mapping keys are escaped
when necessary. Tags distinguish mappings, ordered sequences, empty containers,
and `None`; homogeneous numeric lists use native array datasets. `None` is stored
as a tagged null dataset (no shape or value), with a float64 placeholder dtype
regardless of the optional parameter's type. Values follow the existing
`to_dict()` semantics: NumPy arrays and tuples become lists, complex
values become real/imaginary pairs, and YAML comments and original NumPy dtypes
are not preserved. `include_nones` and deserialization `overrides` behave as in
the text serializers. Filesystem-backed group handles also supply `source_file`
and the anchor for portable relative input-file paths.
Writes to relative-open handles also work without an anchor; paths retain their
existing representation (absolute input paths stay absolute) instead of guessing
the file's original directory. For portable relative paths, open the file with an
absolute filename or wrap writes in
`SerializableConfig.resolving_paths_relative_to(directory)`.
Reads from relative-open handles still require that context with the original
file directory, so later working-directory changes cannot silently select the
wrong input files.
All serialized configs and examples include `git_hash`, cached from [git_hash.txt](git_hash.txt) when `serializable_config` is imported. Later commits or builds do not change a running process's provenance; restart to capture a new revision. Serialization never calls Git or preserves an incoming config's hash.

Builds/installs and Git hooks update the Git-ignored file, which is bundled in distributions. **Existing clones must re-run `pre-commit install`** to add the new stages. Amend/rebase are covered by `post-rewrite`; `git reset` is not. After resets or for uninstalled source use, run [tools/record_git_hash.py](../../../tools/record_git_hash.py) (or rebuild/reinstall) before importing. Runtime does not validate Git HEAD.

Unavailable provenance warns once at import and stays `null`, even with `include_nones=False`. It identifies committed code, not local edits; legacy configs remain supported.

The cached revision contributes to `Simulation.config_hash()`: a new process using a different revision invalidates completed-run skipping, while an existing process's hash stays stable.

## Internal Maintenance Layers

`_density_implementations/` isolates the numerical mechanics behind public profiles, including conical targets and HDF5 interpolation. This keeps `density_profiles.py` a stable catalogue for simulation assembly while model-specific code handles interpolation, composition, and validation.

`_laser_implementations/` fulfills a similar role in storing lengthy and specialized laser pulse implementations, while `laser.py` holds the base class and the basic profiles.

`_doc_management/` parses the `attrs` configuration hierarchy without importing it, merges inherited docstrings and `Args:` entries, and generates `.pyi` stubs with explicit keyword-only constructors. Pylance can therefore display inherited required parameters and documentation alongside subclass fields. Run `tools/sync_config_docstrings.py --check` to detect stub/documentation drift.

## Testing

`tests/test_lib/test_serializable_config.py`, `test_config_yaml.py`, and `test_config_yaml_limitations.py` cover tagged loading, path resolution, YAML comments, and serialization. `test_density.py`, `test_density_profiles.py`, and `test_laser.py` cover model validation and physical-profile behavior. `test_simulation.py` covers assembly, derived grid values, diagnostic periods, hashing, skip behavior, and a lightweight setup-and-step integration path. `tests/test_doc_management/` covers MRO documentation merging and generated stubs.