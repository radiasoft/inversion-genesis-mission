# LASY Laser Simulation Demo

This example builds and runs an FBPIC LPA simulation driven by a `LasyLaserPulse`
with `inversion_fbpic.lib`. The YAML configs are saved for reference and later
re-running without the original script.

Where the other simulation demos use an analytic `GaussianLaserPulse`, this one
uses LASY to construct an 800 nm, 4.5 J, 40 fs pulse with a super-Gaussian
(order 3) transverse profile and a few Zernike phase aberrations (astigmatism,
coma, spherical) defined at focus. The pulse is back-propagated 3 mm to the
simulation start plane, written to a LASY HDF5 file, and emitted into the box by
a stationary laser antenna. The plasma is a hydrogen flattop whose length is
calculated from the requested 430 MeV target energy and the laser's `a0`.

Two things differ from the Gaussian demos:

- `a0` is **measured**, not derived analytically. `LasyLaserPulse` takes
  `energy` only; when the pulse is prepared, the LASY field is propagated to
  focus and its peak normalized vector potential is read off and recorded as
  `out_a0`. The script calls `laser.prepare(...)` up front so `a0` is available
  to size the plasma, and prints it next to the `a0` of an ideal Gaussian with
  the same energy. `Simulation.setup_simulation()` would otherwise run the same
  build itself.
- The laser is emitted by an antenna (`method="antenna"` is forced). FBPIC
  resets the LASY time axis to zero, so the peak leaves the antenna at
  `t_start + 3 * tau_fwhm`. The script places the antenna 5 um inside the front
  of the box and sizes the window so the full pulse fits behind it.

The LASY build takes roughly 20 s on the default (600, 900) grid with five
azimuthal modes and runs on MPI rank 0 only; other ranks wait at a barrier.

## File

| File | Purpose |
|------|---------|
| `run_simulation.py` | Defines the LASY laser, a reference Gaussian, the density profile, and the simulation hyperparameters; launches FBPIC |

## Run the demo

Activate an environment containing `inversion_fbpic`, FBPIC, LASY, MPI, and the
plotting dependencies. Run from this directory so generated files stay with the
demo:

```bash
cd lpa/Simulation_FBPIC/demos/demo_lasy_laser_simulation
python run_simulation.py
```

The script detects whether it was launched with multiple MPI ranks. For a
multi-rank run, use your MPI launcher, for example:

```bash
mpirun -n 2 python run_simulation.py
```

## Outputs

The run writes the following artifacts in the demo directory:

- `diags/lasy_laser_00000.h5` is the LASY file FBPIC emits from the antenna.
- `cfgs/` contains YAML records of the hyperparameters, calculated grid
  parameters, the LASY laser (including the measured `out_a0`), the reference
  Gaussian laser, and the density profile.
- `plots/lasy_laser.png` shows the start-plane on-axis envelope of the LASY
  pulse and a face-on map of its field amplitude with the polarization marked.
- `plots/laser_comparison.png` overlays the on-axis envelope of the LASY pulse
  at the start plane (3 mm before focus, so its amplitude is below its focus
  `a0`) with the focal envelope of the ideal Gaussian of equal energy.
- `plots/density_profiles.png` shows the longitudinal density profile.
- `diags/` contains the FBPIC diagnostic dumps.
- `plots/stills/`, `plots/rho.mp4`, and `plots/eme.mp4` contain field frames
  and the generated movies; `plots/beam_analysis.png` and
  `plots/beam_summary.txt` summarize the accelerated electrons.

## Customize the setup

Edit the constants at the start of `run_simulation.py` to change the target
energy, laser energy, spot size, super-Gaussian order, Zernike coefficients, or
plasma density. Zernike names are the keys of
`inversion_fbpic.utils.laser.ZERNIKE_OSA_INDICES`; unknown names are rejected
and missing names default to zero. Setting `super_gaussian_order=2.0` with no
Zernike terms reproduces a Gaussian, and its measured `a0` matches the analytic
value of `GaussianLaserPulse` to well under a percent.

`center_and_remove_tilt` is switched off in this demo. That option re-centers
the start-plane fluence and removes the mean tilt by interpolating the field on
a polar grid, which is not accurate enough for aberrated beams: it changes the
measured focus `a0` by several percent even though a pure phase at focus cannot
change the peak amplitude there. The coma-induced centroid offset at the start
plane is only about 1 um here, so re-centering is not needed.

The simulation saves a hash representation of itself; if it has already run
successfully, it will not attempt to re-run if only non-simulation-related items
are altered (e.g., if post-processing is modified within the same script). The
measured `out_a0` is excluded from that hash, so preparing the laser does not
change it.
