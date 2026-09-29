# Single-Simulation Demo

This example builds and runs an FBPIC LPA simulation with
`inversion_fbpic.lib`. Here the YAML configs are saved for reference and later re-running without needing the original script.

The setup uses an 800 nm Gaussian laser (4.5 J, 40 fs, 30 um waist) in a
helium flattop plasma with 3% nitrogen doping. The flattop
length is calculated from the requested 430 MeV target energy and the laser's
estimated `a0`.

Furthermore, since we are most interested in the statistics of the injected electrons, primarily sourced from nitrogen, the particles-per-cell is increased for this species and decreased for the helium species.

## File

| File | Purpose |
|------|---------|
| `run_simulation.py` | Defines the laser, density profiles, and simulation hyperparameters; launches FBPIC |

## Run the demo

Activate an environment containing `inversion_fbpic`, FBPIC, MPI, and the
plotting dependencies. Run from this directory so generated files stay with
the demo:

```bash
cd lpa/Simulation_FBPIC/demos/demo_ionization_simulation
python run_simulation.py
```

The script detects whether it was launched with multiple MPI ranks. For a
multi-rank run, use your MPI launcher, for example:

```bash
mpirun -n 2 python run_simulation.py
```

## Outputs

The run writes the following artifacts in the demo directory:

- `cfgs/` contains YAML records of the hyperparameters, calculated grid
   parameters, laser, and both density profiles.
- `plots/density_profiles.png` shows the longitudinal flattop and downramp
   density profiles.
- `diags/` contains the FBPIC diagnostic dumps.
- `plots/stills/` and `plots/rho.mp4` contain charge-density frames and the
   generated rho movie.

## Customize the setup

Edit the constants at the start of `run_simulation.py` to change the target
energy, laser, plasma density, or dopant fraction. The script recalculates the
flattop length whenever the target energy, laser, or flattop density changes.

The simulation saves a hash representation of itself; if it has already run successfully, it will not attempt to re-run if only non-simulation-related items are altered (e.g., if post-processing is modified within the same script).
