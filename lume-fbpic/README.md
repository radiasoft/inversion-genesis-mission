# lume-fbpic

A [LUME](https://github.com/lume-science) model for laser-plasma accelerator (LPA) simulations,
wrapping [`inversion_fbpic`](../lpa/Simulation_FBPIC) the way
[`lume-cheetah`](https://github.com/lume-science/lume-cheetah) wraps Cheetah. With it you can:

- **run** an fbpic simulation through `get()` / `set()`;
- **archive** it, **load** it back, and **serve** its outputs as EPICS PVs (via `lume-pva`);
- use its bunch as the **source of the HTU transport twin** (`geecs-lume-twin`), with a selector
  that switches between several archived runs;
- turn the **existing NERSC runs** (`runs/initial_sample`) into archives that serve the same way.

## Install

```bash
pip install -e ../lpa/Simulation_FBPIC      # inversion_fbpic, which this package wraps
pip install -e .                            # lume_fbpic
pip install -e .[dev]                       # also pytest
```

`lume-pva` (for `lume-fbpic-serve`) and `geecs-lume-twin` (for the twin parts) are not dependencies;
they are imported only when used. The twin is not on PyPI: put its checkout on `PYTHONPATH`.

## Quick start

An `FBPICSimulator` holds the config (hyperparameters, laser, density profiles); the scripts in
`docs/examples/` build one for each case.

```python
from lume_fbpic.model import LUMEFBPICModel

model = LUMEFBPICModel.from_simulator(simulator)
model.set({"laser_energy": 2.5})          # applies the value, then runs fbpic
model.get(["charge_pc", "energy_mean_mev"])

model.archive("run.h5", save_final_particles=True)
model = LUMEFBPICModel.from_archive("run.h5")      # actions come back with it
```

- `set()` runs the simulation, as `lume-cheetah` and `lume-impact` do. `dummy_run=True` makes it
  only update parameters.
- The **actions** are the model's inputs and outputs: laser and density fields, Zernike
  coefficients, the dopant fraction, run statistics, the 33-scalar moment descriptor and the final
  particles.
- An archive holds the config, the actions, the input values at execution, the output values and
  optionally the final particles (as a `ParticleGroup`).

Serve the outputs as PVs, and optionally feed the HTU twin (see `docs/phoebus/README.md`):

```bash
lume-fbpic-serve run.h5 --prefix LPA:SIM:
lume-fbpic-serve <archive dir> --twin --synthesize-bunch      # needs geecs-lume-twin on PYTHONPATH
```

## Layout

| Path | Purpose |
|------|---------|
| `lume_fbpic/model.py` | `LUMEFBPICModel`: the LUME model, with `archive()` / `from_archive()` |
| `lume_fbpic/simulator.py` | `FBPICSimulator`: wraps `inversion_fbpic`'s `Simulation`; archive I/O, `load_results()` |
| `lume_fbpic/actions.py` | The input and output actions; `make_actions()` and `make_descriptor_actions()` |
| `lume_fbpic/density_profiles.py` | Density profiles for the examples (`LinearRampFlattop`, `GeneralizedGaussianProfile`) |
| `lume_fbpic/handoff.py` | Final particles to a beam-line bunch; a bunch from a moment descriptor; bunch moments |
| `lume_fbpic/twin.py` | `TwinStage` and `build_chain()`: the HTU twin with the LPA bunch as its source |
| `lume_fbpic/selector.py` | `ArchiveSelector`: several archived runs behind one enum variable (`LPA_Archive`) |
| `lume_fbpic/serve.py` | `lume-fbpic-serve`: serve archives as EPICS PVs, optionally with the twin |
| `docs/examples/` | Runnable examples: LWFA, a parametric scan, ionization injection, building archives from the NERSC dataset, the LPA-to-twin chain |
| `docs/phoebus/` | Phoebus displays for the twin chain and the scripts that generate them (own README) |
| `tests/` | Tests for this package |

Run the examples from this directory so their output lands in `diags/`.

## Tests

```bash
pytest                                                      # most tests
PYTHONPATH=<geecs-lume-twin checkout> pytest                # also the twin tests
```

The twin tests need `htu` importable and `lume-pva` installed; without them they are skipped.

## Status and limits

- Archives reconstructed from `runs/initial_sample` hold the recorded outputs but no particles. To
  feed the twin, a bunch is built from the moment descriptor (`--synthesize-bunch`): an
  approximation, not the real bunch.
- Served values are recorded results; nothing re-runs the simulation when an input changes.
- `lume-pva` currently runs against `lume-base` 0.5.0, although it declares `>= 0.6.0`.

## Relationship to `inversion_fbpic`

This package depends on `inversion_fbpic` (from the sibling
[`Simulation_FBPIC`](../lpa/Simulation_FBPIC) directory) as an external dependency and does not
modify or vendor that code.
