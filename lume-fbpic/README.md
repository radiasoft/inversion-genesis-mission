# lume-fbpic

A [LUME](https://github.com/lume-science) model for laser-plasma accelerator (LPA) simulations,
wrapping [`inversion_fbpic`](../lpa/Simulation_FBPIC) the way
[`lume-cheetah`](https://github.com/lume-science/lume-cheetah) wraps Cheetah. With it you can:

- **run** an fbpic simulation through `get()` / `set()`;
- **archive** it, **load** it back, and **serve** its outputs as EPICS PVs (via `lume-pva`);
- use its bunch as the **source of the HTU transport twin**, with a selector
  that switches between several archived runs;
- turn the **existing NERSC runs** (`runs/initial_sample`) into archives that serve the same way.

## Install

```bash
pip install -e ../lpa/Simulation_FBPIC      # inversion_fbpic, which this package wraps
pip install -e .                            # lume_fbpic
pip install -e .[dev]                       # also pytest
```

`lume-pva` (for `docs/htu-twin/serving/serve.py`) is not a dependency; it is imported only when used.

## Quick start

An `FBPICSimulator` holds the config (hyperparameters, laser, density profiles); the scripts in
`docs/examples/` build one for each case.

```python
from lume_fbpic.actions import FinalParticlesAction, LaserFieldAction, StatAction
from lume_fbpic.model import LUMEFBPICModel

actions = [   # the model's inputs and outputs; docs/examples build the lists for each case
    LaserFieldAction(name="laser_energy", field_name="energy", unit="J"),
    StatAction(name="charge_pc", stat_name="charge_pc", unit="pC", read_only=True),
    StatAction(name="energy_mean_mev", stat_name="energy_mean_mev", unit="MeV", read_only=True),
    FinalParticlesAction(name="final_particles", read_only=True),
]
model = LUMEFBPICModel(simulator, actions)
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

To serve the outputs as PVs, or to feed the HTU twin, see `docs/htu-twin/README.md`.

## Layout

| Path | Purpose |
|------|---------|
| `lume_fbpic/model.py` | `LUMEFBPICModel`: the LUME model, with `archive()` / `from_archive()` |
| `lume_fbpic/simulator.py` | `BaseSimulator` (lifecycle, results, archive I/O), `FBPICSimulator` (LWFA, wraps `inversion_fbpic`'s `Simulation`) and `PWFASimulator` (beam-driven, built directly on fbpic) |
| `lume_fbpic/pwfa_config.py` | PWFA configs: `PWFAGrid` and the electron bunches (`FlatTopBunch`, `GaussianBunch`) |
| `lume_fbpic/actions.py` | The input and output actions; `make_descriptor_actions()` and `make_pwfa_actions()` |
| `lume_fbpic/density_profiles.py` | Density profiles for the examples (`LinearRampFlattop`, `GeneralizedGaussianProfile`, `UpDownRampProfile`) |
| `docs/examples/` | runnable examples: LWFA, two beam-driven PWFA setups (FACET-II type, and CLARA FEBE energy doubling), ionization injection |
| `docs/htu-twin/` | The HTU twin demo, with the LPA archive as the twin's source (own README), in three parts: |
| &nbsp;&nbsp;`twin/` | `twin.py`: `TwinStage` and `build_chain()`, the HTU twin with the LPA bunch as its source, and its helpers (bunch moments, ballistic drift); `lpa_to_twin.py`: an example running the LPA-to-twin chain; `initial_sample_archives.py`: builds the archives (from the NERSC dataset) that the twin demo serves |
| &nbsp;&nbsp;`serving/` | `serve.py`: serve archives as EPICS PVs, optionally with the twin; `selector.py`: `ArchiveSelector`, several archived runs behind one enum variable (`LPA_Archive`) |
| &nbsp;&nbsp;`phoebus/` | Phoebus displays for the twin chain and the scripts that generate them (own README) |
| `tests/` | Tests for this package; `downramp_actions.py` is the HTU downramp action set their fixtures use |

Run the examples from this directory so their output lands in `diags/`.

## Tests

```bash
pytest
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
