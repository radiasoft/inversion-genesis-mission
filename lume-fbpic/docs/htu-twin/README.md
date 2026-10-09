# HTU twin demo

Runs the HTU transport twin (`htu`) with a `lume-fbpic` run as its source: the LPA bunch goes in
at the start of the line and the twin tracks it through the magnets and screens. The twin and the
LPA archives can be served as EPICS PVs and watched in Phoebus; choosing another archive changes
the twin's source.

## How the pieces fit

An LPA run is an archive (`LUMEFBPICModel.archive()`): the config, the action values and, if saved,
the final particles. `twin/twin_stage.py` wraps the twin as a `StagedModel` stage that takes those final
particles as its source beam. `serving/serve.py` loads one or more archives, picks the active one
through `archive_selector.py`, and serves the archive's outputs and the twin's variables as PVs. The
displays in `phoebus/` are laid out for those PVs.

## Files

**`twin/`**

- `twin_stage.py`: `TwinStage` wraps the twin's model so that `StagedModel` can hand it the LPA stage's
  `final_particles`. It moves the bunch from the simulation's lab frame to a bunch frame, selects
  the particles (`uz_min`, `central_fraction`), optionally drifts it to the twin's source plane,
  replaces the twin's source beam, and makes the twin's `Source_*` PVs read-only readbacks of the
  bunch's moments. `build_chain()` returns `StagedModel([lpa_model, TwinStage(twin_model)])`.
- `lpa_to_twin.py`: a script that runs that chain once, from an archive or a finished run's
  diagnostics, and prints what the twin received and the fraction of the charge that reaches each
  screen.
- `initial_sample_archives.py`: turns the seven runs of the NERSC `initial_sample` dataset into
  archives. They hold the recorded 33 moment-descriptor values and no particles, so the twin gets a
  synthesized bunch from them.

**`serving/`**

- `serve.py`: serves archives as EPICS PVs through `lume-pva`. It loads each archive without
  running anything (`dummy_run=True`) and serves the model's variables as `<prefix><name>`: the
  outputs, and the inputs read-only unless `--include-inputs` is given. With `--twin` it also
  builds the chain and serves the twin's variables, which re-track on each put.
- `archive_selector.py`: `ArchiveSelector` holds several archives and presents the active one as a single
  LPA model, with one writable enum PV (`LPA_Archive`) to switch. A switch reaches the twin like
  a new LPA result. For archives without particles it can build an approximate bunch from the
  moment descriptor.

**`phoebus/`**

- Phoebus displays for the twin when it is fed by an LPA run: the twin's synoptic with an LPA strip
  and run selector, and `lpa_bunch.bob` showing the bunch, the 33 descriptor values and the run
  statistics. Two scripts generate them, so do not edit the `.bob` files; see its README.

## Install

You need Python 3.11 to 3.13, git, an MPI library (`fbpic` needs `mpi4py`) and a Phoebus 5.x build.
Run these from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e lpa/Simulation_FBPIC      # inversion_fbpic, with fbpic
pip install -e lume-fbpic                # lume_fbpic
pip install -e geecs-lume-twin           # the twin (htu), with Cheetah, lume-pva and p4p
```

The twin installs the `lume-pva` commit it pins (from GitHub), replacing any other `lume-pva` in the
environment.

## Run the twin

From the repository root, with the environment active:

```bash
cd lume-fbpic
python docs/htu-twin/twin/initial_sample_archives.py      # writes ./initial_sample_archives
python docs/htu-twin/serving/serve.py initial_sample_archives --twin --synthesize-bunch
```

Wait for `serving 131 variables` (a few seconds) and leave the terminal open. The first read of a
PV can show 0 for about a second, until the server's first update.

The writable LPA inputs (`laser_energy`, the Zernike coefficients, ...) are served read-only, so
all 131 variables are in the served config: the `lume-pva` the twin pins fails with
`KeyError: 'laser_energy'` if a variable of the model is missing from it. `--include-inputs` serves
them read-write instead. A put to one then changes the config only (the server runs with
`dummy_run=True`), and the outputs stay the recorded results.

### Check the PVs (optional)

From another terminal, with the EPICS command-line tools:

```bash
pvget HTU:SIM:LPA_Archive HTU:SIM:STATUS HTU:SIM:Source_Charge_pC
```

Expect `sim_0000`, `Idle` and about 511 pC. `HTU:SIM:STATUS` is `Idle` or `Simulating`, and is
`Simulating` while the twin re-tracks. If your Phoebus settings pin `epics_pva_addr_list`, add
`127.0.0.1` to that list.

### Open the display in Phoebus

From the Phoebus product directory, with an absolute path to the display:

```bash
java -jar target/product-<version>.jar -resource <repository>/lume-fbpic/docs/htu-twin/phoebus/htu_synoptic_lpa.bob
```

Start the server first, and restart Phoebus after a server restart. Phoebus restores its last
session, so older windows may open too: use the one with the LPA strip along the bottom.

- The "Run:" combo box in the strip selects the active archive. The values and the twin's source
  follow after the twin re-tracks, about 2 seconds.
- The "Twin:" LED shows `Idle` (green) or `Simulating` (amber).
- The `SRC` element opens `lpa_bunch.bob`. The other elements are the twin's own controls and
  camera views.

`phoebus/README.md` has the details: the display macros, regenerating the displays, and notes.

To stop, press Ctrl-C in the server terminal and close Phoebus.

### Options

- Archives are files or directories of `.h5` files; their file names (without `.h5`) are the options
  of the `HTU:SIM:LPA_Archive` selector PV. The first one is active at start.
- `--synthesize-bunch` builds an approximate bunch from each archive's recorded moment descriptor.
  It is needed for archives without final particles (such as the reconstructed `initial_sample`
  runs); leave it out for archives saved with `save_final_particles=True`.
- Other options: `--prefix` (default `HTU:SIM:` with `--twin`), `--screen-binning N` (camera images
  N times coarser, default 4; 1 keeps the twin's 1024 x 1024), `--bunch-particles N` (synthetic
  bunch size, default 20000), `--wait-for-puts` (acknowledge a put only after the twin re-tracks,
  for scan clients; the default acknowledges at once).
- Without `--twin`, `serve.py` serves just the archives' variables, by default with the prefix
  `LPA:SIM:`: `python docs/htu-twin/serving/serve.py run.h5 --prefix LPA:SIM:`.

### Run the chain once

To run the chain once without a server, from an archive saved with `save_final_particles=True` or
from the working directory of a finished `ionization_injection.py` run:

```bash
python docs/htu-twin/twin/lpa_to_twin.py --archive run.h5
python docs/htu-twin/twin/lpa_to_twin.py --run-dir <working directory>
```

## Limits

- The reconstructed `initial_sample` archives hold no particles, so their bunch is synthesized from
  the moment descriptor: an approximation, not the real bunch.
- Served values are recorded results; nothing re-runs the simulation when an input changes.
- The twin's magnets keep the settings they have for a 100 MeV beam, so a bunch of very different
  energy or large energy spread is mostly lost. The demo shows the hand-off, not matched optics.
