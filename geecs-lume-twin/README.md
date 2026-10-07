# geecs-lume-twin

Virtual accelerator / digital twin of the **BELLA HTU transport line**,
served as EPICS PVs. Built on the [LUME](https://github.com/lume-science)
stack — [Cheetah](https://github.com/desy-ml/cheetah) does the particle
tracking, [lume-pva](https://github.com/lume-science/lume-pva) serves the
model over CA + PVA — so every existing EPICS client (Phoebus, ophyd-async,
Bluesky, OSPREY, camonitor) talks to the twin exactly as it talks to the
real machine.

Standalone peer of [GEECS-Plugins](https://github.com/GEECS-BELLA/GEECS-Plugins):
imports nothing from the monorepo; couples to GEECS only through the PV
naming convention (sim namespace `HTU:SIM:`). Integration edges (ophyd
bridge devices, sim queueserver/Tiled profiles) belong in GEECS-Plugins,
not here.

**Agents: read [CLAUDE.md](CLAUDE.md) first** — it carries the pinned
physics rules and conventions that are not re-derivable from the code.

## Quickstart

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -e .
./scripts/postinstall_fixes.sh          # REQUIRED after every (re)install
./.venv/bin/python tests/validate_physics.py   # 35 checks, all must pass
./.venv/bin/htu-twin-serve              # the twin is now on the network
```

Phoebus: open `display/htu_synoptic.bob`. All PVs are `pva://HTU:SIM:*`.
On the serving machine no settings are needed **unless** your
`phoebus_settings.ini` pins `epics_pva_addr_list` — then add `127.0.0.1`
to that list. Restart Phoebus after any server restart (stuck-channel
slow retry).

## What the twin models

```
LPA source ──► PMQ triplet ──► chicane (coil amps) ──► EMQ triplet ──► screens
(explicit     (fixed magnets,   (4 bends, operator     (measured        (camera-true
 unknowns      bore apertures)   R56[um] formula)       current fits)    geometry)
 as PVs)                                    │
                       S1..S4 steering      └─► Dutch MagSpec (COSY maps,
                       (coil amps)               on/off + B knob) ──► A-line
```

- **Physics rule (pinned):** no single-reference-energy linear optics.
  Every element tracks `drift_kick_drift` (Cheetah's Bmad-X integrator,
  exact per-particle energy dependence). LPA energy spread is first-class.
- **Machine layer:** PVs are in operator units (coil amps, R56 in um,
  B in T). `htu/calibrations.py` holds the measured conversions
  (EMQ testing-report fits, kicker integrated fields, chicane formula) —
  invertible, so readbacks report machine units.
- **Authoritative state:** `(machine_state dict, source_params)` on the
  simulator; `_apply_machine_state()` re-derives every element parameter
  at the current beam energy. Consequence: magnet PVs hold their currents
  through source-energy changes — energy jitter shifts the optics, as on
  the real machine.
- **Source as explicit unknowns:** energy, spread, charge, per-plane
  Twiss (beta/alpha/normalized emittance), centroid and pointing are
  input PVs (`Source_*`). The estimator that fits them from measurements
  is future work (see docs/vision.md).
- **MagSpec:** replays the 2013 Nakamura characterization — trajectory
  table + 1600 third-order COSY maps (bundled, `htu/data/dutch/`) applied
  per-particle. Off = drift; on = spectra on the front/side screens and
  the downstream line goes dark. See `docs/magspec-design.md` for the
  full file-format decode and conventions.
- **Apertures:** PMQ/VISA quad bores (real data) + the 20 mm-ID beam pipe
  from 10 cm upstream of EMQ1 to the magspec entrance. Steering into the
  wall loses the beam; chromatic tails scrape realistically. The
  mid-chicane collimating slit (two independent horizontal tungsten jaws
  at the ChicaneSlit camera, ~6 mm dispersion per unit dp/p at R56=200um)
  makes energy selection an operator knob.

## PV surface (prefix `HTU:SIM:`)

| PV | Kind | Meaning |
|---|---|---|
| `EMQ{1H,2V,3H}_Current` | rw, A | EMQ coil currents (measured calibration) |
| `S{1-4}{H,V}_Current`, `VS{1-8}{H,V}_Current` | rw, A | steering coil currents (transport line / VISA undulator sections) |
| `ChicaneDipole_Current` | rw, A | chicane coil current (measured B-I table, ~18.1 mT/A; 2.169 A = R56 200 um) |
| `Chicane_R56_um` | ro, um | derived R56 readback at the 100 MeV ops convention |
| `ChicaneSlit_Jaw{1,2}_mm` | rw, mm | energy-slit jaws, LAB-frame positions relative to the straight (R56=0) axis; Jaw1 inserts from +x (cuts low energy), Jaw2 from -x |
| `PMQTriplet_{X,Y,Z}_mm` | rw, mm | hexapod/rail: rigid-unit triplet motion (X/Y transverse, Z = downstream) |
| `PMQ{1V,2H,3V}_Tilt_mrad` | rw, mrad | per-magnet rotation about the beam axis (skew coupling) |
| `VisaBump_{X,Y}_mm`, `VisaBump_{Xp,Yp}_mrad` | rw | composite views over S3+S4: orthogonal position/angle bumps at the VISA entrance (readbacks derive from raw currents; excluded from config snapshots) |
| `MagSpec_On` | rw, bool | magspec mode: off = transport, on = characterize |
| `MagSpec_B_T` | rw, T | magspec peak field (default 0.825) |
| `Source_Energy_MeV`, `Source_EnergySpread_pct`, `Source_Charge_pC` | rw | the (unknown) source, as explicit inputs; plus `Source_NumParticles` (macro-particle count: statistics vs speed) |
| `Source_{BetaX,BetaY}_mm`, `Source_{AlphaX,AlphaY}`, `Source_{NormEmitX,NormEmitY}_um` | rw | per-plane source Twiss; emittance is NORMALIZED mm·mrad (geometric = emit_n/gamma) |
| `Source_{X,Y}_um`, `Source_{Xp,Yp}_mrad` | rw | source centroid offsets and pointing angles |
| `{TCPhosphor,ChicaneSlit,DCPhosphor,Phosphor1,UC_ALineEbeam1,UC_ALineEBeam2,UC_ALineEBeam3}_image` | ro | camera images, experimental geometry (1024x1024, 7 um/px default; ECS-dump loader in `htu/screens.py`) |
| `MagSpec_{FrontScreen,SideScreen}_image` | ro | magspec screens in screen-mm space |

Puts are `PutMode.Complete` — they ACK only after the re-track (~0.2 s
full line), so scan clients can set-and-wait.

## Configurations

Code defaults = the pristine physics baseline; matched-to-experiment
states live in `configs/` as named, git-tracked YAML files (flat
`{pv_suffix: value}` maps — diff two to see machine/source drift between
sessions; doctrine in `configs/README.md`).

```bash
./.venv/bin/htu-twin-config save configs/mysession.yaml --note "matched 08-25"
./.venv/bin/htu-twin-config load configs/mysession.yaml       # apply to a running server
./.venv/bin/htu-twin-serve --config configs/mysession.yaml    # boot with it as defaults
```

## Repo map

```
htu/
  calibrations.py   measured unit conversions (provenance in comments)
  lattice.py        the beamline (geometry from the prior twin repos)
  magspec.py        Dutch magnet map element + calibration parser
  model.py          LUMEModel: machine-state architecture, all PVs
  screens.py        camera geometry incl. GEECS ECS-dump parser
  source.py         LPA source parameterization (per-plane Twiss + pointing)
  serve.py          entry point (htu-twin-serve, --config for boot defaults)
  config_tool.py    htu-twin-config: save/load PV snapshots (YAML)
  data/dutch/       trajectory table, 1600 COSY maps, camera calib
configs/
  baseline.yaml     code-default snapshot; named session configs live here
display/
  generate_synoptic.py  REGENERATE htu_synoptic.bob after lattice changes
  *.bob                 synoptic + per-element control popups + camera view
docs/
  magspec-design.md     the full HTU magspec decode (formats, conventions)
  magspec-porting-guide.md  REPLICABLE playbook: port another magspec (HTT, PW)
  vision.md             where this is going (VA game, surrogate, estimator)
tests/
  validate_physics.py   the acceptance checks — run after ANY physics change
  test_config_roundtrip.py  headless config save/load round-trip
scripts/
  postinstall_fixes.sh  venv patches for upstream bugs — after every install
```

## Known upstream bugs (worked around here)

1. **lume-cheetah 0.1.0** assigns raw put values onto torch buffers
   (rejected for plain floats) — worked around by the machine-layer
   variables in `htu/model.py` (they always assign tensors).
2. **lume-pva Runner dies on a model exception during a put** (no
   per-put error isolation) — mitigated by transactional source puts;
   any new `_set` must not commit state before it can still raise.
3. **lume-pva writes NTNDArray `dimension[]` in numpy order** (NT spec:
   `dimension[0]` = fastest axis) — non-square images scramble in
   compliant clients. Patched by `scripts/postinstall_fixes.sh`.
4. **pcaspy 0.8.1 macOS arm64 wheel** ships a broken rpath — fixed by
   `scripts/postinstall_fixes.sh`.

## Provenance

- Lattice geometry + calibrations: `BLAST-AI-ML/bella_htu_digital_twin`
  (BSD-3-LBNL) and the local `BeamTransportDigitalTwin` — numbers, not
  architecture.
- MagSpec: K. Nakamura, "Electron Spectrometer Using the EUTERPE Magnet"
  (2013 tech note) + 2018 Moore Dutch 1 setup slides (copies in repo
  root, gitignored) + the calibration files bundled in `htu/data/dutch/`.

## Validated against (2026-08-25)

Run `tests/validate_physics.py` — every claim below is a check in it:
pencil beams land at the magspec table positions exactly; spectrum rms
scales linearly with energy spread; magnet currents hold through
source-energy changes while optics shift; S3 steering shifts apparent
momentum without charge loss; steering into the pipe loses the beam;
0% spread is a cold beam, not a crash.
