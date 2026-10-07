# geecs-lume-twin — Agent Context

HTU transport-line virtual accelerator / digital twin. README.md has the
architecture, PV surface, and quickstart; this file carries the rules and
conventions an agent cannot re-derive from the code. Owner: Sam (skbarber)
— solo, spare-time; prefer minimal, verifiable increments.

## Non-negotiable physics rules

1. **No single-reference-energy linear optics, ever.** LPA beams have
   percent-to-tens-of-percent energy spread; second-moment/sigma-matrix
   propagation and Cheetah's default `tracking_method='linear'` are both
   banned. Every tracking element uses `drift_kick_drift` (Cheetah's
   Bmad-X integrator, per-particle k1 = b1/(L(1+pz))). If you add an
   element, set its tracking method explicitly — the Cheetah default is
   the wrong one.
2. **PVs speak machine units** (coil amps, R56 in um, tesla), never
   physics units (k1, angles). All conversions go through
   `htu/calibrations.py` — measured numbers with provenance comments; do
   not invent or "improve" calibration constants.
3. **Authoritative state lives in `(sim.machine_state, sim.source_params)`**,
   never in element attributes. `_apply_machine_state()` re-derives every
   element parameter from state at the current per-element beam energy.
   New knobs follow this pattern: add a state key + an apply branch + a
   `MachineStateVariable`. Never write element attributes directly from a
   variable's `_set` — that breaks the energy-rederivation invariant.
4. **Variable `_set` must be transactional**: never commit state before
   everything that can raise has run (a poisoned state kills the lume-pva
   Runner AND its crash-recovery path — this took the server down once).

## Conventions that will bite you

- **MagSpec screen coordinates**: front screen 0 = beam axis; side screen
  0 = magnet center. **Camera-file "screen" labels are FLIPPED vs the
  trajectory file**: cam "back" = Trj "front screen" (high energy), cam
  "front" = Trj "side screen". Fully documented in docs/magspec-design.md
  — read it before touching `htu/magspec.py`.
- **Slit jaw PVs are LAB-frame** (relative to the straight R56=0 axis;
  chamber-mounted plates). Cheetah tracks in the reference-orbit frame, so
  _apply_machine_state translates via _slit_ref_offset (and shifts the
  ChicaneSlit camera the same way). Anything else chamber-mounted near the
  chicane must get the same treatment.
- **Momentum, not energy, scales with B** (p_actual = p_table x B). The
  trajectory table + COSY maps are computed at field-map max = 1 T.
- **Custom Cheetah elements must override `is_skippable -> False`** or
  the Segment folds them into a transfer matrix and `track()` never runs.
- **Map/row alignment caveat**: `htu/data/dutch/map/` has 1600 files vs
  1597 trajectory rows; aligned from the lowest momentum (warning logged
  at load). Not yet independently verified — a small offset would not
  show in the pencil-landing checks (on-axis pencils don't exercise the
  maps' first-order terms).
- **Non-square NTNDArray images require upstream fix #3** (see README);
  if images look scrambled in Phoebus, `scripts/postinstall_fixes.sh`
  was not run after a reinstall.
- **The synoptic display is generated** — never hand-edit
  `display/htu_synoptic.bob`; edit `display/generate_synoptic.py` and
  rerun it (`.venv/bin/python display/generate_synoptic.py` after
  `pip install -e .`).
- Placeholder vs real data: quad bores, EMQ calibrations, kicker fields,
  chicane B-I scan (Sam 2026-08-25) + R56 closed form, pipe aperture (r=10mm, EMQ1-10cm -> magspec), magspec
  table/maps = REAL. Default camera geometry (7um/1024^2 when no ECS dump
  is loaded) and the source parameterization (per-plane Twiss Gaussian,
  no energy-slice structure yet) = idealized.

## Workflow

- **After ANY physics change**: `.venv/bin/python tests/validate_physics.py`
  — 35 checks, all must pass. Add a check when you add physics; every
  bug found so far (angle-clip artifact, dim-order scramble, 0%-spread
  crash) would have been caught by the right check existing earlier.
- Serve with `.venv/bin/htu-twin-serve`; kill with `pkill -f htu-twin-serve`
  (note: `pkill -f serve` patterns have missed processes before — verify
  with `ps aux | grep htu` and `lsof -nP -iUDP:5076` for port squatters).
- Restart Phoebus after a server restart (off-subnet channels back off to
  glacial retry).
- Commit style: small commits, physics claims verified in the message.
  This repo is not under the GEECS-Plugins PR ritual — Sam works it
  directly — but the same honesty rules apply.

## Relationship to GEECS-Plugins

This repo stays dependency-free of the monorepo. The `HTU:SIM:` namespace
mirrors the real naming convention below the prefix (mirror by reference,
never import). When work needs ophyd bridge devices, queueserver/Tiled
sim profiles, or OSPREY policy changes ("SIM:* is agent-writable"), that
code belongs in GEECS-Plugins.

## Roadmap pointers

docs/vision.md is the direction (hidden-error "VA game", surrogate,
source estimator); docs/learning-stack.md is the tuning-ML tool ladder;
docs/magspec-porting-guide.md is the replicable recipe for porting the
other BELLA magspecs (HTT, PW). Multi-beamline growth: sibling packages
next to htu/ share a common core extracted at the SECOND beamline
(rule of two) — never before, never by copy-paste. Nearest concrete increments, per Sam: more knobs for
source/EMQs/PMQs (follow rule 3's pattern), an error layer (a second
privileged dict alongside source_params), a Transmission readback PV,
per-camera magspec rendering (phase 2 in docs/magspec-design.md).
