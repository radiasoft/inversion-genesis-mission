# Where this is going

Sam's framing (2026-08-25), recorded so future work builds toward it
rather than rediscovering it.

## The core idea

Seed the virtual accelerator with **hidden truth** — unknown source
parameters *and* unknown lattice errors (calibration offsets,
misalignments, tilts) — and let a player (an AI agent, an optimizer, or a
human trainee) tune the beam using **only the real controls and real
diagnostics**. Run many episodes. Because the twin's interface is the
real EPICS interface, what is learned is expressed in the operational
vocabulary and transfers to the real machine.

## The three products of the episode corpus

Each episode yields a (hidden truth, control trajectory, observations,
outcome) tuple. Thousands of them feed three distinct artifacts:

1. **Surrogate** — fast forward model (controls + source -> diagnostics).
   The standard LUME endgame; could eventually replace Cheetah as the
   live-serving backend (kHz instead of ~5 Hz) with Cheetah as ground
   truth.
2. **Estimator** — the inverse map (observations -> hidden parameters).
   This is the source-estimator designed at project start, now with
   unlimited labeled training data. Pointed at real scan data, it
   re-seeds the twin from the machine — the actual closed loop.
   GPSR (Roussel et al., generative phase space reconstruction) is the
   serious prior art for the full-phase-space version.
3. **Operational knowledge** — which knob disambiguates which error,
   which diagnostic signatures identify which failure, what a good
   tuning procedure looks like. The thing beam time never gives cleanly,
   because on the real machine the ground truth is never known.

## Three fidelity tiers (already available)

- **Headless**: call the LUMEModel directly — milliseconds/eval; where
  "many, many episodes" runs.
- **PV-level**: the running server — where an agent interacts through
  the real control-system interface.
- **Full-stack**: through queueserver/Tiled (bridge devices, configs in
  GEECS-Plugins) — where the tuning *workflow* itself is exercised, and
  the CI story for the whole DAQ stack.

Xopt (already used at BELLA) is the natural optimization inner loop; the
OSPREY agent sandbox story is this same twin in PV-level mode.

## The honesty discipline

The game teaches exactly as much as the twin's error model is honest.
Current coverage: measured calibrations, real COSY maps, chromatic
tracking, real apertures. Not covered: field-quality errors, jitter
correlations, drift, wakefields. Policies trained here are right about
the physics the twin contains and silently confident about what it
doesn't. Standing rule: **every time the real machine surprises the
twin, that surprise becomes a new seeded error, not a shrug.**

## Nearest concrete increments

1. Error layer: a second privileged dict alongside `source_params`
   (per-magnet calibration scale/offset, misalignment, roll), applied in
   `_apply_machine_state`, invisible to the control PVs.
2. `Transmission` readback PV — a scalar objective for tuning episodes.
3. More machine knobs (PMQ z-positions / jet-z, EMQ trims) following the
   machine_state pattern (CLAUDE.md rule 3).
4. Per-camera magspec rendering (docs/magspec-design.md phase 2) so the
   real ImageAnalysis analyzer ingests twin images unchanged — the
   strongest end-to-end honesty test available.
