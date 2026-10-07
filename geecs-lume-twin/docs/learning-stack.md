# Learning-based tuning — the tool ladder

Sam's framing (2026-08-25): use the twin to learn how to *tune* the
beamline, leveraging the intrinsic extra knowledge simulation provides.
Worked example: close vertical dispersion at the VISA FODO entrance —
introduced by PMQ misalignment or source pointing, visible as a TILT on
the magspec — by coordinating PMQ hexapod Y and vertical steering to
simultaneously center the beam and remove the tilt.

## The reframe

With a queryable, truth-knowing simulator, most "tuning" problems are not
RL problems. Ladder — each rung only earns its complexity when the one
below fails:

1. **Response-matrix feedback (no learning).** Measure
   d(tilt, centroid)/d(hexapod_Y, S_V) on the twin, invert. The null
   hypothesis every learned method must beat. Interpretable.
2. **Bayesian optimization per episode** — Xopt (already in-house at
   BELLA) on the headless tier. Best-knobs-now for the current hidden
   state; no cross-episode learning. Also the teacher-data generator for
   rung 3.
3. **Supervised learning with privileged information** — the
   underexploited rung, and where "intrinsic extra knowledge" lives. The
   twin knows the seeded hidden truth and can compute the correct
   correction, so generate (observables -> optimal action) pairs and
   train an inverse model by plain supervised learning. Patterns:
   teacher-student distillation ("learning by cheating": teacher sees
   true dispersion, student sees only operator observables) and
   asymmetric actor-critic (critic privileged, actor observable-only).
4. **Actual RL** — earned when the problem is genuinely sequential
   (actions change future observability, e.g. must recover the beam onto
   a screen before the tilt is even measurable) or strongly
   state-dependent. Stack: Gymnasium env wrapper + Stable-Baselines3
   (SAC/TD3/PPO, continuous actions) + domain randomization over hidden
   truth.

## The precedent

Cheetah was BUILT for this: the DESY/KIT ARES work (Kaiser, Eichler et
al.; RL4AA community) trained transverse-tuning policies (quads +
steerers -> target beam on a screen) in Cheetah-backed Gymnasium envs
with SB3, randomized incoming beam + misalignments per episode, and
transferred to the real accelerator. Structurally the same task class as
HTU tuning.

## Mapping onto this twin

- **Training tier = headless** (wrap the LUMEModel, never the PVs):
  ~0.2-0.5 s/step; `Source_NumParticles` is the fidelity dial (train
  low, evaluate high); truncate the line to the relevant stretch;
  SB3 vectorized envs parallelize.
- **The error layer** (roadmap) is the episode seeder = domain
  randomization: reset() draws hidden source pointing/misalignments.
- **Truth probes**: training-only readouts of privileged state, e.g.
  vertical dispersion at the FODO entrance from the particle (y, delta)
  correlation at that plane. Keep the player/referee boundary explicit —
  if ever served, use a separate `HTU:SIM:TRUTH:*` namespace. The
  magspec-tilt observable is a moment of the existing side-screen image.
- **Differentiability**: Cheetah is PyTorch end-to-end -> gradients
  through the lattice (differentiable MPC / analytic policy gradients;
  same machinery as GPSR). CAVEAT: the magspec (numpy COSY maps) and
  slit (masking) break autograd — the differentiable path is
  source -> screens today; torching the magspec is the enabling chore.
- **Deployment rehearsal = the PV tier**: evaluate trained policies over
  the real EPICS interface (the OSPREY sandbox) before hardware.

## First concrete experiment (the dispersion example)

1. Response matrix on the twin + random vertical error seeds; see how
   far inverted-linear feedback gets.
2. Xopt teacher over randomized hidden states -> distill a student that
   maps (magspec tilt, screen centroids) -> (hexapod_Y, S_V corrections).
3. Escalate to SB3 only if student failures look sequential, not
   regression-shaped.

Byproduct either way: the (episode, config, truth) corpus the surrogate
and estimator (docs/vision.md) want.

## The two-twin game (Sam's general formulation, 2026-08-25)

Goal instance: matched transverse phase space + zero Dx/Dy at the
undulator entrance, achieved from OBSERVABLES ONLY.

- **Referee twin**: holds hidden seeds (source + lattice errors); exposes
  only operator PVs/screens; scores on demand via truth probes (mismatch
  parameter, |Dx|, |Dy| at the entrance) AND the number of machine
  interactions spent (shots are the real-world currency — score it from
  day one).
- **Agent**: owns a second twin instance (headless build) as its internal
  model. Loop: measure -> identify (fit its twin's beliefs to the
  measurements: LOCO-lineage system identification; Xopt or gradients
  through Cheetah; GPSR for full phase space) -> solve the correction in
  the calibrated model (deterministic optics, not learning) -> apply ->
  iterate -> declare done -> referee reveals truth.
- Named lineage: LOCO (model calibration from measured responses) +
  model-predictive correction + episode harness; the ARES/Cheetah
  sim2real work covers the policy-learning variant. The harness is the
  new part; every component is standard.
- Dispersion-measurement wrinkle: with only real controls, dispersion is
  measured experimentally via energy jitter — the twin is deterministic
  per put. Two answers: (1) slit-selected energy slices + downstream
  centroid shifts (works TODAY); (2) future "stochastic shot mode"
  (each image draws a jittered beam) — without it the game quietly bans
  every noise-exploiting diagnostic.
- Twin needs (all small): error layer, truth probes + score function,
  optional shot mode. No new ML infrastructure for rung-1 play.
