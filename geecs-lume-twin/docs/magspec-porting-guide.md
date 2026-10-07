# Porting a magnetic spectrometer into the twin — the playbook

How the HTU Dutch magspec went from "a PDF, some txt files, and a folder
of COSY maps" to a validated, servable twin element — codified so the
same process can be replicated for the other BELLA magspecs (HTT, PW),
which use the same characterization file family. The HTU-specific decode
lives in `magspec-design.md`; THIS document is the transferable process.

The port is two roles: the **beamline owner** gathers artifacts and
answers convention questions (Part 1 — no coding); the **developer/agent**
executes the recipe (Parts 2–4). The HTU port took about a day once the
artifacts were in hand; most of that day was decoding conventions that
this guide now hands you for free.

---

## Part 1 — What the beamline owner collects

1. **The trajectory file** (`*Trj.txt`, tab-separated, one row per
   momentum). The characterization pipeline (Nakamura-style) produces
   these for every spectrometer; column sets vary by generation — both
   observed variants are decoded in Part 2.
2. **The COSY map directory** (`map/`, numbered files, one per
   trajectory row) if the characterization produced transfer maps.
   Without maps a first-order port is still possible (central trajectory
   + table sensitivities), but maps make imaging exact.
3. **The camera calibration file(s)** (`*Cam.txt`) — and an answer to:
   *which dated file reflects the current installation?* (HTU lesson:
   several dated versions existed; the newest was NOT the blessed one.)
4. **The lanex calibration file** (`*lanexCalib.txt`) if counts→charge
   matters for your use.
5. **The tech note / setup slides** for the device (e.g. the EUTERPE
   "Dutch magnet" note). This is where coordinate conventions, screen
   placements, and normalization statements live. Get a copy somewhere
   an agent can read it (NOT only on Google Drive — macOS blocks agent
   processes from CloudStorage paths).
6. **The field knob ground truth**: is the magnet an electromagnet
   (then: the B–I scan data, and which PV/device reports the Hall-probe
   field per shot) or fixed-field (then: on/off = insertion, a boolean)?
7. **Where the spectrometer sits in the beamline**: s-position of the
   magnet center, and what "off" means for downstream transport (drift
   through? physically out of line?).

### Convention questions the owner must answer (or the doc must)

These are exactly the questions that consumed the HTU decoding day:

- Screen coordinate **origins and signs** per screen (HTU: front screen
  0 = beam axis; side screen 0 = magnet center).
- The **camera-file "screen" label ↔ trajectory-file screen column**
  mapping. HTU lesson: they were FLIPPED (cam "back" = Trj "front
  screen"). Never assume; verify against camera positions/FOVs.
- The **map input reference plane**: where is s = 0 of the trajectory
  file physically? (HTU: the `ups/center/dws boundary s` columns
  revealed s=0 = the map input plane, magnet center at ≈0.157 m.)
- Physical screen dimensions, esp. the **non-dispersed height** (HTU:
  ~1" LANEX strip — the one number that came from memory, flagged as an
  assumption in code).

---

## Part 2 — Decoding the files (what generalizes)

### Trajectory table

- One row per momentum, ascending; computed at **field-map max
  normalized to 1 T**. The universal rule: **momentum is linear in B**
  — a particle of momentum p in field B follows the row at p_eff = p/B.
  (Energy is NOT linear in B; always work in momentum.) A per-row
  `max B` column, if present, is recorded metadata (low-p trajectories
  deflect before reaching the 1 T region), not the scaling knob.
- `side logic` (0/1) splits rows between the two screens; per-screen
  landing-coordinate columns (`front screen [m]` / `side screen [m]`)
  are THE dispersion map — monotonic within a screen's subset, so
  screen-position ↔ momentum interpolation is well-posed.
- Per-screen incidence/bending-angle columns: needed to project the
  map's trajectory-perpendicular offset onto the screen coordinate
  (divide by cos of angle-from-screen-normal; drop grazing rows,
  |cos| < 0.1, at the table edges).
- Older format (HTT) carries `momentum rsl [%/mrad]` + conv-fct columns
  instead of maps — first-order sensitivities, only needed for map-less
  ports.

### COSY maps

- One file per trajectory row (verify the count! HTU: 1600 files vs
  1597 rows — aligned from the lowest momentum with a logged warning;
  confirm alignment for a new device if you can).
- File format: coefficient rows `c1 c2 c3 c4 c5 EEEEEE` where the five
  columns are outputs (x, a, y, b, t) and `EEEEEE` is a 6-digit
  exponent string over inputs (x, a, y, b, t, δ) — e.g. `100000` = x,
  `000001` = δ. Dashed line terminates. Count terms by total exponent
  to learn the order (HTU: 6+15+35 → 3rd order).
- **Usage decision (made once, applies to all LPA spectrometers):**
  interpolate across the map FAMILY at each particle's own momentum
  with δ = 0, rather than one map + its δ-expansion. Percent-level
  energy spread spans many rows; the family carries the chromatic
  dependence. Consequence: only terms with zero t and δ exponents are
  evaluated — precompute that reduced term set.

---

## Part 3 — Implementation recipe (mirror `htu/magspec.py`)

1. **Calibration class** (`DutchCalibration` pattern): parse the table
   by HEADER NAME (never column index — formats drift), build dense
   per-row coefficient arrays over the union of map terms, expose
   `map_to_screens(x, a, y, b, p_eff, charge)` returning per-screen
   landing coordinates + weights.
2. **Element class** (`DutchMagSpec` pattern): subclass a Cheetah
   element sized to the physical length.
   - `is_skippable -> False` — MANDATORY, or the Segment folds the
     element into a transfer matrix and your `track()` never runs.
   - Off/B=0: behave as a `drift_kick_drift` drift AND zero the
     readings (no stale spectra).
   - On: weight deposits by `particle_charges * survival_probabilities`
     (NOT raw charges — upstream-killed particles must not appear in
     spectra; this was a real bug found late in the HTU port).
   - Downstream: zero outgoing charge (beam bent out of the line).
   - Do NOT put an angular-acceptance cut in the element — acceptance
     is upstream pipe geometry and belongs to Aperture elements in the
     lattice (a flat angle cut wrongly eats deliberately steered
     beams; found via S3 scans on HTU).
3. **Screen images at PHYSICAL size with square pixels** (front/side
   ranges from the table + chamber drawing; height = the measured
   strip). Non-square arrays exercise upstream bug #3 (NTNDArray dim
   order) — `scripts/postinstall_fixes.sh` must be applied.
4. **Machine-state integration**: knob = B [T] (or coil amps through a
   measured B–I table, with derived readbacks) + an On/Off bool; apply
   via the `_apply_machine_state` branch pattern; PVs in machine units.
5. **Nice-to-haves that proved their worth on HTU**: auto-tracking zoom
   window on the long screen (+ window-center readback PV);
   true-aspect Phoebus viewers; per-camera pixel rendering via the Cam
   file (phase 2 — enables the real analysis code to ingest twin
   images unchanged, the strongest honesty test available).

## Part 4 — Validation checks (write them BEFORE trusting anything)

Add to `tests/validate_physics.py`, in this order:

1. On-axis monoenergetic pencils land at the table positions EXACTLY,
   on both screens, across the momentum range (catches parsing,
   interpolation, projection).
2. The front/side split momentum matches the device doc.
3. Spectral line position moves correctly with B (verify against
   p_eff = p/B by hand for one point).
4. Spectrum rms scales linearly with source energy spread (dispersion
   linearity — HTU: ratio 4.97 for a 5:1 spread change).
5. Upstream steering shifts APPARENT momentum without charge loss
   (the maps' first-order terms working; also the instrument
   systematic worth demonstrating to operators).
6. Magspec on → downstream dark; off → downstream restored exactly.
7. Anything the device doc quantifies (resolution vs mrad, foci
   locations) that you can reproduce cheaply.

Classic failure modes these catch: mirror flips, screen-label swaps,
origin offsets, momentum-vs-energy scaling mistakes, map/row
misalignment (partially — on-axis pencils don't exercise first-order
terms; check 5 does).

---

## When device #2 arrives (rule of two)

Do NOT copy `htu/magspec.py` wholesale. Extract the generic core —
table/map parser, map-family evaluation, screen deposit — into a shared
module (e.g. `common/cosy_spectrometer.py`) parameterized by a per-device
description (file paths, screen geometry, split conventions, knob type),
and make the HTU device the first client. Same rule applies to the whole
beamline layout: `htu/` siblings (`htt/`, `pw/`) should share a common
machine-state/serving core extracted at the moment the second beamline
starts, not before.
