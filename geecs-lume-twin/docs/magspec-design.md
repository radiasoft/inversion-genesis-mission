# MagSpec as a trackable element — design note (NOT yet implemented)

Studied 2026-08-25 against the HTT example files in
`GEECS-Plugins/ImageAnalysis/image_analysis/data/HTT-MagSpce-Calibrations/`
(`170925DnnVarianTrj.txt`, `260224DnnVarianCam.txt`, `250101lanexCalib.txt`)
and the consuming code
`ImageAnalysis/image_analysis/analyzers/magspec_manual_calib_analyzer.py`.
The real (HTU) magspec uses the same file family/format; only the numbers
differ.

## What the trajectory file encodes

One row per **momentum** (example spans 10 GeV/c → 3 MeV/c, ~693 rows,
finer spacing at low p), all computed for a **normalized peak field of
1 T**. That normalization makes the table universal by rigidity scaling:

> a particle of momentum `p` in actual field `B` follows the tabulated
> trajectory of the row at `p_eff = p / B`.

(Verified: `orbit radius = p[GeV] / (0.3 · 1 T)` exactly, per row.)

Per row, the columns give the full central-trajectory solution through the
*measured* field map, reduced to hard-edge equivalents plus screen
geometry:

- `bending angle`, `bending angle at screen`, `incident angle [dgr]` —
  trajectory geometry, including the angle at which it strikes the screen
- `drift1`, `drift2`, `total path [m]` — entrance/exit geometry and total
  path length to the screen (energy-dependent — the "screen ends at
  different distances for different trajectories" property)
- `effective length`, `MagRds`, `Bpeak`, `B.m` — hard-edge equivalent of
  the field map at 1 T
- **`side logic` (0/1)** — which screen this trajectory lands on: the
  large-view spectrometer has a *front* ("back" in the camera file) screen
  for high-momentum/small-bend trajectories and a *side* screen for
  low-momentum/large-bend ones (example: split at ~100 MeV/c, bends up to
  ~136°)
- **`front screen [m]` / `side screen [m]`** — the landing coordinate
  along the respective screen. The analyzer takes the per-`side logic`
  subset, ×1000 → mm, sorts, and interpolates screen_mm ↔ momentum
  (monotone spline). This is THE dispersion map.
- `momentum rsl [%/mrad]` — apparent-momentum error per mrad of incoming
  angle: the first-order sensitivity that turns beam divergence into
  energy-measurement blur
- `x/y conv fct rms` — unity in the example file; presumably populated
  for spectrometers where they matter

## What the companion files encode

- **Cam file**: one row per physical camera viewing the screens: `FOV
  [mm]`, `Left edge [mm]` (screen coordinate of pixel column 0), ROI
  bounds, rotation, which screen, calibration `set`, `sensitivity`. Energy
  axis per camera: `x_mm[i] = left_edge − (FOV/width)·i`, then table
  interp → momentum per pixel.
- **Lanex file**: counts→charge: FOV-dependent polynomial per calibration
  set, screen factor (1.0 back / 1.98 side), camera sensitivity → the
  counts-to-fC factor.

## Proposed twin encoding (when we build it)

**A map element, not integrated tracking.** The table already contains the
real field map's answer; re-integrating a hard-edge fit would be less
faithful. Sketch:

1. `MagSpecCalibration` — parses the Trj file into two monotone tables
   (front/side): `p_1T`, `screen_m`, `path_m`, `incident_angle`, `rsl`.
   Pure functions: `(p, B) -> (screen_id, s_mm)` via `p_eff = p/B` lookup;
   out-of-range momenta miss both screens.
2. `MagSpecScreen(cheetah Drift subclass)` with a `bpeak_t` knob:
   - `B = 0` (transport mode, magspec off): behaves as a drift_kick_drift
     drift of the physical length — beam continues downstream.
   - `B > 0` (characterization mode): per particle, `p_i = p0c(1+δ)` →
     lookup at `p_i/B` → screen + `s_mm`; vertical: `y_screen = y +
     y'·path(p_i)`; angle blur: apparent momentum `p(1 + rsl·x'/100)`
     before lookup; deposit charge-weighted histograms per screen;
     **outgoing beam has zeroed charges** so the downstream A-line goes
     dark — matching operations ("magspec on = characterize, off = send
     downstream").
   - Cheetah trap: must override `is_skippable = False`, else the Segment
     merges the element into a transfer matrix and custom `track()` never
     runs.
3. Serving: `MagSpec_B_T` machine PV (0 = off; or coil amps if a
   current→B calibration exists for the HTU device); screen images as
   NDVariables in screen-mm space.
4. **The closed-loop payoff (phase 2):** render per-CAMERA images using
   the Cam-file pixel geometry instead of screen-mm space. Then the real
   ImageAnalysis magspec analyzer can ingest the twin's synthetic images
   unchanged — the full measurement chain (beam → screens → cameras →
   energy spectrum) validated end-to-end against known source spectra.

## HTU device findings (2026-08-25, from /Volumes/hdna2/data/Undulator/Calibrations/)

The real HTU magspec is the **"Dutch magnet"** (docs: DutchMagnetV0b-2.pdf,
180701MooreDutchSetupV1.pptx — Google Drive; not yet read, CloudStorage is
TCC-blocked for agent processes; need copies in an accessible location).

`180702Dutch1Trj.txt` is a RICHER evolution of the HTT format — same idea,
more columns (1597 rows, 11.3 MeV/c -> 1e10 sentinel):

- `max B [T]` = 0.993 — the table appears computed at the actual measured
  peak field, not normalized 1 T (CONFIRM: is rigidity rescaling still the
  intended use, or is the field fixed? If the Dutch magnet is
  permanent-magnet, "on/off" = physical insertion, and the twin knob is a
  boolean, not a B value)
- both `front screen [m]` and `side screen [m]` per row, plus per-screen
  bending angles; `inc angle in/out`; fringe extend/backoff; `eff length
  ups/dws`; `ups/center/dws boundary s [m]` (magnet placement along the
  beamline)
- naming crosswalk hazard: Trj speaks front/side; the Cam files speak
  front/back ("back" mapped to the front-screen column in the HTT
  analyzer). CONFIRM which camera looks at which physical screen.

Camera files: many dated versions (`241029Dutch1Cam.txt` latest; also
`Dutch2Cam` files 2023 — a second magspec?); slightly different columns
from HTT (no name/sensitivity; `setN`; screens "front"/"back").
`forMagSpecDevice/220908Dutch1Cam.txt` = presumably what the online
MagSpec device consumes. `140324lanexCalib.txt`/`221214lanexCalib.txt`
present.

**`map/` directory: 1600 numbered files (1001.txt ...), one per trajectory
row — COSY transfer maps (CONFIRMED by Sam 2026-08-25).** Format per file:
56 coefficient rows + dashed terminator; each row = 5 output-column
coefficients (x_f, a_f, y_f, b_f, t_f) and a 6-digit exponent string over
(x, a, y, b, t, δ) — e.g. `100000` = x, `000001` = δ. Term counts: 6
first-order, 15 second-order, 35 third-order → **3rd-order maps**, midplane
symmetry visible (x-plane and y-plane terms decouple at 1st order; t_f
carries path-length dependence on x and δ). So the twin's magspec element
applies the per-momentum interpolated 3rd-order MAP to each particle —
exact imaging including offset/angle/chromatic cross terms, superseding the
central-trajectory + rsl first-order plan.

Pinned by Sam: camera calibration file = `230215Dutch1Cam.txt` (4 cameras:
1,2,4 on "front", 3 on "back"; note later-dated files like 241029 exist —
230215 is the one Sam points at). Map path = the same
`/Volumes/hdna2/data/Undulator/Calibrations/map/` found above.

## ANSWERS from the source documents (2026-08-25)

Read: Nakamura, "Electron Spectrometer Using the EUTERPE Magnet (A.K.A.
Dutch Magnet)" (DutchMagnetV0b-2.pdf, 2013, 61pp) and
180701MooreDutchSetupV1.pptx (2018 Moore Dutch 1 reinstall). Copies live in
this repo root.

- **Magnet**: EUTERPE dipole, ELECTROMAGNET (max ~1.6 T, gap 25 mm, pole
  120x480 mm, effective field 144.4x504.4 mm). Field is continuously
  monitored by a Hall probe and recorded per shot (legacy scan column
  'StagingMagSpec Bfield') — so the twin's knob is B [T] directly (or the
  HTU device current + 100722BIScan.xlsx calibration if we want amps).
- **Normalization**: all trajectory/map calcs at field-map max = 1 T.
  Momentum is LINEAR in B: p_actual = p_table x B_measured. (Use momentum,
  not energy — energies are not linear in B.) The per-row `max B` column
  (<1 for low p) is the peak field that trajectory actually samples — it's
  recorded metadata, not the scaling knob.
- **Screen coordinates**: front screen 0 = laser/beam axis; side screen
  0 = magnet center. Physical placement (2018): front screen plane at
  z = 354.6 mm, side screen at x = 170 mm w.r.t. magnet center; the
  CHAMBER defines beamline and screens (magnet can float — a measured
  5.7 mm yoke offset exists).
- **Camera <-> screen crosswalk (double-confirmed)**: the Cam-file
  `screen` label is flipped w.r.t. Trj naming: cam-file **"back" = Trj
  "front screen"** (high energy; cam3 in 230215, FOV 149 mm matching the
  ~8-170 mm front range) and cam-file **"front" = Trj "side screen"**
  (cams 1, 2 at left edges 167.4/332.4 mm matching the 2018 side-camera
  positions 170.2/332.4 mm from magnet center; cam4 = later high-res
  addition). Same flip the HTT analyzer encodes ("back" -> front-screen
  column).
- **drf1/drf2**: reference plane -> upstream effective boundary, and
  downstream effective boundary -> screen. Boundaries are per-momentum
  (Enge-fit effective boundaries; fringe extend/backoff columns are the
  COSY FR-range fixups). The `ups/center/dws boundary s [m]` columns are
  in the file's own s coordinate whose **s = 0 is the map input reference
  plane**: ups boundary s == drf1 per row, center s ~= 0.157 m. So the
  twin element's input plane sits ~157 mm upstream of the magnet center —
  that fixes its s-position in the HTU lattice.
- **COSY model**: D1 + DI dipole (equivalent sharp-boundary radius +
  bend angle + edge angles + Enge fringes) + D2, maps computed per
  momentum from the SOURCE/reference plane to the screen. Edge-angle sign
  convention: positive = defocusing in bend plane; 0 = normal incidence.
- **Angular acceptance**: aperture-limited to about +/-2.7 mrad
  (beam-tube ID 15.24 mm); the twin element should clip accordingly.
- **momentum rsl / x conv fct**: HTT-format columns; the HTU Dutch file
  doesn't carry them — with the 3rd-order maps they're unnecessary
  (resolution effects emerge from the maps themselves).
- **"Dutch 1 / Dutch 2"**: 2018 note — "Staging Dutch is now Moore
  Dutch 1 (2 after undulator planned)"; the Dutch2Cam files (2023) are
  that second, post-undulator spectrometer.

## Remaining questions (minor)

- Which GEECS device/variable serves the Hall-probe B at HTU (for twin
  mode following the real machine), and whether the twin knob should be
  B [T] or PSU current [A] (needs 100722BIScan.xlsx if amps).
- Exact absolute s of the magnet center in the HTU beamline (our lattice
  places MagSpec via the transport-repo drifts; verify against the
  354.6/170 mm chamber geometry when implementing).
- Whether cam file `230215Dutch1Cam.txt` reflects today's install vs the
  later-dated 241029 file (Sam pointed at 230215).
