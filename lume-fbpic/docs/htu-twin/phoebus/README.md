# Phoebus displays for the LPA-fed HTU twin

Displays for the HTU transport twin when its source is a `lume-fbpic` run,
served by `serve.py --twin`. All PVs are `pva://HTU:SIM:...`.

| File | What it is |
|---|---|
| `htu_synoptic_lpa.bob` | The twin's synoptic with the LPA tied in: a bottom strip with the LPA run selector and the twin's source readbacks, and the `SRC` element opening `lpa_bunch.bob`. Generated; do not edit. |
| `lpa_bunch.bob` | The LPA bunch: run selector, the twin's `Source_*` readbacks, the 33 moment-descriptor scalars (covariance as a grid), run statistics. Generated; do not edit. |
| `generate_linked_synoptic.py` | Builds `htu_synoptic_lpa.bob` from the twin's own `htu_synoptic.bob`, which is left untouched. |
| `generate_lpa_display.py` | Builds `lpa_bunch.bob` from the descriptor action names. |

## Run the server

See `../README.md` (`serve.py --twin`).

## Open the display

Start Phoebus with this display as the resource, for example from the Phoebus product directory:

```bash
java -jar target/product-<version>.jar -resource <this directory>/htu_synoptic_lpa.bob
```

Start the server first, and restart Phoebus after any server restart. Phoebus restores your last
session, so older windows may open too; use the one with the LPA strip along its bottom.

- **Choose a run:** the "Run:" combo box in the strip (or on `lpa_bunch.bob`) selects the active
  archive. The descriptor values and the twin's source follow after the twin re-tracks, about
  2 seconds.
- **`SRC` element:** opens `lpa_bunch.bob`.
- **Other elements:** the twin's own control and camera displays, found through the `TWIN_DISPLAYS`
  macro. It defaults to the directory the synoptic was generated from; set it when opening the
  display if the twin's displays live elsewhere: `-resource "htu_synoptic_lpa.bob?TWIN_DISPLAYS=<twin display directory>"`.
- **`lpa_bunch.bob` on its own** takes a `P` macro for the PV prefix (default `HTU:SIM:`).

## Regenerate the displays

```bash
python generate_lpa_display.py
python generate_linked_synoptic.py <twin display directory>
```

Run the second whenever the twin's synoptic changes. The twin's own files are only read.

## Things to know

- The twin's own `htu_synoptic.bob` writes `<rotation_step>90.0</rotation_step>` for its vertical
  labels. Phoebus expects the enum's ordinal there (0 none, 1 = 90 degrees, 2 = 180, 3 = 270), so
  it logs a warning for each of the 103 labels and does not rotate them; a newer Phoebus does not
  change that, it is the file. `generate_linked_synoptic.py` converts them (`90.0` becomes `1`), so
  `htu_synoptic_lpa.bob` loads without those warnings. The fix for the twin itself is one line in
  its `display/generate_synoptic.py` (line 50): `<rotation_step>1</rotation_step>`.
- The `Source_*` PVs are read-only: the LPA run is the twin's source. They show the bunch's moments.
- The three run statistics (`charge_pc`, `energy_mean_mev`, `energy_std_mev`) read NaN for a
  reconstructed run: it has no particles to compute them from.
- Phoebus waits one second, on its UI thread, for a put reply; a server that acknowledged only after
  a re-track (about 3 s) made every write time out. That is why puts are acknowledged at once by
  default.
- The tests in `tests/test_phoebus_displays.py` check the generators and that the committed
  `lpa_bunch.bob` is up to date. `htu_synoptic_lpa.bob` depends on where the twin's display directory is, so
  regenerate it on each machine.
