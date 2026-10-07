# configs/ — saved machine configurations

Doctrine:

- **Code defaults are the pristine physics baseline** — the state the
  validation suite (`tests/validate_physics.py`) runs against. They live
  in `htu/model.py` (`DEFAULT_MACHINE_STATE`) and `htu/source.py`
  (`SourceParams`), and only change for physics reasons.
- **Matched-to-experiment states live here** as named, git-tracked YAML
  files with a `note` saying what they represent. Save one with
  `htu-twin-config save configs/<name>.yaml --note "..."` against a
  running server; restore with `htu-twin-config load`, or boot the server
  with one via `htu-twin-serve --config configs/<name>.yaml`.
- **Diff two configs to see machine/source drift** between matching
  sessions — they are flat `{pv_suffix: value}` maps, so `git diff` /
  `diff` reads directly as "what moved".

`baseline.yaml` is the code-default state snapshotted for reference (and
as a diff anchor); it should track the code defaults.
