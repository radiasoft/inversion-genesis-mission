"""Headless config snapshot/restore round-trip (no server needed):

    .venv/bin/python tests/test_config_roundtrip.py

Exercises the exact collect/apply logic the CLI uses (collect_config over
a getter, coerce_value on load) against the model directly, plus the YAML
file round-trip including the MagSpec_On boolean.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from htu.config_tool import (  # noqa: E402
    collect_config,
    load_config_file,
    write_config_file,
)
from htu.model import build_htu_model, writable_pv_names  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}  {detail}")
    if not cond:
        FAILURES.append(name)


m = build_htu_model()

# change a few values away from defaults (incl. the boolean)
changed = {
    "EMQ1H_Current": 0.5,
    "ChicaneDipole_Current": 4.2,
    "Source_BetaX_mm": 7.5,
    "Source_Xp_mrad": 0.3,
    "PMQ2H_Tilt_mrad": 25.0,
    "ChicaneSlit_Jaw1_mm": 2.0,
    "MagSpec_On": True,
}
m._set(changed)

snap = collect_config(lambda name: m._state[name])
check("snapshot covers every writable PV", set(snap) == set(writable_pv_names()),
      f"{len(snap)} PVs")
check("snapshot bool is a python bool", snap["MagSpec_On"] is True)
from htu.model import integer_pv_names  # noqa: E402

_ints = integer_pv_names()
check("snapshot ints are python ints",
      all(isinstance(snap[k], int) and not isinstance(snap[k], bool) for k in _ints))
check("snapshot floats are python floats",
      all(isinstance(v, float) for k, v in snap.items()
          if k != "MagSpec_On" and k not in _ints))

# YAML file round-trip
with tempfile.TemporaryDirectory() as td:
    path = str(Path(td) / "roundtrip.yaml")
    write_config_file(path, snap, note="round-trip test")
    loaded = load_config_file(path)
check("YAML round-trip preserves every value exactly", loaded == snap)

# perturb the model, re-apply the snapshot, verify full restore
m._set({"EMQ1H_Current": -1.0, "MagSpec_On": False, "Source_BetaX_mm": 5.0,
        "ChicaneSlit_Jaw1_mm": 20.0})
m._set(loaded)
after = collect_config(lambda name: m._state[name])
check("re-applied config restores the full state", after == snap)
check("boolean survived the round trip", after["MagSpec_On"] is True)

print(f"\n{len(FAILURES)} failures" if FAILURES else "\nALL CHECKS PASSED")
sys.exit(1 if FAILURES else 0)
