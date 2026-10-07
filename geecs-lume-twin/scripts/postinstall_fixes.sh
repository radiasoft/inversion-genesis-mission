#!/bin/bash
# Post-install fixes for known upstream packaging/library bugs.
# Idempotent — safe to run repeatedly. RUN AFTER EVERY pip (re)install.
# Usage: ./scripts/postinstall_fixes.sh [path-to-venv]   (default: .venv)
set -euo pipefail

VENV="${1:-.venv}"
SP=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

# --- Fix 1: pcaspy macOS arm64 wheel has a broken rpath (points at the
# build machine's conda env). Point it at the system libc++ and re-sign.
SO="$SP/pcaspy/_cas.cpython-311-darwin.so"
if [ -f "$SO" ] && ! "$VENV/bin/python" -c "import pcaspy" 2>/dev/null; then
  install_name_tool -add_rpath /usr/lib "$SO"
  codesign -s - -f "$SO"
  echo "fixed: pcaspy rpath"
else
  echo "ok: pcaspy imports"
fi

# --- Fix 2: lume-pva writes NTNDArray dimension[] in numpy order, but the
# NT spec wants dimension[0] = fastest-varying axis. Square images hide
# it; non-square images render scrambled in Phoebus/p4p. Upstream candidate.
VARS="$SP/lume_pva/variables.py"
if grep -q "for dim in variable.shape" "$VARS"; then
  "$VENV/bin/python" - "$VARS" <<'EOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = "            for dim in variable.shape\n        ]"
new = ("            # NTNDArray spec: dimension[0] is the FASTEST-varying axis;\n"
       "            # numpy C-order flatten makes the LAST axis fastest -> reverse.\n"
       "            for dim in reversed(variable.shape)\n        ]")
assert old in src
p.write_text(src.replace(old, new))
EOF
  echo "fixed: lume-pva NTNDArray dimension order"
else
  echo "ok: lume-pva dimension order already patched"
fi

echo "postinstall fixes complete"
