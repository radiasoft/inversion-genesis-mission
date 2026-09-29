#!/usr/bin/env python3
"""CLI entry point for generating merged-docstring config stubs."""

from __future__ import annotations

import sys
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "inversion_fbpic" / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from _doc_management.stub_sync import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
