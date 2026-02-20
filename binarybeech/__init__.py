"""Compatibility shim for src/ layout.

Load the real package implementation from `src/binarybeech` so tests
and local imports keep working. This module registers the loaded
package in `sys.modules['binarybeech']` so submodule imports like
`binarybeech.math` continue to work.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
_SRC_PKG_INIT = os.path.join(_ROOT, "src", "binarybeech", "__init__.py")

if os.path.exists(_SRC_PKG_INIT):
    spec = importlib.util.spec_from_file_location("binarybeech", _SRC_PKG_INIT)
    _mod = importlib.util.module_from_spec(spec)
    # Register under the canonical package name so imports resolve
    sys.modules["binarybeech"] = _mod
    spec.loader.exec_module(_mod)  # type: ignore

    # Mirror public attributes into this shim module's globals
    for _name, _val in list(_mod.__dict__.items()):
        if not _name.startswith("__"):
            globals()[_name] = _val
else:
    raise ImportError(
        "src/binarybeech package not found — ensure `src/binarybeech/__init__.py` exists"
    )
