"""Compatibility shim for src/ layout.

This module loads the real package implementation from `src/binarybeech`
so local imports continue to work while the repository uses the `src/`
layout for packaging.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
_SRC_PKG_INIT = os.path.join(_ROOT, "src", "binarybeech", "__init__.py")

if os.path.exists(_SRC_PKG_INIT):
    spec = importlib.util.spec_from_file_location("_binarybeech_src", _SRC_PKG_INIT)
    _mod = importlib.util.module_from_spec(spec)
    sys.modules["_binarybeech_src"] = _mod
    spec.loader.exec_module(_mod)  # type: ignore

    # Re-export public attributes from the real package
    for _name, _val in list(_mod.__dict__.items()):
        if not _name.startswith("_"):
            globals()[_name] = _val

    # Keep a reference to the real module
    __real_module__ = _mod
else:
    raise ImportError("src/binarybeech package not found — run from repository root after moving package into src/")
