import os
import sys

# Ensure the project's `src` directory is on sys.path so tests import the package
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
