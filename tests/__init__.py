"""Test suite for the rPPG package.

Automatically configures the Python search path so tests run reliably
both from the project root and from parent directories.
"""

import sys
from pathlib import Path

# Ensure project root and package parent directory are in sys.path
_project_root = Path(__file__).resolve().parent.parent
_parent_dir = _project_root.parent

for _p in [str(_parent_dir), str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
