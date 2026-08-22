"""Entry point: ``python -m mlcookbook`` launches the desktop app.

The analytical package stays UI-free — this is a trampoline that locates the
repo-root ``app.py`` (the gui <-> core seam) and hands off to it. It works
from any working directory because it walks up from this source file.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "app.py").is_file() and (parent / "gui").is_dir():
            return parent
    raise SystemExit(
        "Could not find the MLcookbook app (expected app.py and gui/ above the "
        "mlcookbook source tree). Run from a source checkout, or use the packaged exe."
    )


root = _find_repo_root()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app import main  # noqa: E402

raise SystemExit(main())
