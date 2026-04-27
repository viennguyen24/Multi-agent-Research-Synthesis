from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root() -> None:
    """Add the project root to sys.path so eval scripts run directly from the repo."""
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
