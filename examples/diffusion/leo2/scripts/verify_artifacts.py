"""Run the packaged Leo2 artifact verifier from a source checkout."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if (_REPO_ROOT / "unirl").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

main = importlib.import_module("unirl.models.leo2.verify_artifacts").main


if __name__ == "__main__":
    main()
