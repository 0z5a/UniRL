"""Launch the retained native Leo2 sampler from packaged vendor sources."""

from __future__ import annotations

import sys
from pathlib import Path

from unirl.models.transformers_compat import install_transformers_flash_attention_compat


def main() -> None:
    """Bootstrap packaged sources and delegate to the native sampler CLI."""
    vendor_root = Path(__file__).resolve().parent / "vendor" / "gen_ar"
    for path in (vendor_root, vendor_root / "deps/hy_parallelism", vendor_root / "deps/IndexKits"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    install_transformers_flash_attention_compat()

    from hymm.samplers.entry import run

    run()


if __name__ == "__main__":
    main()
