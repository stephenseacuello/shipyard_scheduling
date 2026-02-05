"""Aggregate package for shipyard scheduling modules.

This file sets up the namespace `shipyard_scheduling` so that
subpackages such as `simulation`, `agent`, `phm`, `baselines`, `mes` and
`utils` can be imported via `import shipyard_scheduling.simulation` etc.
Internally it redirects these imports to modules located one level up
(`src/simulation`, etc.) for convenience. This indirection avoids the
need to physically nest the modules under `shipyard_scheduling/`.
"""

from __future__ import annotations

import importlib
import sys


def _redirect_submodule(name: str) -> None:
    """Redirect a submodule import to the top‑level module.

    When Python attempts to import `shipyard_scheduling.<name>` this
    function ensures that it resolves to the module at `simulation`,
    `agent`, `phm`, `baselines`, `mes` or `utils` in the parent package.
    """
    module = importlib.import_module(name)
    sys.modules[__name__ + "." + name.split(".")[-1]] = module


# Redirect known subpackages
for sub in ["simulation", "agent", "phm", "baselines", "mes", "utils"]:
    try:
        _redirect_submodule(sub)
    except ImportError:
        # Ignore if module not found (tests may import partially)
        pass


# Version helper
__version__ = "0.1.0"


def get_version() -> str:
    return __version__