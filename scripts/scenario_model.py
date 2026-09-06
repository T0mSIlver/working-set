"""Compatibility alias: the model now lives in the `workingset` package.

Every study script that does `import scenario_model as M` keeps working;
`M` IS `workingset.model` (same module object, so private names, dataclass
identities and monkeypatches all agree). New code imports the package.
"""
from __future__ import annotations

import sys

try:
    from workingset import model as _model
except ImportError:  # running from a checkout without the package installed
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from workingset import model as _model

sys.modules[__name__] = _model

if __name__ == "__main__":
    _model._selfcheck()
