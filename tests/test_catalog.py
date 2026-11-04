"""The catalogue (src/workingset/data/catalog.toml) and its explorer copy."""
import subprocess
import sys
from pathlib import Path

import pytest

from workingset import model as M

ROOT = Path(__file__).resolve().parents[1]


def test_explorer_copy_is_current():
    # interactive/src/catalog.js is generated: an edit to the TOML that was
    # not regenerated would price the explorer on the old numbers
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "catalog.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_a_misspelt_field_is_refused():
    row = dict(M.load_catalog()["models"]["27B"], kv_bpt_typo=1)
    with pytest.raises(ValueError, match="kv_bpt_typo"):
        M._model_from_catalog("27B", row)
