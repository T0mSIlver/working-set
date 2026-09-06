"""The model's own self-checks, run as a test. They are the calibration
anchors (measured pools, MFU band, decode measurement) and take ~1 min."""
import sys
from pathlib import Path

import pytest

from workingset import model as M


@pytest.mark.slow
def test_selfcheck():
    M._selfcheck()


def test_scripts_alias_is_the_same_module():
    """scripts/scenario_model.py must alias workingset.model, not copy it."""
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import scenario_model as S  # noqa: F401
    finally:
        sys.path.remove(str(scripts))
    assert S is M
    assert S.MODELS is M.MODELS
