"""Small numerics shared by every probe.

`pct` is the harness's percentile verbatim (linear interpolation, nan on
empty) rather than numpy's: the probes must produce byte-identical statistics
to the retired standalone harness, and a percentile convention is exactly the
kind of thing that drifts silently between two implementations.
"""
from __future__ import annotations

import math
from typing import Sequence

# Freeze thresholds reported as a LADDER, not a single number. A lone
# threshold is load-bearing in the worst way: set it at 250 ms and a
# configuration whose worst freeze is 171 ms reads as "zero freezes", turning
# a 7.5x quantitative win into a spurious qualitative one. The ladder shows
# the distribution and lets the reader pick.
FREEZE_LADDER_MS: tuple[float, ...] = (50.0, 100.0, 250.0, 500.0, 1000.0)

DEFAULT_RUNGS = "0.25,0.5,0.75,1,1.25,1.5,2"


def pct(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; nan on empty input."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    frac = pos - lo
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def fmt(x, unit: str = "", nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}{unit}"


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def restore_nans(cls, d: dict) -> dict:
    """JSON has no nan, so a record writes those fields as null. Fields whose
    dataclass DEFAULT is nan mean "not measured", and the report reads them
    with math.isfinite — so null goes back to nan on the way in, and only
    there. Fields that default to None keep their None."""
    out = dict(d)
    for name, f in cls.__dataclass_fields__.items():
        if out.get(name, "keep") is None and isinstance(f.default, float) \
                and math.isnan(f.default):
            out[name] = float("nan")
    return out
