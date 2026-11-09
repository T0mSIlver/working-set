"""The geometric load ladder bracketing a predicted limit.

Ported from `build_ladder` in the retired standalone harness.
"""
from __future__ import annotations


def build_ladder(predicted_limit_users: float, rungs: str,
                 max_users: int = 1024,
                 operating_point_users: float | None = None,
                 rungs_given: bool = False,
                 dropped: list | None = None) -> list[int]:
    """Populations to measure: `rungs` as multipliers of the predicted limit,
    plus the operating point, unless `rungs_given` (the user named the rungs).

    The operating point is always in the ladder because ttft_miss_s and B* are
    predictions AT that load — sampling them at the predicted LIMIT instead
    systematically failed a correct model.

    THIS is where a fractional operating point becomes a whole number, and one
    of only two such places (the other is RunContext._burst_pop). Everything
    upstream — config.WorkloadCfg.users, Predictions.operating_point_users —
    keeps the load as a float, because a DP8 deployment carrying 4 users runs
    0.5 per group and rounding that to 1 doubles the arrival rate the whole
    queue model is priced at. A ladder rung, by contrast, is a count of
    sessions this process is about to open, so it has to be an integer >= 1.
    """
    mults = [float(x) for x in rungs.split(",") if x.strip()]
    pops = {max(1, round(m * predicted_limit_users)) for m in mults}
    if operating_point_users and not rungs_given:
        pops.add(max(1, round(operating_point_users)))
    ladder = [p for p in sorted(pops) if p <= max_users]
    if dropped is not None:
        dropped.extend(p for p in sorted(pops) if p > max_users)
    if not ladder:
        raise ValueError(
            "the load ladder is empty: "
            + (f"every rung ({', '.join(f'{p:,}' for p in sorted(pops))} "
               f"users) is above --max-users {max_users:,}; raise "
               "--max-users or name smaller --rungs" if pops else
               f"--rungs {rungs!r} names none; give multipliers of the "
               "predicted limit, e.g. --rungs 0.5,1,2"))
    return ladder
