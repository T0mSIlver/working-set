"""What one agentic-coding seat costs on a metered API or a coding plan,
priced from the same workload the self-host side uses.

The self-host €/seat divides a 24/7 bill by the concurrent sessions a
configuration carries. A metered API bills tokens, so its seat price depends
on how many hours the seat is live. Everything below is closed form in the
normal CDF, so the explorer's mirror (interactive/src/apicost.js) reproduces
it to rounding.

Token flows (research/api_prices.md § 3). A request of the session's main
agent carries a prompt L, a clipped lognormal ([sys_user, cap], median
user_median, sigma user_sigma); a subagent request the same shape on its own
parameters. Within a session each prompt is the previous one plus Δ fresh
tokens (the warm turn the self-host side prefills on a hit), so on a cache hit

    read  = L - Δ        written = Δ

and on a miss (probability q) the whole prompt is written again. Output is
`out_tokens` per request, every class. Requests arrive at
(1 + sub_ratio) / think_time_s per live session, the self-host arrival rate.

The miss probability is the configured invalidation plus the cache's expiry:
the provider's cache entry outlives a gap only up to its TTL, and the gap
after a request is a tool wait (n_t - 1 of every n_t requests) or a human
wait (the last of the turn), both lognormal from the traced anchors
(model.MEASURED_*). A provider offering several TTLs is priced at the
cheapest for this workload, as an operator who read the price sheet would.

Long-context tiers price a whole request at the tier's rates once its prompt
passes the threshold (every vendor here bills that way); the share of
requests and tokens above it comes from the same clipped lognormal.

A seat is one person with access. With the population layer off every
person is a live session at the peak (peak_active_share 1), the self-host
table's unit. An API seat bills `active_hours` of that session per month;
divided by peak_active_share it is the price per concurrent session, the
unit the frontier table prints for self-hosted rows.
"""
from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

from . import model as M

# ---- defaults ---------------------------------------------------------------
# Active hours per seat-month: 4 h of live agent session per working day,
# 21 working days. Requesty's public gateway data (research/api_prices.md § 5)
# put the median Claude Code user at ~8 h and the P95 user at ~97 h on this
# workload; 84 h is a full-time agentic user, not the median one.
ACTIVE_HOURS_DEFAULT = 84.0
WORKDAYS_PER_MONTH = 21.0
WEEKS_PER_MONTH = 30.0 / 7.0          # the explorer's flat 30-day month

# The gap after a request, from the traced anchors (model.py, MEASURED_*):
# tool waits lognormal, median 0.61 s, sigma 2.43; human waits median 58 s,
# mean 275 s, so sigma = sqrt(2 ln(275 / 58)).
TOOL_GAP_MEDIAN_S = 0.61
TOOL_GAP_SIGMA = 2.43
HUMAN_GAP_MEDIAN_S = 58.0
HUMAN_GAP_SIGMA = math.sqrt(2.0 * math.log(M.MEASURED_T_HUMAN_S / HUMAN_GAP_MEDIAN_S))


# ---- the normal CDF, the same arithmetic as the mirror -----------------------
def _erfc(x: float) -> float:
    """Complementary error function, Numerical Recipes' erfcc (fractional
    error < 1.2e-7 everywhere). Not math.erfc: the mirror runs the same
    polynomial, so the two sides agree to float rounding, not to 1e-7."""
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196
          + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
          + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return ans if x >= 0 else 2.0 - ans


def phi(x: float) -> float:
    """Standard normal CDF."""
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return 0.5 * _erfc(-x / math.sqrt(2.0))


def gap_exceeds(ttl_s: float, median_s: float, sigma: float) -> float:
    """P(gap > ttl) for a lognormal gap."""
    if ttl_s <= 0:
        return 1.0
    return 1.0 - phi((math.log(ttl_s) - math.log(median_s)) / sigma)


def expiry_miss(ttl_s: float, req_per_turn: float = M.MEASURED_REQ_PER_TURN) -> float:
    """Share of requests whose cache entry lapsed in the gap before them:
    n_t - 1 tool gaps per human gap."""
    if req_per_turn < 1:
        raise ValueError(f"req_per_turn must be >= 1, got {req_per_turn!r}")
    tool = gap_exceeds(ttl_s, TOOL_GAP_MEDIAN_S, TOOL_GAP_SIGMA)
    human = gap_exceeds(ttl_s, HUMAN_GAP_MEDIAN_S, HUMAN_GAP_SIGMA)
    return ((req_per_turn - 1.0) * tool + human) / req_per_turn


# ---- the clipped lognormal -----------------------------------------------------
def clipped_partial(median: float, sigma: float, lo: float, hi: float,
                    a: float, b: float) -> tuple[float, float]:
    """For L = clip(X, lo, hi), X lognormal(median, sigma): returns
    (P(a < L <= b), E[L; a < L <= b]). The atoms at lo and hi count in the
    band that holds them."""
    if not (median > 0 and sigma > 0):
        raise ValueError("median and sigma must be > 0")
    if lo > hi:
        lo = hi
    a, b = max(a, -math.inf), min(b, math.inf)
    if b <= a:
        return 0.0, 0.0
    mu, s2 = math.log(median), sigma * sigma
    mean_x = median * math.exp(s2 / 2.0)

    def z(t: float) -> float:
        if t <= 0:
            return -math.inf
        if t == math.inf:
            return math.inf
        return (math.log(t) - mu) / sigma

    p = e = 0.0
    # the atom at lo: X <= lo
    if a < lo <= b:
        w = phi(z(lo))
        p += w
        e += lo * w
    # the continuous part on (max(a, lo), min(b, hi)) — open at hi, whose atom
    # (X >= hi) is counted below
    u, v = max(a, lo), min(b, hi)
    if v > u:
        p += phi(z(v)) - phi(z(u))
        e += mean_x * (phi(z(v) - sigma) - phi(z(u) - sigma))
    # the atom at hi: X > hi
    if a < hi <= b and hi > lo:
        w = 1.0 - phi(z(hi))
        p += w
        e += hi * w
    return p, e


# ---- the catalogue -------------------------------------------------------------
@dataclass(frozen=True)
class CacheOption:
    ttl_s: float
    read: float           # $/Mtok for a token read from the cache
    write: float          # $/Mtok for a written token (== input when writes are free)


@dataclass(frozen=True)
class PriceBand:
    """One price band: every token of a request whose prompt is above
    `above_tokens` (and at or below the next band's) bills at these rates."""
    above_tokens: float
    input: float
    output: float
    cache: tuple[CacheOption, ...]


@dataclass(frozen=True)
class ApiOffer:
    key: str
    name: str
    vendor: str
    api_id: str
    bands: tuple[PriceBand, ...]          # ascending above_tokens, the first at 0
    tb40: float | None = None
    ws_model: str | None = None
    meta: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Plan:
    """A coding plan: a price, the API offer whose model it serves (and whose
    list price bills any overage), a meter and budgets in the meter's unit.

    The meter prices one request in the plan's own unit: API-list dollars
    (`meter` None: the offer's own prices), vendor credits (per-token
    multipliers, Z.ai) or requests (Gemini CLI's daily cap). Budgets are per
    window: "5h", "day", "week", "month"."""
    key: str
    name: str
    vendor: str
    usd_month: float                     # list price, monthly billing, before tax
    api: str                             # ApiOffer key
    unit: str                            # "usd" | "credits" | "requests"
    meter: ApiOffer | None               # None: the API offer's own prices
    per_request: float                   # units every request costs on top of its tokens
    budgets: dict                        # window -> units
    overage: str                         # "api" (billed past the budget) | "none" (stops)
    overage_factor: float = 1.0          # overage price over API list
    confidence: str = "opaque"
    meta: dict = field(default_factory=dict, compare=False, hash=False)


WINDOWS = ("5h", "day", "week", "month")


def _band(d: dict, scale: float = 1.0) -> PriceBand:
    """One band of the price file; `scale` is the offer's time-of-day blend
    (1 for every vendor but DeepSeek, research/api_prices.md § 2)."""
    return PriceBand(float(d.get("above_tokens", 0)), float(d["input"]) * scale,
                     float(d["output"]) * scale,
                     tuple(CacheOption(float(c["ttl_s"]), float(c["read"]) * scale,
                                       float(c["write"]) * scale)
                           for c in d["cache"]))


def price_scale(a: dict) -> float:
    """The multiplier an offer's off-peak hours put on its list prices for a
    team working `blend.peak_share` of its hours at peak."""
    b = a.get("time_blend")
    if not b:
        return 1.0
    return b["peak_share"] + (1.0 - b["peak_share"]) * b["offpeak_factor"]


def load_prices() -> dict:
    """The shipped price file (data/api_prices.toml), as parsed TOML."""
    with resources.files("workingset").joinpath("data/api_prices.toml").open("rb") as f:
        return tomllib.load(f)


@lru_cache(maxsize=1)
def catalogue() -> dict:
    """The shipped price file, built: {"fx", "read", "apis", "plans"}."""
    raw = load_prices()
    apis = {}
    for a in raw["apis"]:
        k = price_scale(a)
        bands = tuple(sorted((_band(b, k) for b in a["bands"]), key=lambda b: b.above_tokens))
        if any(len(b.cache) != len(bands[0].cache) for b in bands):
            raise ValueError(f"{a['key']}: every band needs the same cache options")
        if not bands or bands[0].above_tokens != 0:
            raise ValueError(f"{a['key']}: the first price band must start at 0")
        q = a.get("quality") or {}
        apis[a["key"]] = ApiOffer(a["key"], a["name"], a["vendor"], a["api_id"], bands,
                                  q.get("tb40"), a.get("ws_model"), a)
    plans = {}
    for p in raw.get("plans", []):
        if p["api"] not in apis:
            raise ValueError(f"{p['key']}: unknown api {p['api']!r}")
        m = p.get("meter")
        meter = None
        if m is not None:
            k = price_scale(m)
            meter = ApiOffer(p["key"] + ":meter", p["name"], p["vendor"], "",
                             tuple(sorted((_band(b, k) for b in m["bands"]),
                                          key=lambda b: b.above_tokens)))
        budgets = {w: float(v) for w, v in (p.get("budgets") or {}).items()}
        if set(budgets) - set(WINDOWS):
            raise ValueError(f"{p['key']}: unknown budget window in {sorted(budgets)}")
        plans[p["key"]] = Plan(p["key"], p["name"], p["vendor"], float(p["usd_month"]),
                               p["api"], (m or {}).get("unit", "usd"), meter,
                               float((m or {}).get("per_request", 0.0)), budgets,
                               p.get("overage", "none"), float(p.get("overage_factor", 1.0)),
                               p.get("confidence", "opaque"), p)
    return {"fx": raw["fx"], "read": raw["read"], "apis": apis, "plans": plans}


def usd_per_eur() -> float:
    return float(catalogue()["fx"]["usd_per_eur"])


# ---- the per-request cost ------------------------------------------------------
def _class_cost(offer: ApiOffer, opt_index: int, median: float, sigma: float,
                prefix: float, cap: float, delta: float, q: float,
                out_tokens: float) -> float:
    """Expected $ of one request of one class at one cache option."""
    lo = min(prefix, cap)
    total = 0.0
    for i, band in enumerate(offer.bands):
        top = offer.bands[i + 1].above_tokens if i + 1 < len(offer.bands) else math.inf
        p, e = clipped_partial(median, sigma, lo, cap, band.above_tokens if i else -math.inf, top)
        if p <= 0:
            continue
        opt = band.cache[opt_index]
        d = min(delta, lo)                 # a fresh tail never longer than the prompt
        read_tok = (1.0 - q) * (e - d * p)
        write_tok = (1.0 - q) * d * p + q * e
        total += (opt.read * read_tok + opt.write * write_tok
                  + band.output * out_tokens * p) / 1e6
    return total


@dataclass(frozen=True)
class RequestCost:
    usd: float                # mean $ per request, main and subagent mixed
    ttl_s: float              # the cache option chosen
    miss: float               # q at that option
    hit_rate: float           # read tokens / prompt tokens


def request_cost(offer: ApiOffer, wl: M.Workload, turn_tokens: float = 2_000,
                 out_tokens: float = M.OUT_TOKENS_DEFAULT,
                 req_per_turn: float = M.MEASURED_REQ_PER_TURN) -> RequestCost:
    """Mean $ per request on `offer` for workload `wl`, at the cheapest of the
    offer's cache options."""
    if turn_tokens < 0 or out_tokens < 0:
        raise ValueError("turn_tokens and out_tokens must be >= 0")
    s = wl.sub_ratio
    sub_prefix = wl.sys_user if wl.sub_shares_prefix else wl.sys_sub
    best = None
    for k, opt in enumerate(offer.bands[0].cache):
        q = min(1.0, wl.invalidation + (1.0 - wl.invalidation) * expiry_miss(opt.ttl_s, req_per_turn))
        main = _class_cost(offer, k, wl.user_median, wl.user_sigma, wl.sys_user, wl.cap,
                           turn_tokens, q, out_tokens)
        sub = _class_cost(offer, k, wl.sub_median, wl.sub_sigma, sub_prefix, wl.cap,
                          turn_tokens, q, out_tokens)
        usd = (main + s * sub) / (1.0 + s)
        if best is None or usd < best[0] - 1e-15:
            best = (usd, opt.ttl_s, q, opt.read < offer.bands[0].input)
    usd, ttl, q, discounted = best
    # the hit rate the chosen option implies, prompt-token weighted
    lm = clipped_partial(wl.user_median, wl.user_sigma, min(wl.sys_user, wl.cap), wl.cap, -math.inf, math.inf)[1]
    ls = clipped_partial(wl.sub_median, wl.sub_sigma, min(sub_prefix, wl.cap), wl.cap, -math.inf, math.inf)[1]
    lbar = (lm + s * ls) / (1.0 + s)
    # the share of prompt tokens billed at a read price below input: zero on
    # an offer without a cache discount, whatever its prefix reuse
    hit = (1.0 - q) * max(0.0, 1.0 - min(turn_tokens, lbar) / lbar) if discounted else 0.0
    return RequestCost(usd, ttl, q, hit)


def requests_per_hour(wl: M.Workload, think_time_s: float = M.THINK_TIME_S) -> float:
    """Requests one live session sends per hour, subagents included: the
    self-host arrival rate (model.request_rate) for one user."""
    return M.request_rate(1.0, think_time_s, wl.sub_ratio) * 3600.0


@dataclass(frozen=True)
class SeatCost:
    key: str
    name: str
    kind: str                  # "api" | "plan"
    eur_seat: float            # per person with access, per month
    eur_session: float         # per concurrent session at the peak: the table's unit
    usd_request: float
    requests_month: float
    ttl_s: float
    hit_rate: float
    fits: bool = True          # a plan: the workload fits its budgets
    plans_per_seat: float = 1.0
    overflow_eur: float = 0.0  # a plan: extra usage billed past its budgets
    binds: str = ""            # a plan: "5h" | "week" | ""


def api_seat_cost(offer: ApiOffer, wl: M.Workload, *, think_time_s: float = M.THINK_TIME_S,
                  turn_tokens: float = 2_000, out_tokens: float = M.OUT_TOKENS_DEFAULT,
                  active_hours: float = ACTIVE_HOURS_DEFAULT,
                  peak_active_share: float = 1.0,
                  sessions_per_active_user: float = 1.0,
                  usd_eur: float | None = None) -> SeatCost:
    if active_hours < 0:
        raise ValueError(f"active_hours must be >= 0, got {active_hours!r}")
    if not 0 < peak_active_share <= 1:
        raise ValueError("peak_active_share must be in (0, 1]")
    fx = usd_per_eur() if usd_eur is None else usd_eur
    rc = request_cost(offer, wl, turn_tokens, out_tokens)
    n = requests_per_hour(wl, think_time_s) * active_hours * sessions_per_active_user
    eur = n * rc.usd / fx
    return SeatCost(offer.key, offer.name, "api", eur,
                    eur / (peak_active_share * sessions_per_active_user),
                    rc.usd, n, rc.ttl_s, rc.hit_rate)


def plan_seat_cost(plan: Plan, wl: M.Workload, *, think_time_s: float = M.THINK_TIME_S,
                   turn_tokens: float = 2_000, out_tokens: float = M.OUT_TOKENS_DEFAULT,
                   active_hours: float = ACTIVE_HOURS_DEFAULT,
                   peak_active_share: float = 1.0,
                   sessions_per_active_user: float = 1.0,
                   usd_eur: float | None = None) -> SeatCost:
    """A seat on a coding plan (research/api_prices.md § 6).

    The seat sends the API seat's requests; the meter prices each one in the
    plan's unit. A working day is H / 21 hours, worked in consecutive 5-hour
    windows; a week is H / (30/7) hours of it, a month H. A budget the seat
    overspends in some window binds. Past it the excess either bills at API
    list x overage_factor (overage "api": Claude's extra usage, Copilot's
    budget, Mistral's pay-as-you-go) or cannot be served (overage "none"):
    then plans_per_seat is the number of plans the seat's usage needs, which
    most vendors forbid stacking, so read it as the size of the shortfall.
    The budgets overlap, so the overage billed is the largest window's
    excess, a lower bound on the real spill.
    """
    if active_hours < 0:
        raise ValueError(f"active_hours must be >= 0, got {active_hours!r}")
    if not 0 < peak_active_share <= 1:
        raise ValueError("peak_active_share must be in (0, 1]")
    fx = usd_per_eur() if usd_eur is None else usd_eur
    offer = catalogue()["apis"][plan.api]
    lam = requests_per_hour(wl, think_time_s) * sessions_per_active_user
    rc = request_cost(offer, wl, turn_tokens, out_tokens)
    unit_req = (rc.usd if plan.meter is None
                else request_cost(plan.meter, wl, turn_tokens, out_tokens).usd) + plan.per_request
    rate_h, usd_h = lam * unit_req, lam * rc.usd
    hours_day = active_hours / WORKDAYS_PER_MONTH
    full, rem = divmod(hours_day, 5.0)
    win5 = [5.0] * int(full) + ([rem] if rem > 1e-12 else [])
    # window -> (the hours of each window in a month, how many such months hold)
    spans = {"5h": (win5, WORKDAYS_PER_MONTH), "day": ([hours_day], WORKDAYS_PER_MONTH),
             "week": ([active_hours / WEEKS_PER_MONTH], WEEKS_PER_MONTH),
             "month": ([active_hours], 1.0)}
    need, over = {}, {}
    for w, budget in plan.budgets.items():
        hours, times = spans[w]
        need[w] = max(h * rate_h for h in hours) / budget if budget > 0 else math.inf
        # the excess share of each window, billed at the API's own price
        over[w] = times * sum(max(0.0, h * rate_h - budget) / (h * rate_h) * h * usd_h
                              for h in hours if h * rate_h > 0)
    binds = max(need, key=need.get) if need and max(need.values()) > 1.0 else ""
    plans, overflow_eur = 1.0, 0.0
    if binds:
        if plan.overage == "api":
            overflow_eur = max(over.values()) * plan.overage_factor / fx
        else:
            plans = float(math.ceil(need[binds] - 1e-9))
    eur = plan.usd_month / fx * plans + overflow_eur
    return SeatCost(plan.key, plan.name, "plan", eur,
                    eur / (peak_active_share * sessions_per_active_user),
                    rc.usd, lam * active_hours, rc.ttl_s, rc.hit_rate,
                    not binds, plans, overflow_eur, binds)


def comparison(wl: M.Workload | None = None, **kw) -> list[SeatCost]:
    """Every API offer and plan in the catalogue, cheapest seat first."""
    wl = wl or M.Workload()
    cat = catalogue()
    rows = [api_seat_cost(o, wl, **kw) for o in cat["apis"].values()]
    rows += [plan_seat_cost(p, wl, **kw) for p in cat["plans"].values()]
    return sorted(rows, key=lambda r: r.eur_seat)


# ---- ws api -----------------------------------------------------------------------
def _fmt_eur(x: float) -> str:
    return "—" if not math.isfinite(x) else f"€{x:,.0f}" if x >= 100 else f"€{x:,.1f}"


def cmd_api(args) -> int:
    wl, kw = M.Workload(), {}
    if args.config:
        from .config import load_config
        cfg = load_config(args.config)
        w = cfg.workload
        wl = cfg.to_workload()
        kw = dict(think_time_s=w.think_time_s, turn_tokens=w.warm_turn_tokens,
                  out_tokens=w.max_output_tokens)
        if w.headcount is not None:
            kw.update(peak_active_share=w.peak_active_share,
                      sessions_per_active_user=w.sessions_per_active_user)
    kw["active_hours"] = args.hours
    rows = comparison(wl, **kw)
    cat = catalogue()
    if args.json:
        print(json.dumps({"read": cat["read"], "fx": cat["fx"], "active_hours": args.hours,
                          "rows": [r.__dict__ for r in rows]}, indent=2, allow_nan=False,
                         default=lambda x: None))
        return 0
    think = kw.get("think_time_s", M.THINK_TIME_S)
    print(f"one seat, {args.hours:g} active hours a month: "
          f"{requests_per_hour(wl, think) * args.hours:,.0f} requests, "
          f"prompt median {wl.user_median:,.0f} tokens, {kw.get('turn_tokens', 2_000):,.0f} fresh "
          f"and {kw.get('out_tokens', M.OUT_TOKENS_DEFAULT):,.0f} output tokens each")
    print(f"prices read {cat['read']}, before tax; €1 = ${usd_per_eur():g} (ECB {cat['fx']['date']})")
    print()
    print(f"  {'':4} {'offer':48} {'TB 4.0':>7} {'€/seat':>8} {'$/req':>8} {'hit':>6}  plan check")
    for r in rows:
        offer = cat["apis"][r.key] if r.kind == "api" else cat["apis"][cat["plans"][r.key].api]
        tb = offer.tb40
        name = r.name if r.kind == "api" else f"{r.name} · {offer.name}"
        check = ""
        if r.kind == "plan":
            p = cat["plans"][r.key]
            if r.fits:
                check = "fits" if p.budgets else "no published budget"
            elif r.overflow_eur:
                check = f"over the {r.binds} budget: +{_fmt_eur(r.overflow_eur)} at API price"
            else:
                check = f"over the {r.binds} budget: needs {r.plans_per_seat:.0f} plans"
        print(f"  {r.kind:4} {name[:48]:48} {(f'{tb*100:.1f}%' if tb is not None else '—'):>7} "
              f"{_fmt_eur(r.eur_seat):>8} {r.usd_request:8.4f} {r.hit_rate:6.1%}  {check}")
    if kw.get("peak_active_share", 1.0) < 1.0 or kw.get("sessions_per_active_user", 1.0) > 1.0:
        print(f"\n  per concurrent session at the peak (the explorer's €/seat unit): "
              f"divide by {kw['peak_active_share'] * kw['sessions_per_active_user']:g}")
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("api", help="what one seat costs on each API and coding plan",
                       description="Price one agentic-coding seat on every API and coding "
                                   "plan in the shipped catalogue, from CONFIG's workload "
                                   "(the study's reference workload without one). "
                                   "research/api_prices.md has every source.")
    p.add_argument("config", nargs="?", help="workingset.toml / .json (its [workload])")
    p.add_argument("--hours", type=float, default=ACTIVE_HOURS_DEFAULT,
                   help=f"active agent hours per seat per month (default {ACTIVE_HOURS_DEFAULT:g}: "
                        "4 h a day, 21 days)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_api)
