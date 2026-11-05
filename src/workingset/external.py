"""Numbers from outside Working Set: benchmarks, deployments, calculators.

Each file under `data/external/` holds one source and its records (schema:
`data/external/README.md`). A record is one system under test (model, GPU
count, engine, workload) with the metrics the source reported for it, the URL
and quote they can be checked against, the date they were read and their kind:

  measured   a third party ran it under a stated method
  vendor     self-reported by a party with a stake (a seller, a deployer)
  derived    computed by us from other records (note says how)
  guess      an assumption with no measurement behind it

and the use it is put to (method: research/sources/README.md):

  calibrate  full config and known workload: may set a constant by a fit
  check      the model predicts the same quantity; the ratio is reported,
             no constant moves
  context    shown with its provenance, never fitted or checked

Each source also says whether its data may be redistributed (`yes`), only
cited point by point under its terms (`cite`), or not at all (`no`); a `no`
source never has a file in the public data directory.

Nothing here feeds the model's defaults automatically. `spread` reads the
range of one metric across records for a reader to set a bracket against;
`stale` flags records past their source's refresh window.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

KINDS = ("measured", "vendor", "derived", "guess")
USES = ("calibrate", "check", "context")
REDISTRIBUTE = ("yes", "cite", "no")
OPS = ("=", "<", "<=", ">", ">=", "~")
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "external"


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Metric:
    name: str
    value: float | None
    unit: str = ""
    op: str = "="
    low: float | None = None
    high: float | None = None
    at: str | None = None        # the condition it holds at (tier, concurrency)
    kind: str | None = None      # overrides the record's kind
    use: str | None = None       # overrides the record's use
    band: str | None = None      # low/high are "reported" by the source or "assumed" by us


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    url: str
    kind: str
    refresh_days: int
    redistribute: str = "yes"
    terms: str = ""              # the clause that sets `redistribute`, quoted
    publisher: str = ""
    license: str = ""
    method: str = ""


@dataclass(frozen=True)
class Record:
    id: str
    source: Source
    url: str
    read: dt.date
    kind: str
    use: str
    config: dict[str, Any]
    metrics: tuple[Metric, ...]
    workload: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()           # record ids a derived record is computed from
    mapping: dict[str, Any] = field(default_factory=dict)  # model function + args that predict it
    published: str | None = None   # ISO date, or YYYY-MM when the source gives only a month
    quote: str = ""
    note: str = ""

    def metric(self, name: str) -> Metric | None:
        return next((m for m in self.metrics if m.name == name), None)

    def kind_of(self, m: Metric) -> str:
        return m.kind or self.kind

    def use_of(self, m: Metric) -> str:
        return m.use or self.use


def _date(v: Any, where: str) -> dt.date:
    try:
        return dt.date.fromisoformat(v)
    except (TypeError, ValueError):
        raise SchemaError(f"{where}: not an ISO date: {v!r}") from None


def _published(v: Any, where: str) -> str | None:
    if not v:
        return None
    if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}", v):
        return v
    return _date(v, where).isoformat()


def _need(d: dict, keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if d.get(k) in (None, "")]
    if missing:
        raise SchemaError(f"{where}: missing {', '.join(missing)}")


def _num(v: Any, where: str) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SchemaError(f"{where}: not a number: {v!r}")
    return float(v)


def _kind(v: Any, where: str) -> str:
    if v not in KINDS:
        raise SchemaError(f"{where}: kind {v!r} not in {KINDS}")
    return v


def _enum(v: Any, allowed: tuple, what: str, where: str) -> str:
    if v not in allowed:
        raise SchemaError(f"{where}: {what} {v!r} not in {allowed}")
    return v


def _metric(d: dict, where: str) -> Metric:
    _need(d, ("name",), where)
    w = f"{where} {d['name']}"
    op = d.get("op", "=")
    if op not in OPS:
        raise SchemaError(f"{w}: op {op!r} not in {OPS}")
    m = Metric(name=d["name"], value=_num(d.get("value"), w), unit=d.get("unit", ""),
               op=op, low=_num(d.get("low"), w), high=_num(d.get("high"), w),
               at=d.get("at"), kind=_kind(d["kind"], w) if "kind" in d else None,
               use=_enum(d["use"], USES, "use", w) if "use" in d else None,
               band=_enum(d["band"], ("reported", "assumed"), "band", w) if "band" in d else None)
    if m.value is None and m.low is None and m.high is None:
        raise SchemaError(f"{w}: needs value, low or high")
    return m


def parse(doc: dict, where: str = "<doc>") -> list[Record]:
    """Validate one source document and return its records."""
    s = doc.get("source") or {}
    _need(s, ("id", "name", "url", "kind", "refresh_days"), f"{where} source")
    src = Source(id=s["id"], name=s["name"], url=s["url"],
                 kind=_kind(s["kind"], f"{where} source"),
                 refresh_days=int(s["refresh_days"]),
                 redistribute=_enum(s.get("redistribute", "yes"), REDISTRIBUTE,
                                    "redistribute", f"{where} source"),
                 terms=s.get("terms", ""), publisher=s.get("publisher", ""),
                 license=s.get("license", ""), method=s.get("method", ""))
    out = []
    for i, r in enumerate(doc.get("records") or []):
        w = f"{where} record {r.get('id', i)}"
        _need(r, ("id", "url", "read", "config", "metrics"), w)
        if not r["id"].startswith(src.id + "/"):
            raise SchemaError(f"{w}: id must start with '{src.id}/'")
        if not r["metrics"]:
            raise SchemaError(f"{w}: no metrics")
        kind = _kind(r.get("kind", src.kind), w)
        inputs = tuple(r.get("inputs") or ())
        if kind == "derived" and not inputs and not r.get("note"):
            raise SchemaError(f"{w}: a derived record names its inputs or its formula")
        out.append(Record(
            id=r["id"], source=src, url=r["url"], read=_date(r["read"], w),
            kind=kind, use=_enum(r.get("use", "context"), USES, "use", w),
            config=r["config"], inputs=inputs, mapping=r.get("mapping") or {},
            metrics=tuple(_metric(m, w) for m in r["metrics"]),
            workload=r.get("workload") or {},
            published=_published(r.get("published"), w),
            quote=r.get("quote", ""), note=r.get("note", "")))
    return out


def load(directory: Path | str = DEFAULT_DIR, *, public: bool | None = None) -> list[Record]:
    """Every record under `directory`, validated; ids unique, derived inputs
    resolvable. `public` (default: True for the repo's own directory) refuses
    a source whose terms forbid redistribution."""
    directory = Path(directory)
    if public is None:
        public = directory.resolve() == DEFAULT_DIR
    recs: list[Record] = []
    for p in sorted(directory.glob("*.json")):
        got = parse(json.loads(p.read_text()), p.name)
        if public and got and got[0].source.redistribute == "no":
            raise SchemaError(f"{p.name}: source {got[0].source.id} may not be "
                              "redistributed; keep it out of the public data directory")
        recs += got
    ids = {r.id for r in recs}
    if len(ids) != len(recs):
        dup = sorted({r.id for r in recs if sum(q.id == r.id for q in recs) > 1})
        raise SchemaError(f"duplicate record ids {dup}")
    for r in recs:
        missing = [i for i in r.inputs if i not in ids]
        if missing:
            raise SchemaError(f"{r.id}: inputs not found: {missing}")
    return recs


def stale(r: Record, today: dt.date | None = None) -> bool:
    """True once the record is older than its source's refresh window."""
    today = today or dt.date.today()
    return (today - r.read).days > r.source.refresh_days


def select(recs: Iterable[Record], *, metric: str | None = None,
           **config: Any) -> list[Record]:
    """Records that carry `metric` and whose config matches every keyword."""
    return [r for r in recs
            if (metric is None or r.metric(metric) is not None)
            and all(r.config.get(k) == v for k, v in config.items())]


@dataclass(frozen=True)
class Range:
    metric: str
    low: float
    high: float
    n: int
    ids: tuple[str, ...]


def spread(recs: Iterable[Record], metric: str, *,
           kinds: tuple[str, ...] = ("measured", "derived"),
           **config: Any) -> Range | None:
    """Range of `metric` over the matching records (point values and
    `low`/`high` bounds alike). Vendor claims and guesses are left out unless
    `kinds` asks for them. A range to read a constant's bracket against, not
    a prior: only a `calibrate` record's fit moves a constant."""
    vals: list[float] = []
    ids: list[str] = []
    for r in select(recs, metric=metric, **config):
        m = r.metric(metric)
        if r.kind_of(m) not in kinds:
            continue
        pts = [v for v in (m.value, m.low, m.high) if v is not None]
        vals += pts
        ids.append(r.id)
    if not vals:
        return None
    return Range(metric, min(vals), max(vals), len(ids), tuple(ids))


def curves(recs: Iterable[Record], metric: str = "tok_s_per_user",
           x: str = "concurrency", **config: Any) -> dict[str, list[tuple[float, float]]]:
    """`metric` against `x` per serving config (`config.extra.curve`), sorted
    by `x`: the measured counterpart of the model's decode curve."""
    out: dict[str, list[tuple[float, float]]] = {}
    for r in select(recs, metric=metric, **config):
        cid = (r.config.get("extra") or {}).get("curve")
        xm, ym = r.metric(x), r.metric(metric)
        if cid and xm is not None and xm.value is not None and ym.value is not None:
            out.setdefault(cid, []).append((xm.value, ym.value))
    return {k: sorted(v) for k, v in out.items()}
