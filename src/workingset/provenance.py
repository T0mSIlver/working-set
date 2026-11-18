"""Where each model and GPU constant comes from.

Every quantity that feeds a verdict carries a provenance in
`data/catalog.toml`, under `[models.KEY.provenance]` and
`[gpus.KEY.provenance]`, one inline table per quantity:

  kind   data/external's vocabulary (external.KINDS):
           measured  read off a run, a log or the checkpoint's shard headers,
                     by us or a third party under a stated method
           vendor    a figure from a party with a stake: a datasheet, a model
                     card, a seller's price
           derived   computed by us from the model's config or checkpoint
                     (basis says which) or from other figures
           guess     an assumption with no measurement behind it
  basis  derived only: "config" or "checkpoint"
  on     measured only: the GPU it was measured or fitted on, when the
         quantity depends on the GPU
  model  measured only: the model it was measured on, when not this one
  note   how, in one line; ref: the research note or doc that has the rest

A group of constants (the decode constants, the power rows) takes the kind of
its weakest member; the note names the stronger ones.

`label` turns one provenance into the words the explorer shows for the GPU
on screen: a value measured on another GPU or another model reads "fitted on
X and transferred". A quantity without a provenance is an error.
"""
from __future__ import annotations

from dataclasses import dataclass

from workingset.external import KINDS
from workingset.model import CATALOG

# the quantities that feed a verdict, with the words the explorer uses
MODEL_QUANTITIES = {
    "weights": "weights",
    "weights_nvfp4": "NVFP4 weights",
    "kv": "KV bytes per token",
    "decode": "decode bytes",
    "draft": "draft pass",
    "acceptance": "draft acceptance",
    "prefill": "prefill FLOPs",
    "quality": "quality score",
}
GPU_QUANTITIES = {
    "memory": "GPU memory",
    "reserve": "pool reserve",
    "decode": "decode constants",
    "prefill": "prefill constants",
    "power": "power",
    "price": "price",
}
BASES = ("config", "checkpoint")
FIELDS = ("kind", "basis", "on", "model", "note", "ref")


class ProvenanceError(ValueError):
    pass


@dataclass(frozen=True)
class Provenance:
    kind: str
    note: str
    ref: str = ""
    basis: str | None = None
    on: str | None = None
    model: str | None = None


def _parse(row: dict, where: str) -> Provenance:
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise ProvenanceError(f"{where}: unknown fields {sorted(unknown)}")
    if row.get("kind") not in KINDS:
        raise ProvenanceError(f"{where}: kind {row.get('kind')!r} not in {KINDS}")
    if not row.get("note"):
        raise ProvenanceError(f"{where}: no note saying how the value was obtained")
    kind = row["kind"]
    if kind == "derived" and row.get("basis") not in BASES:
        raise ProvenanceError(f"{where}: a derived value needs basis in {BASES}")
    if kind != "derived" and "basis" in row:
        raise ProvenanceError(f"{where}: basis is for derived values only")
    for f in ("on", "model"):
        if f in row and kind != "measured":
            raise ProvenanceError(f"{where}: {f} is for measured values only")
    return Provenance(**row)


def required_model_quantities(key: str) -> tuple[str, ...]:
    row = CATALOG["models"][key]
    return tuple(q for q in MODEL_QUANTITIES
                 if q != "weights_nvfp4" or row.get("nvfp4_w") is not None)


def required_gpu_quantities(key: str) -> tuple[str, ...]:
    return tuple(GPU_QUANTITIES)


def _lookup(table: str, key: str, quantity: str) -> Provenance:
    rows = CATALOG[table].get(key)
    if rows is None:
        raise ProvenanceError(f"no {table[:-1]} {key!r} in the catalogue")
    where = f"catalog.toml [{table}.{key}.provenance] {quantity}"
    row = (rows.get("provenance") or {}).get(quantity)
    if row is None:
        raise ProvenanceError(f"{where}: missing; every quantity that feeds a "
                              f"verdict needs a kind and a note")
    return _parse(row, where)


def model_provenance(key: str, quantity: str) -> Provenance:
    if quantity not in MODEL_QUANTITIES:
        raise ProvenanceError(f"unknown model quantity {quantity!r}")
    return _lookup("models", key, quantity)


def gpu_provenance(key: str, quantity: str) -> Provenance:
    if quantity not in GPU_QUANTITIES:
        raise ProvenanceError(f"unknown GPU quantity {quantity!r}")
    return _lookup("gpus", key, quantity)


def _model_name(key: str) -> str:
    row = CATALOG["models"].get(key)
    return row["explorer"]["short"] if row else key


def label(p: Provenance, gpu: str, model: str | None = None) -> str:
    """The words for `p` with `gpu` on screen; `model` is the model whose
    quantity it is (None for a GPU quantity)."""
    if p.kind == "measured":
        if p.model is not None and p.model != model:
            return f"fitted on {_model_name(p.model)} and transferred"
        if p.on is not None and p.on != gpu:
            return f"fitted on {p.on} and transferred"
        return "measured on this GPU" if p.on is not None else "measured"
    if p.kind == "derived":
        return f"derived from the {p.basis}"
    return {"vendor": "vendor figure", "guess": "guess"}[p.kind]


def labels() -> dict:
    """{gpu: {"gpu": {quantity: label}, "models": {model: {quantity: label}}}}
    for every GPU and model in the catalogue. Raises on the first quantity
    without a provenance."""
    out = {}
    for g in CATALOG["gpus"]:
        out[g] = {
            "gpu": {q: label(gpu_provenance(g, q), g) for q in required_gpu_quantities(g)},
            "models": {m: {q: label(model_provenance(m, q), g, m)
                           for q in required_model_quantities(m)}
                       for m in CATALOG["models"]},
        }
    return out
