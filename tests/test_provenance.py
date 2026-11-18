"""Every model and GPU quantity that feeds a verdict says where it came from."""
from pathlib import Path

import pytest

from workingset import model as M
from workingset import provenance as P

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key", list(M.CATALOG["models"]))
def test_every_model_quantity_has_a_provenance(key):
    for q in P.required_model_quantities(key):
        P.model_provenance(key, q)


@pytest.mark.parametrize("key", list(M.CATALOG["gpus"]))
def test_every_gpu_quantity_has_a_provenance(key):
    for q in P.required_gpu_quantities(key):
        P.gpu_provenance(key, q)


def test_a_missing_provenance_is_an_error(monkeypatch):
    row = dict(M.CATALOG["models"]["27B"])
    row["provenance"] = {q: v for q, v in row["provenance"].items() if q != "kv"}
    monkeypatch.setitem(M.CATALOG["models"], "27B", row)
    with pytest.raises(P.ProvenanceError, match=r"models\.27B\.provenance\] kv: missing"):
        P.labels()


def test_kinds_are_the_external_vocabulary():
    with pytest.raises(P.ProvenanceError, match="kind 'estimated'"):
        P._parse({"kind": "estimated", "note": "x"}, "here")


def test_every_ref_is_a_file_in_the_repo():
    for table in ("gpus", "models"):
        for key, row in M.CATALOG[table].items():
            for q, p in row["provenance"].items():
                if "ref" in p:
                    assert (ROOT / p["ref"]).is_file(), f"{table}.{key}.{q}: {p['ref']}"


def test_transfers_name_a_gpu_or_model():
    gpus = {g.name for g in M.GPUS.values()} | {"H100"}
    for table in ("gpus", "models"):
        for key, row in M.CATALOG[table].items():
            for q, p in row["provenance"].items():
                if "on" in p:
                    assert p["on"] in gpus, f"{table}.{key}.{q}: on {p['on']!r}"


def test_acceptance_measured_agrees_with_the_provenance():
    # the flag predates the provenance table; they must not disagree. The flag
    # counts a predecessor of the same shape (Qwen3.6-27B for Qwen3.8-27B) as
    # the model's own; the provenance still says the value was transferred
    for key, m in M.MODELS.items():
        p = P.model_provenance(key, "acceptance")
        own = p.kind == "measured" and p.model not in M.MODELS
        assert m.acceptance_measured == own, key


def test_labels():
    lab = P.labels()
    assert lab["H200"]["gpu"]["reserve"] == "measured on this GPU"
    assert lab["B300"]["gpu"]["reserve"] == "fitted on H200 and transferred"
    assert lab["H200"]["models"]["35BA3B"]["acceptance"] == "fitted on Qwen3.8-27B and transferred"
    assert lab["B300"]["models"]["27B"]["kv"] == "derived from the config"
    assert lab["B300"]["gpu"]["prefill"] == "guess"
    assert "weights_nvfp4" not in lab["H200"]["models"]["K31P"]
