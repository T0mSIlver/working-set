import datetime as dt

import pytest

from workingset import external as X


def test_every_data_file_validates():
    recs = X.load()
    assert recs, "data/external/ holds no records"
    for r in recs:
        assert r.url.startswith("http"), r.id


def _doc(**metric):
    return {
        "source": {"id": "s", "name": "S", "url": "https://s", "kind": "measured",
                   "refresh_days": 30},
        "records": [
            {"id": "s/a", "url": "https://s/a", "read": "2026-11-01",
             "config": {"gpu": "H200"}, "metrics": [{"name": "mbu", "value": 0.3}]},
            {"id": "s/b", "url": "https://s/b", "read": "2026-11-01", "kind": "vendor",
             "config": {"gpu": "H200"}, "metrics": [{"name": "mbu", "value": 0.9}]},
            {"id": "s/c", "url": "https://s/c", "read": "2026-11-01",
             "config": {"gpu": "H200"}, "metrics": [{"name": "mbu", **metric}]},
        ],
    }


def test_spread_spans_bounds_and_drops_vendor_claims():
    rng = X.spread(X.parse(_doc(low=0.2, high=0.4)), "mbu", gpu="H200")
    assert (rng.low, rng.high, rng.ids) == (0.2, 0.4, ("s/a", "s/c"))


def test_stale_after_refresh_window():
    r = X.parse(_doc(value=0.3))[0]
    assert not X.stale(r, dt.date(2026, 12, 1))
    assert X.stale(r, dt.date(2026, 12, 2))


@pytest.mark.parametrize("metric", [{"value": "0.3"}, {"op": "≈", "value": 1}, {}])
def test_bad_metric_rejected(metric):
    with pytest.raises(X.SchemaError):
        X.parse(_doc(**metric))


def test_no_redistribution_source_refused_in_public_dir(tmp_path):
    import json
    d = _doc(value=0.3)
    d["source"]["redistribute"] = "no"
    (tmp_path / "s.json").write_text(json.dumps(d))
    with pytest.raises(X.SchemaError):
        X.load(tmp_path, public=True)
    assert len(X.load(tmp_path, public=False)) == 3


def test_curves_group_points_by_serving_config():
    d = _doc(value=0.3)
    for i, (conc, tps) in enumerate([(8, 300.0), (1, 345.0)]):
        d["records"][i]["config"]["extra"] = {"curve": "glm-b300"}
        d["records"][i]["metrics"] = [{"name": "concurrency", "value": conc},
                                      {"name": "tok_s_per_user", "value": tps}]
    assert X.curves(X.parse(d)) == {"glm-b300": [(1, 345.0), (8, 300.0)]}
