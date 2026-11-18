"""ws market: rent against provider prices. Runs on tests/fixtures/market
only (fake sellers and providers), never on data/market."""
import json
import math
import shutil
from pathlib import Path

import pytest

from workingset import api_cost as A
from workingset import market as MK

FIX = Path(__file__).parent / "fixtures" / "market"


@pytest.fixture(scope="module")
def snap():
    return MK.load_snapshot(FIX)


@pytest.fixture(scope="module")
def glm(snap):
    wv = MK.default_workload()
    return MK.compare("GLM52", snap, gpus=("B300",), evaluator=MK.Evaluator(wv, n_iter=100))


def _provider(res, name):
    return next(p for p in res["providers"] if p["provider"] == name)


def test_flows_partition_the_prompt():
    wl = MK.default_workload().wl
    for q in (0.0, 0.3, 1.0):
        f = A.request_flows(wl, q)
        assert f.read + f.written == pytest.approx(f.prompt)
    assert A.request_flows(wl, 1.0).read == 0.0
    # a hit writes only the fresh tail, never more than the 2,000-token turn
    assert A.request_flows(wl, 0.0).written <= 2_000 + 1e-9


def test_rent_levels_over_sellers_only(snap):
    lv = MK.rent_levels(snap["rental"], "H200", "on-demand")
    assert (lv["low"], lv["median"], lv["high"]) == (2.0, 3.0, 4.0)
    assert lv["n_sellers"] == 3 and lv["n_aggregators_excluded"] == 1
    assert MK.rent_levels(snap["rental"], "B300", "on-demand")["median"] == 5.0


def test_rent_one_figure_per_seller_and_nulls_recorded():
    rental = {"records": [
        {"gpu": "H200", "seller": "s", "kind": "seller", "term": "on-demand", "usd_per_gpu_h": 5.0},
        {"gpu": "H200", "seller": "s", "kind": "seller", "term": "on-demand", "usd_per_gpu_h": 3.0},
        {"gpu": "H200", "seller": "t", "kind": "seller", "term": "on-demand", "usd_per_gpu_h": None}]}
    lv = MK.rent_levels(rental, "H200", "on-demand")
    assert lv["n_sellers"] == 1 and lv["low"] == lv["high"] == 3.0
    assert lv["sellers_without_price"] == ["t"]


def test_price_per_session_hour_without_a_cache_price(glm):
    """No cache-read price: every prompt token bills at input, whatever the
    cache does, so the revenue is closed form."""
    wv = MK.default_workload()
    p = _provider(glm, "fixture-b")
    want = wv.req_per_hour * (0.5 * wv.prompt_mean + 2.0 * wv.out) / 1e6
    assert p["usd_per_session_hour"] == pytest.approx(want, rel=1e-9)
    assert p["usd_per_mtok_mixed"] == pytest.approx(
        want / (wv.req_per_hour * (wv.prompt_mean + wv.out)) * 1e6)
    assert p["cache_assumption"].startswith("no cache-read price")


def test_u_star_is_rent_per_session_over_price(glm):
    p = _provider(glm, "fixture-a")
    cfgs = {c["id"]: c for c in glm["configs"]}
    for level in MK.LEVELS:
        b = p["u_star"]["on-demand"][level]
        c = cfgs[b["config"]]
        cost = c["cost"]["on-demand"][level]
        assert cost["usd_per_hour"] == c["n_gpu"] * cost["usd_per_gpu_h"]
        assert b["u_star"] == pytest.approx(cost["usd_per_hour"] / c["limit"]
                                            / p["usd_per_session_hour"])
        # the cheapest configuration over the arms the endpoint maps to
        same = [x for x in glm["configs"] if x["arm"] in p["arms"]]
        assert c["cost"]["on-demand"][level]["usd_per_session_hour_full"] == min(
            x["cost"]["on-demand"][level]["usd_per_session_hour_full"] for x in same)
    us = [p["u_star"]["on-demand"][lv]["u_star"] for lv in MK.LEVELS]
    assert us == sorted(us)


def test_cost_at_utilisation(glm):
    c = glm["configs"][0]
    med = c["cost"]["on-demand"]["median"]
    for k, at in med["at"].items():
        u = float(k)
        assert at["usd_per_session_hour"] == pytest.approx(med["usd_per_hour"] / (c["limit"] * u))
    assert med["at"]["0.2"]["usd_per_mtok_mixed"] == pytest.approx(
        4 * med["at"]["0.8"]["usd_per_mtok_mixed"])


def test_split_cost_convention():
    d = {"d_prefill": 0.2, "d_decode": 0.2, "fresh_per_request": 1_000.0}
    # half the hour each way; 10 sessions x 100 req/h x 1,000 fresh, x 500 out
    s = MK.split_cost(10.0, 10, 1.0, d, 100.0, 500.0)
    assert s["usd_per_mtok_input"] == pytest.approx(5.0 / 1e6 * 1e6)
    assert s["usd_per_mtok_output"] == pytest.approx(5.0 / 5e5 * 1e6)


def test_missing_price_is_recorded_not_priced(glm):
    p = _provider(glm, "fixture-c")
    assert p["usd_per_session_hour"] is None
    assert all(p["u_star"]["on-demand"][lv] is None for lv in MK.LEVELS)


def test_context_below_cap_is_flagged(glm):
    p = _provider(glm, "fixture-b")
    assert p["fits_context"] is False
    assert 0 < p["share_requests_over_context"] < 1
    assert _provider(glm, "fixture-a")["fits_context"] is True


def test_flags_and_arms(glm):
    b = _provider(glm, "fixture-b")
    assert any("discount" in f for f in b["flags"])
    assert b["arms"] == ["fp8", "nvfp4"]                  # declared unknown
    assert _provider(glm, "fixture-a")["arms"] == ["fp8"]
    assert MK._arms_for("K31P", "mxfp4")[0] == ["fp8"]   # native MXFP4 base arm
    arms, note = MK._arms_for("GLM52", "bf16")
    assert arms == ["fp8", "nvfp4"] and "lower bound" in note


def test_unchecked_model_says_so(glm):
    for c in glm["configs"]:
        assert c["error"]["status"] == "unchecked"
        assert c["error"]["limit_low"] is None


def test_json_is_strict(glm):
    json.dumps(glm, allow_nan=False, default=MK._json_default)


def test_fail_loud_on_a_missing_field(tmp_path):
    shutil.copy(FIX / "gpu_rental-2026-11-18.json", tmp_path)
    raw = json.loads((FIX / "provider_prices.json").read_text())
    del raw["models"][0]["endpoints"][0]["usd_per_mtok_cache_read"]
    (tmp_path / "provider_prices.json").write_text(json.dumps(raw))
    snap = MK.load_snapshot(tmp_path)
    with pytest.raises(MK.MarketDataError, match="usd_per_mtok_cache_read"):
        MK.provider_rows(snap, "GLM52", MK.default_workload(), 300.0)


def test_snapshot_errors(tmp_path):
    with pytest.raises(MK.MarketDataError, match="does not exist"):
        MK.load_snapshot(tmp_path / "nope")
    with pytest.raises(MK.MarketDataError, match="no price file"):
        MK.load_snapshot(tmp_path)
    shutil.copy(FIX / "provider_prices.json", tmp_path)
    with pytest.raises(MK.MarketDataError, match="gpu_rental"):
        MK.load_snapshot(tmp_path)


def test_rows_shape_is_read(tmp_path):
    """provider_prices.json as one row per (model, provider, tier)."""
    shutil.copy(FIX / "gpu_rental-2026-11-18.json", tmp_path)
    ep = json.loads((FIX / "provider_prices.json").read_text())["models"][0]["endpoints"][0]
    (tmp_path / "provider_prices.json").write_text(json.dumps(
        {"source": {"id": "fixture-rows"}, "rows": [{**ep, "ws_model": "GLM52", "tier": "list"}]}))
    rows = MK.provider_rows(MK.load_snapshot(tmp_path), "GLM52", MK.default_workload(), 300.0)
    assert len(rows) == 1 and rows[0]["source"] == "fixture-rows" and rows[0]["tag"] == "list"
