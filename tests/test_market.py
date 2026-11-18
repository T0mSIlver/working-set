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
    assert lv["other_tiers_excluded"] == ["fixture-d (community)"]
    b = MK.rent_levels(snap["rental"], "B300", "on-demand")
    # fixture-e is GB300 and never counts; fixture-f is sold out and counts
    assert (b["low"], b["median"], b["high"]) == (4.0, 6.0, 7.0)
    assert b["gb300_excluded"] == ["fixture-e"] and b["sold_out"] == ["fixture-f"]
    a = MK.rent_levels(snap["rental"], "H200", "on-demand", "all")
    assert a["low"] == 1.0 and a["n_sellers"] == 4


def test_rent_one_figure_per_seller_and_nulls_recorded():
    rental = {"records": [
        {"gpu": "H200", "seller": "s", "kind": "seller", "tier": "managed", "platform": "HGX", "term": "on-demand", "usd_per_gpu_h": 5.0},
        {"gpu": "H200", "seller": "s", "kind": "seller", "tier": "managed", "platform": "HGX", "term": "on-demand", "usd_per_gpu_h": 3.0},
        {"gpu": "H200", "seller": "t", "kind": "seller", "tier": "managed", "platform": "HGX", "term": "on-demand", "usd_per_gpu_h": None}]}
    lv = MK.rent_levels(rental, "H200", "on-demand")
    assert lv["n_sellers"] == 1 and lv["low"] == lv["high"] == 3.0
    assert lv["sellers_without_price"] == ["t"]


def test_price_per_session_hour_without_a_cache_price(glm):
    """No cache-read price: every prompt token bills at input, whatever the
    cache does, so the revenue is closed form."""
    wv = MK.default_workload()
    p = _provider(glm, "fixture-b")
    want = wv.req_per_hour * (0.5 * wv.prompt_mean + 2.0 * wv.out) / 1e6
    assert p["u_star_no_cache_discount"]["on-demand"]["median"] == pytest.approx(
        p["u_star"]["on-demand"]["median"]["u_star"])
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
    assert any("after a published promotion of 50%" in f for f in b["flags"])
    assert b["arms"] == ["fp8", "nvfp4"]                  # declared unknown
    assert _provider(glm, "fixture-a")["arms"] == ["fp8"]
    assert MK._arms_for("K31P", "mxfp4")[0] == ["fp8"]   # native MXFP4 base arm
    arms, note = MK._arms_for("GLM52", "bf16")
    assert arms == ["fp8", "nvfp4"] and "lower bound" in note


def test_unchecked_model_says_so(glm):
    for c in glm["configs"]:
        assert c["error"]["status"] == "unchecked"
        assert c["error"]["limit_low"] is None
    for p in glm["providers"]:
        if p["u_star"]["on-demand"]["median"]:
            assert p["u_star"]["on-demand"]["model_error"] == {"status": "capacity unchecked"}


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


@pytest.fixture(scope="module")
def glm_sens(snap):
    wv = MK.default_workload()
    ev = MK.Evaluator(wv, n_iter=100)
    res = MK.compare("GLM52", snap, gpus=("B300",), evaluator=ev)
    return MK.sensitivity(res, snap, ev)


def test_sensitivity_identities(glm_sens):
    a = {x["key"]: x for x in glm_sens["assumptions"]}
    # u* = C / L / R: rent elasticity +1, the limit's -1, exactly
    assert a["rent"]["median_elasticity"] == pytest.approx(1.0)
    assert a["limit"]["median_elasticity"] == pytest.approx(-1.0)
    mags = [x["median_abs_dln"] or 0 for x in glm_sens["assumptions"]]
    assert mags == sorted(mags, reverse=True)


def test_cache_switch_moves_only_a_discounted_endpoint(glm_sens):
    p = {x["provider"]: x for x in glm_sens["providers"]}
    # fixture-b publishes no cache price: billing cached tokens at input changes nothing
    assert p["fixture-b"]["abs_dln_by_assumption"]["cache_as_input"] == pytest.approx(0.0, abs=1e-12)
    # fixture-a's cache read is a fifth of input: u* falls when it earns full input
    assert p["fixture-a"]["abs_dln_by_assumption"]["cache_as_input"] > 0.5
    assert p["fixture-a"]["most_sensitive_to"] not in ("rent", "limit")


def test_committed_split_by_length(snap):
    short = MK.rent_levels(snap["rental"], "H200", "committed-short")
    long_ = MK.rent_levels(snap["rental"], "H200", "committed-long")
    assert short["n_sellers"] == 0 and long_["n_sellers"] == 1
    assert long_["commit_months"] == [12]
    rental = {"records": [
        {"gpu": "H200", "seller": "s", "kind": "seller", "tier": "managed", "platform": "HGX", "term": "committed",
         "commit_months": 1, "usd_per_gpu_h": 4.0},
        {"gpu": "H200", "seller": "t", "kind": "seller", "tier": "managed", "platform": "HGX", "term": "committed",
         "commit_months": 6, "usd_per_gpu_h": 3.0},
        {"gpu": "H200", "seller": "v", "kind": "seller", "tier": "managed", "platform": "HGX", "term": "committed",
         "commit_months": None, "usd_per_gpu_h": 3.0}]}
    s = MK.rent_levels(rental, "H200", "committed-short")
    assert s["low"] == 4.0 and s["commitments_left_out"] == ["t", "v"]
    assert MK.rent_levels(rental, "H200", "committed-long")["n_sellers"] == 0


def test_rent_record_without_tier_fails_loudly():
    rental = {"records": [{"gpu": "H200", "seller": "s", "kind": "seller", "platform": "HGX",
                           "term": "on-demand", "usd_per_gpu_h": 3.0}]}
    with pytest.raises(MK.MarketDataError, match=r"records\[0\].*'tier'"):
        MK.rent_levels(rental, "H200", "on-demand")
    rental["records"][0]["tier"] = "boutique"
    with pytest.raises(MK.MarketDataError, match="tier 'boutique'"):
        MK.rent_levels(rental, "H200", "on-demand")


def test_time_weighted_price_is_priced(tmp_path):
    shutil.copy(FIX / "gpu_rental-2026-11-18.json", tmp_path)
    raw = json.loads((FIX / "provider_prices.json").read_text())
    ep = raw["models"][0]["endpoints"][1]           # fixture-b: no cache price
    ep["time_weighted"] = {"usd_per_mtok_input": 1.0, "usd_per_mtok_output": 4.0,
                           "usd_per_mtok_cache_read": None}
    (tmp_path / "provider_prices.json").write_text(json.dumps(raw))
    wv = MK.default_workload()
    row = MK.provider_rows(MK.load_snapshot(tmp_path), "GLM52", wv, 300.0)[1]
    # every price doubled over the listed window: twice the revenue
    assert row["usd_per_session_hour"] == pytest.approx(2 * row["usd_per_session_hour_listed_window"])
    assert any("time-weighted" in f for f in row["flags"])


def test_no_cache_contrast_bills_every_prompt_token_at_input(snap):
    ev = MK.evaluator_for("no-cache", n_iter=50)
    res = MK.compare("GLM52", snap, gpus=("B300",), evaluator=ev, workload_name="no-cache")
    wv = ev.wv
    p = _provider(res, "fixture-a")
    want = wv.req_per_hour * (1.0 * wv.prompt_mean + 4.0 * wv.out) / 1e6
    assert p["usd_per_session_hour"] == pytest.approx(want, rel=1e-9)
    assert res["workload"]["mix_at_miss_rate"]["cached"] == 0.0
    # the cap is explicit, not predict()'s recommendation of 1
    c = res["configs"][0]
    cfg = ev.limit("GLM52", c["gpu"], c["tp"], c["arm"])["cfg"]
    assert cfg.deployment.max_num_seqs > 1


def test_no_cache_bills_writes_at_input_not_the_cache_write_price():
    # Alibaba-like sheet: a cache write dearer than input. With nothing
    # reused, the prompt is billed at input; the default keeps the write price.
    ep = {"usd_per_mtok_input": 0.15, "usd_per_mtok_output": 0.47,
          "usd_per_mtok_cache_read": 0.03, "usd_per_mtok_cache_write": 0.20}
    default, _ = MK._offer(ep, "x", 300.0)
    fresh, note = MK._offer(ep, "x", 300.0, no_cache=True)
    assert default.bands[0].cache[0].write == 0.20
    assert fresh.bands[0].cache[0].write == 0.15
    assert note.startswith("no cache")


def test_empty_sensitivity_renders_its_note():
    text = MK.render_sensitivity({"term": "committed-short", "level": "median",
                                  "assumptions": [], "providers": [],
                                  "note": "no endpoint has a u* here"})
    assert "no endpoint has a u* here" in text


def test_fidelity_buckets(snap):
    rows = MK.provider_rows(snap, "GLM52", MK.default_workload(), 300.0)
    f = MK.fidelity_summary("GLM52", rows)
    assert (f["rows"], f["providers"], f["released_dtype"]) == (3, 3, "fp8")
    # fixture-a and -c offer the whole 1,048,576 tokens; fixture-b 131,072 = 12.5%
    assert f["context"]["100%"] == {"rows": 2, "providers": 2}
    assert f["context"]["under 25%"] == {"rows": 1, "providers": 1}
    # 131,072 and 32,768 are inside 32k-128k; fixture-c publishes none
    assert f["max_output"]["32k-128k"]["rows"] == 2
    assert f["max_output"]["none"]["rows"] == 1
    # fp8 against an fp8 release, fp4 below it, unknown not declared
    q = f["quantization"]
    assert (q["same as released"]["rows"], q["lower precision"]["rows"],
            q["not declared"]["rows"], q["higher precision"]["rows"]) == (1, 1, 1, 0)


def test_fidelity_edges_and_unestablished_release():
    assert MK._ctx_bucket(524_288, 1_048_576) == "50-99%"
    assert MK._ctx_bucket(262_144, 1_048_576) == "25-49%"
    assert MK._ctx_bucket(None, 1_048_576) == "not published"
    assert MK._out_bucket(32_767) == "under 32k" and MK._out_bucket(131_073) == "over 128k"
    assert MK._quant_bucket("bf16", "fp8") == "higher precision"
    assert MK._quant_bucket("fp8", None) == "released dtype not established"
    with pytest.raises(MK.MarketDataError, match="'fp5'"):
        MK._quant_bucket("fp5", "fp8")
    # every catalogue model has an entry, with its source or why there is none
    assert set(MK.RELEASED_DTYPE) == set(MK.M.MODELS)
    assert all(src for _, src in MK.RELEASED_DTYPE.values())


def test_fidelity_doc_table_is_current():
    """docs/market.md's Provider fidelity table is what
    scripts/market_fidelity_doc.py writes from provider_prices.json."""
    import importlib.util
    import sys
    path = Path(__file__).resolve().parents[1] / "scripts" / "market_fidelity_doc.py"
    spec = importlib.util.spec_from_file_location("_market_fidelity_doc", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.main(["--check"]) == 0, "run uv run scripts/market_fidelity_doc.py"


def test_context_share_uses_the_native_context(snap):
    rows = MK.provider_rows(snap, "GLM52", MK.default_workload(), 300.0)
    assert MK.fidelity_summary("GLM52", rows)["context_label"] == "1,048,576"
    n, label = MK.native_context("Q38FN")
    assert n == 262_144 and label == "262,144 native, 1,048,576 with YaRN"
    assert MK._ctx_bucket(262_144, n) == "100%"
