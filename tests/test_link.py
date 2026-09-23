"""`ws link` against the explorer it links to.

The fixtures tests/fixtures/explorer_*.toml each carry, in their header, the
share URL of the page state they were downloaded at. `ws link` on the file must
give that URL back: the same keys, the same values, nothing the page would
leave out. The key set and the control ranges are read from the explorer's own
sources, so a renamed or re-ranged control fails here rather than in a browser.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qsl

import pytest

from workingset import cli
from workingset.config import RunConfig, load_config
from workingset.link import DEFAULT_BASE, KNOBS, explorer_link

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = sorted((ROOT / "tests" / "fixtures").glob("explorer_*.toml"))
MAIN_JS = (ROOT / "interactive" / "src" / "main.js").read_text(encoding="utf-8")
INDEX = (ROOT / "interactive" / "index.html").read_text(encoding="utf-8")


def _frag(url: str) -> dict[str, str]:
    return dict(parse_qsl(url.partition("#")[2], keep_blank_values=True))


def _decoded_keys() -> set[str]:
    """Every key main.js applyURLState() reads."""
    def block(name: str) -> str:
        m = re.search(rf"const {name}\s*=\s*([\[{{][\s\S]*?[\]}}]);", MAIN_JS)
        assert m, f"{name} not found in main.js"
        return m.group(1)
    enums = set(re.findall(r"^\s*(\w+):", block("URL_ENUMS"), re.M))
    enums |= set(re.findall(r"(\w+): \(\) =>", block("URL_ENUMS")))
    sliders = set(re.findall(r"\['s-\w+','(\w+)'", block("sliderMap")))
    extra = set(re.findall(r"(\w+): \[", block("URL_EXTRA_NUM")))
    bools = set(re.findall(r'"(\w+)"', block("URL_BOOLS")))
    assert {"model", "chunk"} <= enums and "users" in sliders and "tp" in extra
    return enums | sliders | extra | bools


def _slider(key: str) -> tuple[float, float]:
    m = re.search(rf'<input type="range" id="s-{key}" min="([\d.]+)" max="([\d.]+)"', INDEX)
    assert m, f"no slider s-{key} in index.html"
    return float(m.group(1)), float(m.group(2))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_link_reproduces_the_fixture_url(path):
    header = re.search(r"^# reproduce this page: (\S+)", path.read_text(), re.M).group(1)
    url, warnings = explorer_link(load_config(path))
    assert url.startswith(DEFAULT_BASE)
    assert _frag(url) == _frag(header)
    assert warnings == []


def test_every_knob_is_a_key_the_explorer_decodes():
    assert {k.key for k in KNOBS} <= _decoded_keys()


def test_knob_ranges_match_the_controls():
    rescaled = {"user_median": 120, "sub_median": 60}   # main.js ctxScale bases
    cfg = RunConfig()
    for k in KNOBS:
        if k.lo is None or not re.search(rf'id="s-{k.key}"', INDEX):
            continue
        lo, hi = _slider(k.key)
        assert k.lo == lo, k.key
        if k.key in rescaled:
            assert hi == rescaled[k.key]
        elif k.key == "cap":
            assert k.hi(cfg) == 1049          # the 27B's 1M context top stop
        else:
            assert k.hi == hi, k.key


def test_fixture_set_covers_every_knob():
    seen = set()
    for p in FIXTURES:
        seen |= set(_frag(explorer_link(load_config(p))[0]))
    # headcount/active/spu ride the population fixture; every other knob the
    # every-knob one
    assert {k.key for k in KNOBS} == seen


def test_default_config_is_the_bare_page():
    assert explorer_link(RunConfig()) == (DEFAULT_BASE, [])


def _warned(cfg) -> str:
    return "\n".join(explorer_link(cfg)[1])


def test_unmapped_fields_warn():
    c = RunConfig()
    c = replace(c, workload=replace(c.workload, subagent_prefix_tokens=5000),
                slo=replace(c.slo, percentile=99),
                deployment=replace(c.deployment, max_num_seqs=32,
                                   max_num_batched_tokens=3000),
                calibration=replace(c.calibration, decode_pricing="latency"))
    w = _warned(c)
    for field in ("workload.subagent_prefix_tokens", "slo.percentile",
                  "deployment.max_num_seqs", "calibration.decode_pricing",
                  "deployment.max_num_batched_tokens"):
        assert field in w


def test_clamped_values_warn_with_what_the_page_shows():
    c = RunConfig()
    c = replace(c, workload=replace(c.workload, think_time_s=7.5, users=2000.0),
                deployment=replace(c.deployment, model="35BA3B", weight_overhead=0.1))
    url, warnings = explorer_link(c)
    w = "\n".join(warnings)
    assert "workload.think_time_s" in w and "will show 7 s" in w
    assert "workload.users x replicas" in w and "will show 1024" in w
    assert "deployment.weight_overhead" in w and "the link sets 0.15" in w
    assert _frag(url)["think"] == "7" and _frag(url)["users"] == "1024"


def test_headcount_off_the_users_grid_warns():
    c = RunConfig()
    c = replace(c, workload=replace(c.workload, users=None, headcount=250,
                                    peak_active_share=0.5))
    assert "125 sessions" in _warned(c) and "prices 124" in _warned(c)


def test_cli_prints_the_url_and_warns_on_stderr(tmp_path, capsys):
    c = RunConfig()
    p = tmp_path / "w.toml"
    p.write_text(replace(c, slo=replace(c.slo, percentile=99)).dumps())
    assert cli.main(["link", str(p), "--base", "http://127.0.0.1:8123/"]) == 0
    out, err = capsys.readouterr()
    assert out.strip() == "http://127.0.0.1:8123/"
    assert "slo.percentile" in err
