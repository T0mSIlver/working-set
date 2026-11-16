import itertools

import httpx
import pytest

from workingset.bench import default_metrics_url
from workingset.metrics import identity as I


def _serve(monkeypatch, starts):
    """httpx.get answers each scrape with the next start time (None: absent)."""
    it = itertools.cycle(starts)

    def get(url, **kw):
        t = next(it)
        body = "vllm:num_requests_running 0\n" + (
            "" if t is None else f"process_start_time_seconds {t}\n")
        return httpx.Response(200, text=body, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", get)


def test_two_pods_behind_one_url_stop_the_run(monkeypatch):
    _serve(monkeypatch, [1.79456091605e9, 1.79456337253e9])
    with pytest.raises(ValueError, match="2 different servers"):
        I.check_one_server("http://x/metrics", {})


def test_one_server_passes_and_a_missing_start_time_is_a_note(monkeypatch):
    _serve(monkeypatch, [1.7e9])
    assert I.check_one_server("http://x/metrics", {}) is None
    _serve(monkeypatch, [None])
    assert "cannot tell" in I.check_one_server("http://x/metrics", {})


@pytest.mark.parametrize("base, want", [
    ("http://h:8000/v1", "http://h:8000/metrics"),
    ("https://gw/ns/llm-large-v2/v1/", "https://gw/ns/llm-large-v2/metrics"),
    ("http://h:8000", "http://h:8000/metrics"),
])
def test_default_metrics_url_keeps_the_route_prefix(base, want):
    assert default_metrics_url(base) == want
