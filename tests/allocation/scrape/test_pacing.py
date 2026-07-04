"""Throttle avoidance: request pacing, connection reuse, 429/503 backoff.

Exercises ``_http_get`` with a fake ``requests`` module (monkeypatched
onto the module) — no network, and ``time.sleep`` is captured so the
spacing/backoff decisions are asserted without actually waiting.
"""

from __future__ import annotations

import sys
import time
import types

import pytest

from icepick.allocation.scrape import realmath as source
from icepick.allocation.scrape.checkpoint import ScrapeCheckpoint

_ONE_PAPER_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2604.00001v1</id>
    <title>Paper One</title>
    <summary>Abstract one.</summary>
    <arxiv:primary_category term="math.AP"/>
    <category term="math.AP"/>
  </entry>
</feed>"""

_EMPTY_FEED = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class _Resp:
    def __init__(self, status=200, text="ok", headers=None):
        self.status_code = status
        self.text = text
        self.content = text.encode()
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _FakeRequests.HTTPError(f"{self.status_code} error")


class _FakeRequests(types.SimpleNamespace):
    """Just enough of the requests API for _http_get / _session."""

    class RequestException(Exception):
        pass

    class HTTPError(RequestException):
        pass

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self.calls = []

    def Session(self):  # noqa: N802 — mirrors requests.Session
        owner = self

        class _Session:
            def __init__(self):
                self.headers = {}

            def get(self, url, timeout=None):
                owner.calls.append(url)
                resp = owner._responses.pop(0)
                if isinstance(resp, Exception):
                    raise resp
                return resp

        return _Session()


@pytest.fixture
def _fake_net(monkeypatch):
    def _install(responses):
        fake = _FakeRequests(responses)
        monkeypatch.setitem(sys.modules, "requests", fake)
        # Fresh module state so the session/pacer don't leak across tests.
        monkeypatch.setattr(source, "_http_session", None)
        monkeypatch.setattr(source, "_last_request_at", 0.0)
        sleeps: list = []
        monkeypatch.setattr(source.time, "sleep", lambda s: sleeps.append(s))
        # Deterministic clock so pacing math is exact.
        clock = {"t": 1000.0}
        monkeypatch.setattr(source.time, "monotonic", lambda: clock["t"])
        return fake, sleeps, clock

    return _install


def test_requests_are_spaced_by_the_min_interval(monkeypatch, _fake_net):
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 3.0)
    fake, sleeps, clock = _fake_net([_Resp(), _Resp()])
    # First call: last_request_at=0, clock=1000 → already past interval, no wait.
    source._http_get("u1", timeout=5, retries=2, backoff=1)
    assert sleeps == []
    # Second call 1s later: only 1s elapsed of the 3s interval → wait 2s.
    clock["t"] = 1001.0
    source._http_get("u2", timeout=5, retries=2, backoff=1)
    assert sleeps == [pytest.approx(2.0)]


def test_session_is_reused_across_calls(monkeypatch, _fake_net):
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 0.0)
    fake, sleeps, clock = _fake_net([_Resp(), _Resp()])
    source._http_get("u1", timeout=5, retries=2, backoff=1)
    first_session = source._http_session
    source._http_get("u2", timeout=5, retries=2, backoff=1)
    assert source._http_session is first_session  # one keep-alive connection
    assert fake.calls == ["u1", "u2"]


def test_503_is_retried_with_backoff_like_429(monkeypatch, _fake_net):
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 0.0)
    fake, sleeps, clock = _fake_net([_Resp(status=503), _Resp(status=200, text="recovered")])
    result = source._http_get("u", timeout=5, retries=3, backoff=3)
    assert result.text == "recovered"
    assert sleeps == [pytest.approx(3.0)]  # backoff * 2**0 = 3s on the first (503) attempt


def test_backoff_doubles_across_consecutive_failures(monkeypatch, _fake_net):
    """Two consecutive 429s → 3s then 6s (doubling), then success."""
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 0.0)
    fake, sleeps, clock = _fake_net([
        _Resp(status=429),
        _Resp(status=429),
        _Resp(status=200, text="ok"),
    ])
    result = source._http_get("u", timeout=5, retries=4, backoff=3)
    assert result.text == "ok"
    assert sleeps == [pytest.approx(3.0), pytest.approx(6.0)]  # 3s, 6s — doubling


def test_retry_after_header_is_honored(monkeypatch, _fake_net):
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 0.0)
    fake, sleeps, clock = _fake_net([
        _Resp(status=429, headers={"Retry-After": "12"}),
        _Resp(status=200),
    ])
    source._http_get("u", timeout=5, retries=3, backoff=3)
    assert sleeps == [pytest.approx(12.0)]  # server's Retry-After wins over backoff


def test_final_rate_limit_stamps_checkpoint_marker(monkeypatch, tmp_path, _fake_net):
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 0.0)
    fake, sleeps, clock = _fake_net([_Resp(status=429)])
    checkpoint = ScrapeCheckpoint(tmp_path / "_progress")

    def fetcher(query, *, start, max_results):
        return source._http_get("u", timeout=5, retries=1, backoff=3).text

    with pytest.raises(_FakeRequests.HTTPError):
        source.scrape(
            scrape_window={"category": "math.AP"}, source_name="s", target_count=1,
            fetcher=fetcher, checkpoint=checkpoint,
        )

    assert (tmp_path / "_progress" / "rate_limited_at").exists()
    assert sleeps == []


def test_latex_fetcher_gets_same_rate_limit_marker_treatment(monkeypatch, tmp_path, _fake_net):
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 0.0)
    fake, sleeps, clock = _fake_net([_Resp(status=503), _Resp(status=200, text="tarball")])
    checkpoint = ScrapeCheckpoint(tmp_path / "_progress")

    with source._http_observers(
        on_rate_limit=lambda status, delay: checkpoint.stamp_rate_limited(),
        on_success=checkpoint.clear_rate_limit,
    ):
        assert source.default_latex_source_fetcher("2604.00001", retries=2) == b"tarball"

    assert fake.calls == ["https://arxiv.org/e-print/2604.00001"] * 2
    assert not (tmp_path / "_progress" / "rate_limited_at").exists()
    assert sleeps == [pytest.approx(3.0)]


def test_scrape_reports_rate_limit_telemetry_and_halves_atom_page(monkeypatch, _fake_net):
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 0.0)
    fake, sleeps, clock = _fake_net([
        _Resp(status=429),
        _Resp(status=200, text=_ONE_PAPER_FEED),
        _Resp(status=200, text=_EMPTY_FEED),
    ])

    result = source.scrape(
        scrape_window={"category": "math.AP"}, source_name="s", target_count=2,
    )

    assert result.rate_limit_events == 1
    assert result.rate_limit_backoff_seconds == pytest.approx(3.0)
    assert result.rate_limit_statuses == {"429": 1}
    assert "max_results=50" in fake.calls[0]
    assert "max_results=50" in fake.calls[1]  # retry of the same page
    assert "start=50" in fake.calls[2]
    assert "max_results=25" in fake.calls[2]
    assert sleeps == [pytest.approx(3.0)]


def test_min_interval_zero_disables_pacing(monkeypatch, _fake_net):
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", 0.0)
    fake, sleeps, clock = _fake_net([_Resp(), _Resp(), _Resp()])
    for u in ("a", "b", "c"):
        source._http_get(u, timeout=5, retries=1, backoff=1)
    assert sleeps == []  # tests never sleep for pacing


def test_pacer_inserts_real_delays_across_both_fetchers(monkeypatch):
    """Real wall-clock proof: consecutive requests — the Atom query AND the
    e-print fetch, which share the pacer — are spaced at least the interval
    apart. Asserts a lower bound only (sleep never returns early), so it is
    robust to a loaded machine. Uses a small interval to stay fast.
    """
    interval = 0.05
    monkeypatch.setattr(source, "_MIN_REQUEST_INTERVAL", interval)
    monkeypatch.setattr(source, "_http_session", None)
    monkeypatch.setattr(source, "_last_request_at", 0.0)

    request_times: list = []

    class _RealSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout=None):
            request_times.append(time.monotonic())
            return _Resp()

    fake = types.SimpleNamespace(
        Session=lambda: _RealSession(),
        RequestException=_FakeRequests.RequestException,
        HTTPError=_FakeRequests.HTTPError,
    )
    monkeypatch.setitem(sys.modules, "requests", fake)

    start = time.monotonic()
    # Interleave the two real fetchers — proof the pacer is shared, not per-fetcher.
    source.default_arxiv_fetcher("q", start=0, max_results=1, retries=1)
    source.default_latex_source_fetcher("2604.00001", retries=1)
    source.default_arxiv_fetcher("q", start=1, max_results=1, retries=1)
    elapsed = time.monotonic() - start

    assert len(request_times) == 3
    gaps = [b - a for a, b in zip(request_times, request_times[1:])]
    assert all(gap >= interval - 0.005 for gap in gaps), gaps  # real spacing happened
    assert elapsed >= 2 * interval - 0.01  # two gaps' worth of real time elapsed
    # The first request is never delayed (nothing precedes it).
    assert request_times[0] - start < interval
