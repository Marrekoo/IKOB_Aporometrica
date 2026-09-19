"""OSRM table client and detour calibration (no network)."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from ikob2.skims import osrm


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(recorder):
    def _open(url, timeout=None):
        recorder.append(url)
        # decode the requested sources / destinations
        path = url.split("/driving/")[1].split("?")[0]
        n = len(path.split(";"))
        q = url.split("?")[1]
        srcs = [int(x) for x in q.split("sources=")[1].split("&")[0].split(";")]
        dsts = [int(x) for x in q.split("destinations=")[1].split("&")[0].split(";")]
        assert max(srcs + dsts) < n
        dist = [[1000.0 * (1 + s + 10 * t) for t in range(len(dsts))]
                for s in range(len(srcs))]
        dist[0][0] = None                       # an unreachable pair
        dur = [[60.0 * (1 + s + 10 * t) for t in range(len(dsts))]
               for s in range(len(srcs))]
        dur[0][0] = None
        return _Resp(json.dumps({"code": "Ok", "distances": dist,
                                 "durations": dur}).encode())
    return _open


def test_table_parses_units_and_limits(monkeypatch):
    calls = []
    monkeypatch.setattr(osrm.urllib.request, "urlopen", fake_urlopen(calls))
    o = np.array([[5.1, 52.0], [5.2, 52.1]])
    d = np.array([[5.3, 52.2], [5.4, 52.3], [5.5, 52.4]])
    dist, dur = osrm.osrm_table("http://x", o, d, pause=0)
    assert dist.shape == (2, 3) and dur.shape == (2, 3)
    assert np.isnan(dist[0, 0]) and np.isnan(dur[0, 0])
    assert dist[1, 2] == pytest.approx(1 + 1 + 20)      # km
    assert dur[1, 2] == pytest.approx(22.0)             # minutes
    assert len(calls) == 1 and "annotations=distance,duration" in calls[0]
    with pytest.raises(ValueError, match="limit"):
        osrm.osrm_table("http://x", np.zeros((60, 2)), np.zeros((60, 2)))


def test_error_response_is_retried_then_raised(monkeypatch):
    def bad(url, timeout=None):
        return _Resp(json.dumps({"code": "TooBig"}).encode())
    monkeypatch.setattr(osrm.urllib.request, "urlopen", bad)
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        osrm.osrm_table("http://x", np.zeros((1, 2)), np.zeros((1, 2)),
                        retries=2, pause=0)


def test_routed_pairs_batches_and_drops_unreachable(monkeypatch):
    calls = []
    monkeypatch.setattr(osrm.urllib.request, "urlopen", fake_urlopen(calls))
    o = pd.DataFrame({"id": [f"o{i}" for i in range(5)],
                      "lon": 5.0, "lat": 52.0})
    d = pd.DataFrame({"id": [f"d{j}" for j in range(7)],
                      "lon": 5.5, "lat": 52.5})
    res = osrm.routed_pairs("http://x", o, d, origin_batch=2, dest_batch=3,
                            pause=0)
    assert len(calls) == 3 * 3                           # 3 origin x 3 dest batches
    # every batch's [0,0] is unreachable
    assert len(res) == 5 * 7 - 9
    assert set(res.columns) == {"from_id", "to_id", "route_km", "minutes"}


def test_sample_pairs_covers_near_and_far():
    rng = np.random.default_rng(0)
    pts = pd.DataFrame({"id": [f"z{i}" for i in range(2000)],
                        "x": rng.uniform(0, 250_000, 2000),
                        "y": rng.uniform(0, 250_000, 2000),
                        "lon": 5.0, "lat": 52.0})
    pts.loc[:99, ["x", "y"]] = rng.uniform(120_000, 130_000, (100, 2))
    o, d = osrm.sample_pairs(pts, pts.id[:100], n_origins=10, n_far=40,
                             n_near=30, near_km=15, seed=1)
    assert len(o) == 10 and not set(o.id) & set(d.id)
    dist = np.hypot(d.x - o.x.mean(), d.y - o.y.mean()) / 1000
    assert (dist < 15).sum() >= 20 and (dist > 60).sum() >= 10


def test_calibrate_from_routes():
    rng = np.random.default_rng(0)
    pts = pd.DataFrame({"id": [f"p{i}" for i in range(200)],
                        "x": rng.uniform(0, 100_000, 200),
                        "y": rng.uniform(0, 100_000, 200)})
    a, b = rng.integers(0, 200, 3000), rng.integers(0, 200, 3000)
    keep = a != b
    routes = pd.DataFrame({"from_id": pts.id[a[keep]].to_numpy(),
                           "to_id": pts.id[b[keep]].to_numpy()})
    crow = np.hypot(pts.x[a[keep]].to_numpy() - pts.x[b[keep]].to_numpy(),
                    pts.y[a[keep]].to_numpy() - pts.y[b[keep]].to_numpy()) / 1000
    routes["route_km"] = crow * 1.25
    m = osrm.calibrate_from_routes(pts, routes)
    assert m.route_km([30.0])[0] / 30 == pytest.approx(1.25, rel=1e-6)
