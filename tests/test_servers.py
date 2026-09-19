"""Local Valhalla and OTP helpers (no servers are started)."""

import io
import json

import numpy as np
import pytest

from ikob2.skims import otp_server, valhalla_server
from ikob2.utils.paths import DataLayout


@pytest.fixture
def layout(tmp_path):
    lay = DataLayout(tmp_path / "data")
    lay.ensure()
    return lay


def test_layout_has_server_folders(layout):
    assert layout.valhalla_dir().is_dir() and layout.otp_dir().is_dir()


def test_valhalla_config_is_local_and_has_raised_limits(layout):
    path = valhalla_server.write_config(layout, concurrency=3, port=8123)
    cfg = json.loads(path.read_text())
    assert cfg["httpd"]["service"]["listen"] == "tcp://*:8123"
    assert cfg["mjolnir"]["tile_dir"] == str(layout.valhalla_dir() / "tiles")
    assert cfg["mjolnir"]["concurrency"] == 3
    for costing in ("auto", "bicycle", "pedestrian"):
        lim = cfg["service_limits"][costing]
        assert lim["max_matrix_location_pairs"] >= 10_000
        assert 50_000 <= lim["max_matrix_distance"] <= 200_000    # memory guard
    # the IPC sockets sit in the data folder, not in /tmp
    assert str(layout.valhalla_dir()) in cfg["httpd"]["service"]["loopback"]
    assert valhalla_server._port(layout) == 8123


def test_valhalla_start_needs_a_config_and_status_is_false_when_down(layout):
    assert valhalla_server.status(port=59999) is False
    valhalla_server.write_config(layout, port=59998)
    assert valhalla_server.stop(layout) is False        # nothing running


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_valhalla_matrix_client_parses_distances_and_times(monkeypatch):
    seen = {}

    def fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        rows = [[{"from_index": 0, "to_index": 0, "distance": 12.5, "time": 900},
                 {"from_index": 0, "to_index": 1, "distance": None, "time": None}],
                [{"from_index": 1, "to_index": 0, "distance": 3.0, "time": 240},
                 {"from_index": 1, "to_index": 1, "distance": 0.0, "time": 0}]]
        return _Resp(json.dumps({"sources_to_targets": rows}).encode())

    monkeypatch.setattr(valhalla_server.urllib.request, "urlopen", fake)
    dist, tim = valhalla_server.matrix(
        np.array([[5.1, 52.0], [5.2, 52.1]]), np.array([[5.3, 52.2], [5.4, 52.3]]),
        mode="bike", url="http://h:1")
    assert seen["url"] == "http://h:1/sources_to_targets"
    assert seen["body"]["costing"] == "bicycle"
    assert seen["body"]["units"] == "kilometers"
    assert seen["body"]["sources"][0] == {"lon": 5.1, "lat": 52.0}
    assert dist[0, 0] == 12.5 and tim[0, 0] == pytest.approx(15.0)
    assert np.isnan(dist[0, 1]) and dist[1, 1] == 0.0


def test_otp_prepare_links_inputs_and_writes_configs(layout, tmp_path):
    pbf = tmp_path / "nl.osm.pbf"
    zip_ = tmp_path / "gtfs.zip"
    pbf.write_text("p")
    zip_.write_text("z")
    graph = otp_server.prepare(layout, pbf, zip_, service_start="2026-09-01",
                               service_end="2026-09-30")
    assert (graph / "nl.osm.pbf").is_symlink()
    assert (graph / "gtfs.zip").read_text() == "z"
    bc = json.loads((graph / "build-config.json").read_text())
    assert bc["transitServiceStart"] == "2026-09-01"
    assert bc["transitServiceEnd"] == "2026-09-30"
    # preparing again replaces the links without error
    otp_server.prepare(layout, pbf, zip_, service_start="2026-09-01",
                       service_end="2026-09-30")


def test_otp_build_and_start_need_their_files(layout):
    with pytest.raises(FileNotFoundError, match="missing"):
        otp_server.build(layout)
    (otp_server.jar_path(layout)).write_text("jar")
    with pytest.raises(FileNotFoundError, match="graph.obj"):
        otp_server.start(layout, port=59997)
    assert otp_server.status(port=59997) is False
    assert otp_server.stop(layout) is False
