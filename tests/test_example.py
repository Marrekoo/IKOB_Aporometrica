"""The synthetic example (examples/tiny): build the data folder, run the model
through its command line, and compare with the stored output."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tiny"


def _make_data():
    spec = importlib.util.spec_from_file_location("make_data",
                                                  EXAMPLE / "make_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_example_reproduces_the_stored_output(tmp_path, monkeypatch):
    from ikob2.cli import accessibility

    monkeypatch.delenv("IKOB_DATA_ROOT", raising=False)
    root = tmp_path / "data"
    _make_data().build(root)
    accessibility.main(["--data-root", str(root), "--study", "tiny",
                        "--run", "example", "--modes", "car", "bike", "pt",
                        "--log-level", "ERROR"])
    got = pd.read_csv(root / "outputs/runs/example/accessibility.csv")
    want = pd.read_csv(EXAMPLE / "expected/accessibility.csv")
    assert list(got.columns) == list(want.columns)
    key = ["buurtcode", "mode", "segment"]
    got, want = got.sort_values(key).reset_index(drop=True), \
        want.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[key + ["household_type", "income_class"]],
                                  want[key + ["household_type", "income_class"]])
    for col in ("population", "accessibility", "atom", "accessibility_normalised"):
        np.testing.assert_allclose(got[col], want[col], rtol=1e-5, atol=1e-6,
                                   err_msg=col)
    # the job weights written next to it partition every sector's jobs
    jw = pd.read_csv(root / "outputs/runs/example/job_weights.csv")
    by_sector = jw[jw.income_class != "onbekend"].groupby("sector").weight.sum()
    np.testing.assert_allclose(by_sector, 1.0, atol=1e-9)
    # the run is meaningful: decile 1 reaches only free trips, others more
    by = got.groupby(["mode", "income_class"])["accessibility"].mean()
    assert by[("pt", "D1")] == 0 and by[("pt", "D5")] > 0
    assert by[("car", "D5")] > by[("car", "D1")] > 0


@pytest.mark.parametrize("spec", ["m0u", "m0s", "m1c"])
def test_example_runs_the_calibrated_specifications(tmp_path, monkeypatch, spec):
    """The specifications whose cut-off or cost mean the command line
    calibrates on the population (M0u, M1c) or per segment (M0s)."""
    import json

    from ikob2.cli import accessibility

    monkeypatch.delenv("IKOB_DATA_ROOT", raising=False)
    root = tmp_path / "data"
    _make_data().build(root)
    accessibility.main(["--data-root", str(root), "--study", "tiny",
                        "--run", spec, "--modes", "pt", "--spec", spec,
                        "--log-level", "ERROR"])
    run = root / "outputs/runs" / spec
    got = pd.read_csv(run / "accessibility.csv")
    meta = json.loads((run / "run.json").read_text())
    d1 = got[got.income_class == "D1"]
    if spec == "m0u":
        # one positive cost cut-off for everyone: decile 1 has no atom
        assert meta["scenario"]["m0u"]["cost_cutoff_eur"] > 0
        assert (d1["atom"] == 0).all()
    elif spec == "m0s":
        # decile 1's own cut-off is zero: atom 1, no priced journey passes
        assert meta["scenario"]["m0s"]["cost_cutoff_eur"]["single_D1"] == 0.0
        assert (d1["atom"] == 1).all() and (d1["accessibility"] == 0).all()
    else:
        assert meta["scenario"]["m1c"]["cost_mean_eur"] > 0
        assert (got["atom"] == 0).all()
