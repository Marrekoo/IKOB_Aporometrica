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
    # the run is meaningful: decile 1 reaches only free trips, others more
    by = got.groupby(["mode", "income_class"])["accessibility"].mean()
    assert by[("pt", "D1")] == 0 and by[("pt", "D5")] > 0
    assert by[("car", "D5")] > by[("car", "D1")] > 0
