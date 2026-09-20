import pandas as pd
import pytest

from ikob2.segments.ownership import load_bike_ownership


def write(tmp_path, rows):
    p = tmp_path / "own.csv"
    pd.DataFrame(rows, columns=["buurtcode", "buurtnaam", "pct_with_bicycle"]
                 ).to_csv(p, index=False)
    return p


def test_fractions_and_reindex(tmp_path):
    p = write(tmp_path, [("BU1", "a", 90), ("BU2", "b", 72.5)])
    s = load_bike_ownership(p, ["BU2", "BU1"])
    assert list(s.index) == ["BU2", "BU1"]
    assert s.tolist() == pytest.approx([0.725, 0.90])


def test_blank_is_error_unless_filled(tmp_path):
    p = write(tmp_path, [("BU1", "a", 90), ("BU2", "b", None)])
    with pytest.raises(ValueError, match="BU2"):
        load_bike_ownership(p)
    s = load_bike_ownership(p, ["BU1", "BU2", "BU3"], fill=0.89)
    assert s.tolist() == pytest.approx([0.90, 0.89, 0.89])


def test_range_duplicates_and_columns(tmp_path):
    with pytest.raises(ValueError, match="0-100"):
        load_bike_ownership(write(tmp_path, [("BU1", "a", 0.9), ("BU2", "b", 190)]))
    with pytest.raises(ValueError, match="duplicate"):
        load_bike_ownership(write(tmp_path, [("BU1", "a", 90), ("BU1", "b", 80)]))
    p = tmp_path / "bad.csv"
    pd.DataFrame({"buurtcode": ["BU1"]}).to_csv(p, index=False)
    with pytest.raises(KeyError):
        load_bike_ownership(p)


def test_shipped_utrecht_table_is_complete():
    from pathlib import Path
    p = Path("/home/marco/IKOB data/inputs/veh_owners/bike_ownership_buurten.csv")
    if not p.exists():
        pytest.skip("data folder not available")
    s = load_bike_ownership(p)
    assert len(s) == 111 and s.between(0.7, 1.0).all()
