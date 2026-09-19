"""Rough home-working capability per LISA sector."""

import numpy as np
import pandas as pd
import pytest

from ikob2.segments.lisa import SECTOR_TO_SBI, SECTORS
from ikob2.segments.wfh import (
    lisa_sector_wfh_share,
    sbi_wfh_share,
    sector_education_mix,
    split_jobs_by_wfh,
    wfh_incidence_by_education,
)


def _hw():
    rows = []
    for key, tot, anyw in (("2018700", 1000, 200), ("2018740", 3000, 1200),
                           ("2018790", 2000, 1600), ("T009002", 6000, 3000)):
        rows.append((key, "T001205", tot))
        rows.append((key, "A027929", anyw))
        rows.append((key, "A027934", tot - anyw))
    return pd.DataFrame(rows, columns=["Persoonskenmerken", "Thuiswerken",
                                       "WerkzameBeroepsbevolking_1"])


def test_incidence_by_education():
    inc = wfh_incidence_by_education(_hw())
    assert inc.to_dict() == {"low": 0.2, "middle": 0.4, "high": 0.8}
    bad = _hw()
    bad.loc[bad.Thuiswerken == "A027929", "WerkzameBeroepsbevolking_1"] = 0
    with pytest.raises(ValueError, match="Implausible"):
        wfh_incidence_by_education(bad)
    with pytest.raises(KeyError, match="lacks education"):
        wfh_incidence_by_education(_hw()[lambda d: d.Persoonskenmerken
                                         != "2018790"])


def _mix_raw():
    rows = []
    for sbi, low, mid, high in (("A", 50, 40, 10), ("B", 10, 30, 60)):
        for edu, n in (("18700", low), ("18740", mid), ("18790", high),
                       ("10000", 100)):
            rows.append((edu, sbi, n))
    return pd.DataFrame(rows, columns=["Onderwijsniveau",
                                       "BedrijfstakkenSBI2008",
                                       "BanenVanWerknemers_1"])


def test_sector_education_mix_and_sbi_share():
    mix = sector_education_mix(_mix_raw())
    np.testing.assert_allclose(mix.loc["A"], [0.5, 0.4, 0.1])
    np.testing.assert_allclose(mix.sum(axis=1), 1.0)
    inc = pd.Series({"low": 0.2, "middle": 0.4, "high": 0.8})
    share = sbi_wfh_share(inc, mix)
    assert share["A"] == pytest.approx(0.5 * 0.2 + 0.4 * 0.4 + 0.1 * 0.8)
    assert share["B"] > share["A"]                     # more educated -> more WFH


def test_lisa_sector_share_is_job_weighted_over_sections():
    keys = sorted({k for v in SECTOR_TO_SBI.values() for k in v})
    share = pd.Series(np.linspace(0.2, 0.8, len(keys)), index=keys)
    jobs = pd.Series(1.0, index=keys)
    jobs["403300"], jobs["410200"] = 100.0, 300.0
    out = lisa_sector_wfh_share(share, jobs)
    assert list(out.index) == list(SECTORS)
    assert out["L10"] == pytest.approx(
        (share["403300"] * 100 + share["410200"] * 300) / 400)
    assert out["L01"] == share["301000"]
    with pytest.raises(KeyError, match="Missing SBI"):
        lisa_sector_wfh_share(share.drop("389100"), jobs)


def test_split_adds_up_and_validates():
    jobs = pd.DataFrame({"L01": [10.0, 20.0], "L02": [5.0, 0.0]},
                        index=["b1", "b2"])
    no, yes = split_jobs_by_wfh(jobs, pd.Series({"L01": 0.25, "L02": 1.0}))
    pd.testing.assert_frame_equal(no + yes, jobs)
    assert yes.loc["b2", "L01"] == 5.0 and no.loc["b1", "L02"] == 0.0
    with pytest.raises(ValueError, match="cover every sector"):
        split_jobs_by_wfh(jobs, pd.Series({"L01": 0.25}))
    with pytest.raises(ValueError, match="cover every sector"):
        split_jobs_by_wfh(jobs, pd.Series({"L01": 1.5, "L02": 0.1}))
