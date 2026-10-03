"""
Validation of the buurt job totals of the jobs imputation (docs/data_lineage.md,
"Validation of the buurt totals").

    python validation/jobs_buurt_totals.py --data-root <root>

Truth: LISA 2016 jobs per buurt (the education file, 2016 buurt codes). For
each predictor of the buurt job share within a municipality it reports the
job-weighted total-variation distance to the truth
(`jobs_impute.within_municipality_tv`; 0 = identical shares):

  uniform                       every buurt of a municipality the same share
  IKOB job table                the buurt totals of the IKOB table (2018 values)
  establishments, total         KWB 2016 establishments per buurt
  establishments x job size     each SBI group weighted by national LISA 2016
                                jobs per KWB establishment of that group
  blends                        (1 - w) IKOB table + w establishments, on
                                shares within the municipality, w = 0.25,
                                0.5, 0.75 (the imputation uses 0.25)

Buurten enter when they have a code in all three sources (codes that did not
change between 2016 and 2022). KWB 2016 (83487NED) does not publish the O-Q
group; it is the total minus the other groups. Suppressed group cells count
as zero. The municipality is the GM code inside the buurt code.

Inputs (data folder): inputs/ikob/ (IKOB job table, education file),
inputs/lisa/ (LISA municipal file) and cache/statline/
kwb_establishments_83487NED.csv, which is downloaded from CBS StatLine on the
first run (network).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ikob2.segments import statline
from ikob2.segments.jobs import parse_ikob_jobs
from ikob2.segments.jobs_impute import (parse_education_shares,
                                        within_municipality_tv)
from ikob2.segments.lisa import SECTOR_TO_KWB_GROUP, parse_lisa_sectors
from ikob2.utils.paths import DataLayout

KWB_2016 = "83487NED"
GROUPS = ("A", "B-F", "G+I", "H+J", "K-L", "M-N", "O-Q", "R-U")


def establishments_2016(lay: DataLayout, path: Path | None = None) -> pd.DataFrame:
    path = path or statline.snapshot_path(lay.statline(),
                                          statline.KWB_ESTABLISHMENTS_SNAPSHOT,
                                          KWB_2016, "")
    if not path.exists():
        print(f"Downloading {KWB_2016} (KWB 2016 establishments) ...")
        statline.fetch_kwb_establishments(KWB_2016).to_csv(path, index=False)
    est = pd.read_csv(path, dtype={"buurtcode": str})
    est["buurtcode"] = est["buurtcode"].str.strip()
    est = est.set_index("buurtcode")
    for g in GROUPS:
        if g not in est.columns:
            est[g] = np.nan
    others = [g for g in GROUPS if g != "O-Q"]
    if est["O-Q"].isna().all():
        est["O-Q"] = (est["total"] - est[others].fillna(0).sum(axis=1)).clip(lower=0)
    return est


def within_shares(values: pd.Series, gemeente: pd.Series) -> pd.Series:
    """Shares within each municipality."""
    total = values.groupby(gemeente).transform("sum")
    return values / total.where(total > 0)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("IKOB_DATA_ROOT"))
    p.add_argument("--establishments", default=None, type=Path,
                   help="establishment snapshot to use instead of KWB 2016 "
                        "(83487NED); only for testing the script")
    p.add_argument("--out", default=None, help="CSV with the table")
    args = p.parse_args(argv)
    if not args.data_root:
        raise SystemExit("Give --data-root or set IKOB_DATA_ROOT.")
    lay = DataLayout(Path(args.data_root))

    truth = parse_education_shares(pd.read_excel(lay.education_jobs()))["edu_jobs"]
    ikob = parse_ikob_jobs(pd.read_excel(
        lay.ikob_jobs(), sheet_name="buurten-arbeidsplaatsen", header=2),
        "2018").sum(axis=1)
    est = establishments_2016(lay, args.establishments)

    common = truth.index.intersection(ikob.index).intersection(est.index)
    gem = pd.Series(["GM" + c[2:6] for c in common], index=common)
    truth, ikob, est = truth[common], ikob[common], est.loc[common]

    # national LISA 2016 jobs per KWB 2016 establishment, by SBI group
    lisa16 = parse_lisa_sectors(pd.read_excel(
        lay.lisa(), sheet_name="LISA Gemeenten per sector"), 2016).sum()
    group_jobs = lisa16.groupby(pd.Series(SECTOR_TO_KWB_GROUP)).sum()
    size = group_jobs / est[list(GROUPS)].sum().reindex(group_jobs.index)

    groups = est[list(GROUPS)].fillna(0.0)
    predictors = {
        "uniform": pd.Series(1.0, index=common),
        "IKOB job table (2018)": ikob,
        "KWB 2016 establishments, total": est["total"].fillna(0.0),
        "establishments x job size per group":
            (groups * size.reindex(list(GROUPS)).fillna(0.0)).sum(axis=1),
    }
    s_ikob = within_shares(ikob, gem)
    s_est = within_shares(est["total"].fillna(0.0), gem)
    for w in (0.25, 0.5, 0.75):
        predictors[f"{1 - w:.0%} IKOB table + {w:.0%} establishments"] = \
            (1 - w) * s_ikob + w * s_est

    rows = [{"predictor": name,
             "distance": within_municipality_tv(pred, truth, gem)}
            for name, pred in predictors.items()]
    table = pd.DataFrame(rows)
    n_gem = gem.groupby(gem).size()
    print(f"{len(common)} buurten in {int((n_gem >= 3).sum())} municipalities "
          f"(with >= 3 common buurten); job size per group:")
    print(size.round(1).to_string())
    print()
    print(table.round(3).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
