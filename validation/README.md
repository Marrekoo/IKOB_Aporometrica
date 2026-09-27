# Validation scripts

Analyses behind the validation numbers in the documentation. They are not
part of the package and not run by the tests: they need the full data folder
and, for the PT comparison, a running OpenTripPlanner server.

| Script | Documented in | Status |
|---|---|---|
| `pt_router_vs_otp.py` | `docs/servers.md`, validation of the frequency-model PT router | the samples in `results/` reproduce every documented figure (`summary`) |
| `jobs_buurt_totals.py` | `docs/data_lineage.md`, validation of the buurt totals | reconstructed from the documented method; not yet rerun with KWB 2016 (see below) |

## PT router against OpenTripPlanner

    python -m ikob2.cli.servers otp start --heap 8G        # national graph (docs/servers.md)
    python validation/pt_router_vs_otp.py sample --data-root <root> \
        --out validation/results/otp_vs_freq_national.csv
    python validation/pt_router_vs_otp.py summary validation/results/otp_vs_freq_national.csv

The regional sample used the regional graph and
`--origins 14 --per-origin 10 --bbox 3.6 5.85 51.75 52.95`. `results/` holds
both samples (pairs of buurt codes with the model's and OTP's minutes and
kilometres). A new sample differs from them only where the OTP graph, the
GTFS feed or the skim store differ.

## Buurt job totals against LISA 2016

    python validation/jobs_buurt_totals.py --data-root <root>

Downloads the KWB 2016 establishments (CBS 83487NED) into
`<root>/cache/statline/` on the first run. The original analysis was a
one-off; this script follows the method described in `docs/data_lineage.md`.
It has only been run with the 2022 establishments as a stand-in
(`--establishments`), because CBS StatLine was unreachable when it was
written. In that run the two rows that do not depend on the establishments
came out at 0.452 (uniform) and 0.299 (IKOB job table) on 10,123 buurten in
306 municipalities, against the documented 0.454 and 0.301 on 10,137 buurten
in 304 municipalities. Check, when rerunning with KWB 2016:

* whether the buurt set then matches (10,137 buurten, 304 municipalities);
* the job size per SBI group: the documentation gives about 65 jobs per
  establishment for O-Q, while national LISA jobs per KWB establishment give
  about 8 with the 2022 counts. The definition used originally may have been
  different.

Update the table in `docs/data_lineage.md` with the result.
