# Validation scripts

Analyses behind the validation numbers in the documentation. They are not
part of the package and not run by the tests: they need the full data folder
and, for the PT comparison, a running OpenTripPlanner server.

| Script | Documented in | Status |
|---|---|---|
| `pt_router_vs_otp.py` | `docs/servers.md`, validation of the frequency-model PT router | the samples in `results/` reproduce every documented figure (`summary`) |
| `jobs_buurt_totals.py` | `docs/data_lineage.md`, validation of the buurt totals | reproduces the documented table |

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
`<root>/cache/statline/` on the first run and prints the job size per SBI
group and the table of `docs/data_lineage.md` (`--out` writes it as CSV).
`--establishments FILE` replaces the KWB 2016 establishments by another
snapshot, to test the script without the download.
