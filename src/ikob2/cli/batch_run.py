#!/usr/bin/env python3
"""
Batch driver for ikob2: all IKOBs x all scenarios through run.py.

Each (ikob, scenario) pair runs as a SUBPROCESS, not in-process, for
two reasons: numpy memory is reliably returned to the OS between
runs (the pinned matrices of a large IKOB would otherwise accumulate
across 104 runs), and one crashing run cannot take the sweep down
with it.

Output collision fix: run.py namespaces output by IKOB only, so this
driver passes --out <root>/<scenario> per run, giving
    <root>/<scenario>/<ikob>/accessibility_*.csv
Per-run stdout+stderr goes to <root>/logs/<ikob>_<scenario>.log.

The sweep is RESUMABLE: a run whose four output CSVs already exist
is skipped unless --force is given. Failures are collected and
reported at the end; the driver exits non-zero if any run failed.

Nesting test policy (--nesting): 'first' (default) arms
--nesting-test on the first EXECUTED run only — one certification of
pool mechanics and cache/registry agreement per sweep, at the cost
of one extra segmented run. 'all' arms it everywhere, 'none' never.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Reuse run.py's repo anchor so both scripts agree on where
# examples/test lives, regardless of where THIS file sits.
from ikob2.cli.run import _EXAMPLES_TEST

IKOBS = [f"IKOB{i:02d}" for i in range(1, 27)]
SCENARIOS = ["2018", "2030H", "2040L", "2040H"]

# A run counts as done only if ALL four outputs exist: a run that
# died between writer submissions must not be skipped on resume.
OUTPUT_FILES = (
    "accessibility_hansen.csv",
    "accessibility_segmented_hansen.csv",
    "accessibility_shen.csv",
    "accessibility_segmented_shen.csv",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser("ikob2 batch sweep")

    p.add_argument("--ikobs", nargs="+", default=IKOBS,
                   help="Subset of IKOBs (default: all 26)")
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                   help="Subset of scenarios (default: all 4)")

    # Data roots — defaults are the current project layout.
    p.add_argument("--segs-base", type=Path, default=Path(
        "/home/marco/PycharmProjects/ikob-scripts/segs/"
        "Databronnen SEGS compleet"))
    p.add_argument("--skim-root", type=Path, default=Path(
        "/home/marco/PycharmProjects/ikob-scripts/skims/"
        "Databronnen skims/NRMdata"))
    p.add_argument("--omnummer-root", type=Path, default=Path(
        "/home/marco/PycharmProjects/ikob-scripts/skims/"
        "Databronnen skims/Omnummertabellen"))
    p.add_argument("--filter-config", type=Path,
                   default=_EXAMPLES_TEST / "test_cost_filter.json")

    # Scenario assumptions — passed through verbatim to every run.
    p.add_argument("--fare-fee", type=float, default=2.0)
    p.add_argument("--fare-rate-km", type=float, default=0.12)
    p.add_argument("--fare-detour", type=float, default=1.3)
    p.add_argument("--beta-time", type=float, default=1.0)
    p.add_argument("--beta-distance", type=float, default=0.0)

    p.add_argument("--out-root", type=Path,
                   default=_EXAMPLES_TEST / "output",
                   help="Sweep output root; runs write to "
                        "OUT_ROOT/<scenario>/<ikob>/")
    p.add_argument("--cache-root", type=Path,
                   default=_EXAMPLES_TEST / "cache")

    p.add_argument("--nesting", choices=["first", "all", "none"],
                   default="first",
                   help="When to arm --nesting-test (default: on the "
                        "first executed run only)")
    p.add_argument("--force", action="store_true",
                   help="Re-run pairs whose outputs already exist")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort the sweep on the first failure "
                        "(default: continue and report at the end)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned runs and exit")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args(argv)


def is_done(out_dir: Path) -> bool:
    return all((out_dir / f).exists() for f in OUTPUT_FILES)


def build_cmd(args, ikob, scenario, *, nesting: bool):
    """argv for one run.py invocation, via `python -m` so the child
    resolves ikob2 through the same interpreter/venv as this driver."""
    cmd = [
        sys.executable, "-m", "ikob2.cli.run",
        "--ikob", ikob,
        "--scenario", scenario,
        "--segs-base", str(args.segs_base),
        "--skim-root", str(args.skim_root),
        "--omnummer-root", str(args.omnummer_root),
        "--filter-config", str(args.filter_config),
        "--fare-fee", str(args.fare_fee),
        "--fare-rate-km", str(args.fare_rate_km),
        "--fare-detour", str(args.fare_detour),
        "--beta-time", str(args.beta_time),
        "--beta-distance", str(args.beta_distance),
        "--cache-root", str(args.cache_root),
        # Scenario inserted into the path HERE, because run.py
        # namespaces by IKOB only (see module docstring).
        "--out", str(args.out_root / scenario),
        "--log-level", args.log_level,
    ]
    if nesting:
        cmd.append("--nesting-test")
    return cmd


def main(argv=None):
    args = parse_args(argv)

    pairs = [(ikob, scenario)
             for ikob in args.ikobs
             for scenario in args.scenarios]

    log_dir = args.out_root / "logs"

    if args.dry_run:
        for ikob, scenario in pairs:
            done = is_done(args.out_root / scenario / ikob)
            state = ("SKIP (done)"
                     if done and not args.force else "RUN")
            print(f"{state:12s} {ikob} {scenario}")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)

    results = []          # (ikob, scenario, status, seconds)
    nesting_pending = args.nesting == "first"
    sweep_start = time.monotonic()

    for i, (ikob, scenario) in enumerate(pairs, 1):
        tag = f"{ikob} {scenario}"
        out_dir = args.out_root / scenario / ikob

        if not args.force and is_done(out_dir):
            print(f"[{i:3d}/{len(pairs)}] {tag}: SKIP (outputs exist)")
            results.append((ikob, scenario, "SKIP", 0.0))
            continue

        nesting = (args.nesting == "all") or nesting_pending
        cmd = build_cmd(args, ikob, scenario, nesting=nesting)
        log_path = log_dir / f"{ikob}_{scenario}.log"

        print(f"[{i:3d}/{len(pairs)}] {tag}: running"
              f"{' (+nesting test)' if nesting else ''} "
              f"-> {log_path.name}", flush=True)

        start = time.monotonic()
        with open(log_path, "w") as log:
            # The exact argv at the top of the log, so any run can be
            # reproduced standalone by copy-paste.
            log.write("# " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log,
                                  stderr=subprocess.STDOUT)
        elapsed = time.monotonic() - start

        if proc.returncode == 0:
            # Belt and braces: exit 0 but missing outputs would
            # otherwise be discovered only on resume, as a bogus SKIP.
            if not is_done(out_dir):
                print(f"    FAILED: exit 0 but outputs incomplete "
                      f"in {out_dir}")
                results.append((ikob, scenario, "FAIL", elapsed))
            else:
                print(f"    ok ({elapsed:.1f}s)")
                results.append((ikob, scenario, "OK", elapsed))
                # Nesting certification consumed only by a SUCCESSFUL
                # run; a failed first run should not disarm it.
                nesting_pending = False
        else:
            print(f"    FAILED (exit {proc.returncode}, "
                  f"{elapsed:.1f}s) — see {log_path}")
            results.append((ikob, scenario, "FAIL", elapsed))
            if args.stop_on_error:
                break

    # ── Summary ──────────────────────────────────
    total = time.monotonic() - sweep_start
    n_ok = sum(1 for r in results if r[2] == "OK")
    n_skip = sum(1 for r in results if r[2] == "SKIP")
    failures = [r for r in results if r[2] == "FAIL"]

    print(f"\nSweep finished in {total/60:.1f} min: "
          f"{n_ok} ok, {n_skip} skipped, {len(failures)} failed.")
    if failures:
        print("Failed runs (rerun this script to retry just these — "
              "completed runs are skipped automatically):")
        for ikob, scenario, _, secs in failures:
            print(f"  {ikob} {scenario}  "
                  f"(log: {log_dir / f'{ikob}_{scenario}.log'})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())