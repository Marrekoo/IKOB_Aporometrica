"""A set of accessibility runs from a run plan.

    python -m ikob2.cli.batch --plan paper/runs.toml --data-root <root> \\
        [--dry-run] [--jobs 2] [--only REGEX] [--skip-existing]

The plan (TOML) names the specifications, the scenarios and the run sets:

    [specs]                     tag -> arguments of cli.accessibility
    m2 = ["--spec", "m2"]
    [scenarios]                 name -> arguments
    s1 = ["--price-scales", "lime_price_scales_all_50.csv"]
    [[sets]]
    name = "{prefix}_{spec}_{scenario}"   # run name; {prefix} {spec} {scenario}
    prefix = "sp"
    specs = ["m2"]              # or "all"
    scenarios = "all"           # or a list
    args = ["--modes", "pt"]    # arguments of every run of the set

Every run is `python -m ikob2.cli.accessibility --data-root <root> --run
<name> <set args> <spec args> <scenario args>`, with `{root}` in an
argument replaced by the data folder. The commands are written to
`outputs/logs/<plan>_commands.sh` and each run's output to
`outputs/logs/<plan>/<name>.log`; `--dry-run` only writes the commands.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ikob2 import params as params_mod
from ikob2.utils.paths import DataLayout


def expand(plan: dict) -> list[tuple[str, list[str]]]:
    """(run name, arguments) for every run of the plan's sets, in order. A
    run name that occurs twice is an error."""
    specs, scens = plan.get("specs", {}), plan.get("scenarios", {})
    runs, seen = [], set()
    for k, s in enumerate(plan.get("sets", [])):
        want_specs = list(specs) if s.get("specs", "all") == "all" else s["specs"]
        want_scens = list(scens) if s.get("scenarios", "all") == "all" else s["scenarios"]
        unknown = ([x for x in want_specs if x not in specs]
                   + [x for x in want_scens if x not in scens])
        if unknown:
            raise ValueError(f"set {k}: unknown spec(s) or scenario(s) {unknown}.")
        for spec in want_specs:
            for scen in want_scens:
                name = s.get("name", "{prefix}_{spec}_{scenario}").format(
                    prefix=s.get("prefix", ""), spec=spec, scenario=scen)
                if name in seen:
                    raise ValueError(f"Run {name} occurs twice in the plan.")
                seen.add(name)
                runs.append((name, [str(a) for a in s.get("args", [])]
                             + [str(a) for a in specs[spec]]
                             + [str(a) for a in scens[scen]]))
    return runs


def command(root: Path, name: str, args: list[str]) -> list[str]:
    """The accessibility command of one run; `{root}` in an argument is the
    data folder."""
    return [sys.executable, "-m", "ikob2.cli.accessibility", "--data-root",
            str(root), "--run", name,
            *(a.replace("{root}", str(root)) for a in args)]


def main(argv=None) -> None:
    """Command line entry point (`python -m ikob2.cli.batch`)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--plan", required=True, help="run plan (TOML)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--jobs", type=int, default=1, help="runs in parallel")
    p.add_argument("--only", default=None, metavar="REGEX",
                   help="only the runs whose name matches")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip runs whose run.json exists")
    p.add_argument("--dry-run", action="store_true",
                   help="write the commands without running them")
    args = p.parse_args(argv)
    root = params_mod.data_root(args.data_root, params_mod.from_args(args))
    lay = DataLayout(root)
    plan_path = Path(args.plan)
    runs = expand(tomllib.loads(plan_path.read_text()))
    if args.only:
        runs = [r for r in runs if re.search(args.only, r[0])]
    if args.skip_existing:
        runs = [r for r in runs if not (lay.run_dir(r[0]) / "run.json").exists()]
    logs = lay.root / "outputs" / "logs" / plan_path.stem
    logs.mkdir(parents=True, exist_ok=True)
    script = logs.parent / f"{plan_path.stem}_commands.sh"
    script.write_text("\n".join(shlex.join(command(root, n, a))
                                for n, a in runs) + "\n")
    print(f"{len(runs)} runs; commands in {script}")
    if args.dry_run:
        return

    def one(run):
        name, a = run
        with open(logs / f"{name}.log", "w") as fh:
            code = subprocess.run(command(root, name, a), stdout=fh,
                                  stderr=subprocess.STDOUT).returncode
        print(f"{'ok  ' if code == 0 else 'FAIL'} {name}", flush=True)
        return name, code

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(one, runs))
    failed = [n for n, c in results if c != 0]
    print(f"{len(results) - len(failed)} of {len(results)} runs succeeded"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
