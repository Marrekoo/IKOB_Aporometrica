"""
Create (or check) the data folder layout.

    python -m ikob2.cli.layout --root <root> create [--seed-from DIR | --no-seed]
    python -m ikob2.cli.layout --root <root> link <target> <inputs-subfolder> [--name NAME]

`create` makes the folders and a README, and copies the reference files
shipped in the repository's data/ folder (budgets, time margins, tariff
tables, StatLine snapshots, detour calibration) where they are missing. It
never overwrites, moves or deletes anything, so it can be rerun on an
existing folder. `link` adds a symlink in inputs/<subfolder> to a file that lives
elsewhere, so sources stay where they are.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ikob2 import params as params_mod
from ikob2.utils.paths import INPUT_DIRS, REPO_DATA, DataLayout


def link_input(layout: DataLayout, target: Path, subfolder: str,
               name: str | None = None) -> Path:
    if subfolder not in INPUT_DIRS:
        raise ValueError(f"inputs subfolder must be one of {INPUT_DIRS}.")
    target = Path(target).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    dest = layout.inputs / subfolder / (name or target.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() and dest.resolve() == target:
            return dest
        raise FileExistsError(f"{dest} already exists.")
    dest.symlink_to(target)
    return dest


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--root", default=None)
    sub = p.add_subparsers(dest="command", required=True)
    cr = sub.add_parser("create")
    cr.add_argument("--seed-from", default=None, metavar="DIR",
                    help="reference data folder to copy from (default: the "
                         "repository's data/)")
    cr.add_argument("--no-seed", action="store_true",
                    help="only make the folders")
    lk = sub.add_parser("link")
    lk.add_argument("target")
    lk.add_argument("subfolder", choices=INPUT_DIRS)
    lk.add_argument("--name")
    args = p.parse_args(argv)
    layout = DataLayout(params_mod.data_root(args.root,
                                             params_mod.from_args(args)))
    if args.command == "create":
        created = layout.ensure()
        print(f"{len(created)} folder(s) created under {layout.root}")
        for d in created:
            print("  ", d.relative_to(layout.root))
        if not args.no_seed:
            source = Path(args.seed_from) if args.seed_from else REPO_DATA
            if not source.is_dir():
                raise SystemExit(f"No reference data folder {source}: give "
                                 "--seed-from DIR or --no-seed.")
            copied = layout.seed(source)
            print(f"{len(copied)} reference file(s) copied from {source}")
            for f in copied:
                print("  ", f.relative_to(layout.root))
    else:
        print(link_input(layout, Path(args.target), args.subfolder,
                         args.name))


if __name__ == "__main__":
    main()
