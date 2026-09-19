"""
Create (or check) the data folder layout.

    python -m ikob2.cli.layout create --root "/home/marco/IKOB data"
    python -m ikob2.cli.layout link <target> <inputs-subfolder> [--name NAME]

`create` makes the folders and a README and never moves or deletes
anything. `link` adds a symlink in inputs/<subfolder> to a file that lives
elsewhere, so sources stay where they are.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ikob2.utils.paths import DEFAULT_ROOT, INPUT_DIRS, DataLayout


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
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("create")
    lk = sub.add_parser("link")
    lk.add_argument("target")
    lk.add_argument("subfolder", choices=INPUT_DIRS)
    lk.add_argument("--name")
    args = p.parse_args(argv)
    layout = DataLayout(Path(args.root))
    if args.command == "create":
        created = layout.ensure()
        print(f"{len(created)} folder(s) created under {layout.root}")
        for d in created:
            print("  ", d.relative_to(layout.root))
    else:
        print(link_input(layout, Path(args.target), args.subfolder,
                         args.name))


if __name__ == "__main__":
    main()
