"""The data deposit of the paper runs (Zenodo): the inputs that cannot be
downloaded again in the same version, the skims, and the runs and tables.

    python -m ikob2.cli.archive --data-root <root> build --out <folder> \\
        [--plan paper/runs.toml --plan paper/perturbation.toml] [--odin-aggregates]
    python -m ikob2.cli.archive upload --folder <folder> \\
        --metadata paper/zenodo_data.json [--sandbox]

`build` writes to --out, in the layout of a data folder (DataLayout), so
that the unpacked deposit together with `cli.layout create` (the reference
files of the repository's data/) is a data folder for the paper runs:

  inputs.tar        GTFS feed, OpenStreetMap extracts (national and the
                    provinces of paths.osm_regions), KWB GeoPackage of the
                    model year, LISA municipal table
  intermediate.tar  skim stores of the study area, imputed sector jobs and,
                    with --odin-aggregates, the ODiN-derived car availability
                    and PT spending tables
  outputs.tar.gz    the run folders of the plans, the paper tables
                    (outputs/comparisons/specs, targeting, precision)
  MANIFEST.sha256   SHA-256 of every file inside the archives (sha256sum
                    format, paths relative to the data folder); the same
                    hashes as `input_files` in each run's run.json
  README.md         what the deposit holds and how to use it

`upload` creates a Zenodo deposition with these files and the metadata of
--metadata, and leaves it as an unpublished draft: the author reviews and
publishes it on zenodo.org. The token is read from $ZENODO_TOKEN (scope
deposit:write); --sandbox uses sandbox.zenodo.org.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tomllib
import urllib.request
from pathlib import Path

from ikob2 import params as params_mod
from ikob2.cli.batch import expand
from ikob2.utils.paths import DataLayout

TABLES = ("specs", "targeting", "precision")
ODIN_AGGREGATES = ("ownership/car_availability_utrecht.csv",
                   "ownership/pt_spend_utrecht.csv")


def contents(lay: DataLayout, prm, plans: list[Path],
             odin_aggregates: bool = False) -> dict[str, list[Path]]:
    """Files of each archive (absolute paths under the data folder)."""
    p = prm.paths
    year = prm.accessibility.kwb_year
    inputs = [lay.inputs / "gtfs" / "gtfs-nl.zip",
              lay.inputs / "osm" / p.osm_national,
              *[lay.inputs / "osm" / f"{r}.osm.pbf" for r in p.osm_regions],
              lay.kwb(year, p.kwb_version), lay.lisa()]
    skims = lay.intermediate / "skims"
    inter = [f for d in sorted(skims.glob(f"{p.study}*")) if d.is_dir()
             for f in sorted(d.rglob("*")) if f.is_file()]
    inter.append(lay.sector_jobs(prm.accessibility.jobs_year))
    if odin_aggregates:
        inter += [lay.intermediate / f for f in ODIN_AGGREGATES]
    runs = [name for plan in plans
            for name, _ in expand(tomllib.loads(Path(plan).read_text()))]
    outputs = [f for r in runs for f in sorted(lay.run_dir(r).rglob("*"))
               if f.is_file()]
    outputs += [f for t in TABLES
                for f in sorted((lay.comparison_dir() / t).rglob("*"))
                if f.is_file()]
    out = {"inputs.tar": inputs, "intermediate.tar": inter,
           "outputs.tar.gz": outputs}
    missing = [str(f) for files in out.values() for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError("Missing: " + "; ".join(missing[:10])
                                + (f" (+{len(missing) - 10})" if len(missing) > 10 else ""))
    return out


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, as in run.json `input_files`."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build(lay: DataLayout, prm, plans: list[Path], out: Path,
          odin_aggregates: bool = False) -> dict[str, int]:
    """Write the archives, MANIFEST.sha256 and README.md to `out`; returns
    the number of files per archive. Symbolic links are stored as the files
    they point to."""
    out.mkdir(parents=True, exist_ok=True)
    root = lay.root.resolve()
    lines, counts = [], {}
    for name, files in contents(lay, prm, plans, odin_aggregates).items():
        mode = "w:gz" if name.endswith(".gz") else "w"
        with tarfile.open(out / name, mode, dereference=True) as tar:
            for f in files:
                rel = Path(os.path.abspath(f)).relative_to(root).as_posix()
                tar.add(f, arcname=rel)
                lines.append(f"{sha256(f)}  {rel}")
        counts[name] = len(files)
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    (out / "README.md").write_text(readme(counts, plans, odin_aggregates))
    return counts


def readme(counts: dict[str, int], plans: list[Path], odin_aggregates: bool) -> str:
    """README.md of the deposit."""
    rows = "\n".join(f"| `{k}` | {v} |" for k, v in counts.items())
    plan_list = ", ".join(f"`{Path(p).as_posix()}`" for p in plans)
    odin = ("the ODiN-derived car availability and PT spending per household "
            "type and decile (aggregates)" if odin_aggregates else
            "not the ODiN-derived car availability: runs with `--ownership` "
            "need `cli.segments car-availability` on ODiN microdata (DANS)")
    return f"""# IKOB Aporometrica: data of the Utrecht paper runs

The inputs, skims, runs and tables behind "Either you can reach it or you
cannot" (Utrecht shared bicycles), for the software IKOB Aporometrica
(https://github.com/Marrekoo/IKOB_Aporometrica).

| Archive | Files |
|---|---|
{rows}

Unpack all archives into one folder and seed it with the repository's
reference files:

    for a in *.tar *.tar.gz; do tar -xf "$a" -C <root>; done
    sha256sum -c MANIFEST.sha256          # in <root>
    python -m ikob2.cli.layout --root <root> create

`<root>` is then a data folder for the run plans {plan_list}
(`python -m ikob2.cli.batch --plan ... --data-root <root>`); the stored runs
under `outputs/runs/` are the reference to compare with. The deposit holds
{odin}. The steps are in `docs/reproduce.md` of the repository.

Licences: OpenStreetMap data and the skims derived from them, ODbL 1.0
(© OpenStreetMap contributors); GTFS NL, OVapi (http://gtfs.ovapi.nl);
CBS Kerncijfers wijken en buurten, CC BY 4.0; LISA municipal data (LISA,
https://www.lisa.nl/gratis-data/); the model outputs, CC BY 4.0.
"""


ZENODO = {"zenodo": "https://zenodo.org/api", "sandbox": "https://sandbox.zenodo.org/api"}


def _call(method: str, url: str, token: str, data=None, json_body=None,
          size: int | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    if size is not None:
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(size)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def upload(folder: Path, metadata: dict, token: str, sandbox: bool = False) -> str:
    """Create a draft deposition with every file of `folder`; returns the
    URL of the draft (not published)."""
    api = ZENODO["sandbox" if sandbox else "zenodo"]
    dep = _call("POST", f"{api}/deposit/depositions", token, json_body={})
    for f in sorted(folder.iterdir()):
        if f.is_file():
            with open(f, "rb") as fh:
                _call("PUT", f"{dep['links']['bucket']}/{f.name}", token,
                      data=fh, size=f.stat().st_size)
    _call("PUT", f"{api}/deposit/depositions/{dep['id']}", token,
          json_body={"metadata": metadata})
    return dep["links"]["html"]


def main(argv=None) -> None:
    """Command line entry point (`python -m ikob2.cli.archive`)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    params_mod.add_arguments(p)
    p.add_argument("--data-root", default=None)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the deposit folder")
    b.add_argument("--out", required=True)
    b.add_argument("--plan", action="append", default=None,
                   help="run plan(s) whose runs to include (default "
                        "paper/runs.toml and paper.precision_plan)")
    b.add_argument("--odin-aggregates", action="store_true",
                   help="include the ODiN-derived car availability and PT "
                        "spending tables")
    u = sub.add_parser("upload", help="create an unpublished Zenodo draft")
    u.add_argument("--folder", required=True)
    u.add_argument("--metadata", required=True, help="JSON with the Zenodo metadata")
    u.add_argument("--sandbox", action="store_true")
    args = p.parse_args(argv)
    if args.command == "upload":
        token = os.environ.get("ZENODO_TOKEN")
        if not token:
            raise SystemExit("Set $ZENODO_TOKEN (zenodo.org: Applications, "
                             "personal access token, scope deposit:write).")
        meta = json.loads(Path(args.metadata).read_text())
        print(upload(Path(args.folder), meta, token, args.sandbox))
        return
    prm = params_mod.from_args(args, {})
    lay = DataLayout(params_mod.data_root(args.data_root, prm))
    plans = [Path(x) for x in (args.plan or ["paper/runs.toml", prm.paper.precision_plan])]
    counts = build(lay, prm, plans, Path(args.out), args.odin_aggregates)
    for name, n in counts.items():
        print(f"{name}: {n} files")
    print(f"MANIFEST.sha256 and README.md written to {args.out}")


if __name__ == "__main__":
    main()
