"""Parameters of a run: one defaults file, overridable by a file and flags.

`defaults.toml` (next to this module) is the single place where model and
run parameters are written down. A run resolves them, in increasing
priority, from

    defaults.toml  <  --params FILE (toml or json, partial)  <  --set a.b=v
                   <  dedicated command-line flags

`load` merges and validates (a key that is not in defaults.toml is an
error, so a typo cannot pass silently) and returns a `Params`: a read-only
tree with attribute access (`p.pt.walk_kmh`) and `p.to_dict()` for run.json.
`DEFAULTS` is the packaged defaults tree, used for the default values of
library signatures so that library and command line agree.
"""

from __future__ import annotations

import copy
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULTS_PATH = Path(__file__).with_name("defaults.toml")
ENV_ROOT = "IKOB_DATA_ROOT"
# Tables whose keys are user data (a car model, a road class, an
# urbanisation class): new keys are allowed there.
OPEN_TABLES = frozenset({"car.models", "peak.factors",
                         "car.parking_arrival_min"})


class Params:
    """Read-only nested parameters with attribute and dotted access."""

    def __init__(self, tree: Mapping[str, Any]):
        object.__setattr__(self, "_tree", dict(tree))

    def __getattr__(self, name: str):
        try:
            value = self._tree[name]
        except KeyError:
            raise AttributeError(name) from None
        return Params(value) if isinstance(value, dict) else value

    def __setattr__(self, name, value):
        raise AttributeError("Params is read-only; use with_values().")

    def __getitem__(self, name: str):
        return self.get(name)

    def __contains__(self, dotted: str) -> bool:
        try:
            self.get(dotted)
        except KeyError:
            return False
        return True

    def get(self, dotted: str):
        node: Any = self._tree
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"Unknown parameter '{dotted}'.")
            node = node[part]
        return Params(node) if isinstance(node, dict) else node

    def to_dict(self) -> dict:
        return copy.deepcopy(self._tree)

    def with_values(self, values: Mapping[str, Any]) -> "Params":
        """Copy with dotted keys replaced ({'pt.walk_kmh': 4.5})."""
        tree = self.to_dict()
        for dotted, value in values.items():
            _set_dotted(tree, dotted, value, self._tree)
        return Params(tree)

    def __repr__(self):
        return f"Params({self._tree!r})"


def _set_dotted(tree: dict, dotted: str, value, schema: dict) -> None:
    parts = dotted.split(".")
    node, ref = tree, schema
    for part in parts[:-1]:
        if not isinstance(ref, dict) or part not in ref:
            raise KeyError(f"Unknown parameter '{dotted}'.")
        node, ref = node[part], ref[part]
    parent = ".".join(parts[:-1])
    if parent in OPEN_TABLES and parts[-1] not in ref:
        node[parts[-1]] = value
        return
    if not isinstance(ref, dict) or parts[-1] not in ref:
        raise KeyError(f"Unknown parameter '{dotted}'.")
    node[parts[-1]] = _coerce(value, ref[parts[-1]], dotted)


def _coerce(value, like, dotted: str):
    """Keep the type of the default (int where the default is int, ...)."""
    if isinstance(like, dict):
        if not isinstance(value, dict):
            raise TypeError(f"'{dotted}' is a table; give a table.")
        return value
    if isinstance(like, bool):
        if not isinstance(value, bool):
            raise TypeError(f"'{dotted}' must be true or false.")
        return value
    if isinstance(like, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{dotted}' must be a number, got {value!r}.")
        return type(like)(value) if isinstance(like, int) and float(
            value).is_integer() else value
    if isinstance(like, list):
        if not isinstance(value, list):
            raise TypeError(f"'{dotted}' must be a list, got {value!r}.")
        return value
    if isinstance(like, str) and not isinstance(value, str):
        raise TypeError(f"'{dotted}' must be text, got {value!r}.")
    return value


def _merge(base: dict, over: Mapping, schema: dict, prefix: str = "") -> None:
    for key, value in over.items():
        dotted = f"{prefix}{key}"
        if key not in schema:
            if prefix.rstrip(".") in OPEN_TABLES:
                base[key] = value
                continue
            raise KeyError(f"Unknown parameter '{dotted}' (not in "
                           f"defaults.toml).")
        if isinstance(schema[key], dict) and isinstance(value, Mapping):
            _merge(base[key], value, schema[key], dotted + ".")
        else:
            base[key] = _coerce(value, schema[key], dotted)


def read_file(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    with path.open("rb") as fh:
        return tomllib.load(fh)


def parse_assignment(item: str) -> tuple[str, Any]:
    """'pt.walk_kmh=4.5' -> ('pt.walk_kmh', 4.5); values are TOML literals,
    anything else is text."""
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"--set expects key=value, got {item!r}.")
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return key.strip(), value


def load(path: str | Path | None = None,
         assignments: Iterable[str] = ()) -> Params:
    """Defaults, then the file, then `key=value` assignments."""
    tree = read_file(DEFAULTS_PATH)
    schema = copy.deepcopy(tree)
    if path:
        _merge(tree, read_file(path), schema)
    p = Params(tree)
    return p.with_values(dict(parse_assignment(a) for a in assignments))


DEFAULTS = load()


def add_arguments(parser) -> None:
    """--params and --set for a command line."""
    parser.add_argument("--params", default=None, metavar="FILE",
                        help="parameter file (toml/json) overriding "
                             "ikob2/defaults.toml")
    parser.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE", dest="set_values",
                        help="override one parameter, e.g. pt.walk_kmh=4.5 "
                             "(repeatable)")


def from_args(args, flags: Mapping[str, str] | None = None) -> Params:
    """Parameters of a command line: file and --set, then each dedicated
    flag that was given (not None). `flags` maps an argparse attribute to a
    dotted parameter, e.g. {'walk_kmh': 'pt.walk_kmh'}."""
    p = load(getattr(args, "params", None), getattr(args, "set_values", []))
    over = {key: getattr(args, attr) for attr, key in (flags or {}).items()
            if getattr(args, attr, None) is not None}
    return p.with_values(over) if over else p


def repo_path(path: str | Path) -> Path:
    """A path of the parameters: as given if it exists, else relative to the
    repository (a source checkout), so `data/...` also works from elsewhere."""
    p = Path(path).expanduser()
    if p.exists() or p.is_absolute():
        return p
    root = Path(__file__).resolve().parents[2]
    return root / p if (root / p).exists() else p


def data_root(explicit: str | Path | None, params: Params) -> Path:
    """Data folder: --data-root, else $IKOB_DATA_ROOT, else paths.data_root.
    No location is built in."""
    for candidate in (explicit, os.environ.get(ENV_ROOT),
                      params.paths.data_root):
        if candidate:
            return Path(candidate).expanduser()
    raise SystemExit("No data folder: give --data-root, set "
                     f"${ENV_ROOT}, or set paths.data_root in a --params "
                     "file.")
