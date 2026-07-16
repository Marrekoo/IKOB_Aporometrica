"""
CSV reading/writing — port of the battle-tested legacy ikob.utils I/O,
with dtype coercion to the ikob2 DTYPE convention.

Pure I/O: nothing in here knows about ModelState.
"""

import logging
import pathlib
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ikob2.core.numerics import DTYPE, is_sparse, ensure_dense

logger = logging.getLogger(__name__)


def read_csv(filename, type_caster=float, has_index_column=True):
    if not isinstance(filename, pathlib.Path):
        filename = pathlib.Path(filename)

    try:
        matrix = np.loadtxt(filename, dtype=type_caster, delimiter=",")
        if has_index_column:
            logger.warning(
                "Reading file %s without headers, but with an index column.",
                filename,
            )
    except ValueError:
        matrix = np.loadtxt(filename, dtype=type_caster, skiprows=1, delimiter=",")

    if has_index_column:
        _check_index_column(matrix, filename)
        matrix = matrix[:, 1:]

    if len(matrix.shape) == 2:
        if len(matrix[0, :]) == 1:
            return matrix[:, 0]
        if len(matrix[:, 0]) == 1:
            return matrix[0]
    return matrix


def read_csv_int(filename, has_index_column=True):
    return read_csv(filename, type_caster=int, has_index_column=has_index_column)


def read_csv_float32(filename, has_index_column=True):
    """Read and coerce to the model DTYPE in one step."""
    data = read_csv(filename, type_caster=float, has_index_column=has_index_column)
    return np.asarray(data, dtype=DTYPE)


def _check_index_column(matrix: npt.NDArray, filename):
    if len(matrix.shape) != 2:
        raise ValueError(
            f"Reading file {filename} as a file with index column, but the "
            f"matrix it contains is not two dimensional."
        )
    index_column = matrix[:, 0]
    prev_index = 0
    for idx in index_column:
        if abs(round(idx) - idx) > 1e-5:
            raise ValueError(
                f"Csv file {filename} has an invalid index column because "
                f"index {idx} is not integer."
            )
        idx = int(round(idx))
        if idx - prev_index != 1:
            raise ValueError(
                f"Csv file {filename} has an invalid index column because "
                f"the index is not sequential."
            )
        prev_index = idx


@dataclass
class CsvIndex:
    name: str = ""
    values: list[int] = field(default_factory=list)

    @classmethod
    def zone_index(cls, num_zones):
        return cls("zone", list(range(1, num_zones + 1)))


def write_csv(matrix, filename, index: CsvIndex | None = None, header=None):
    if not isinstance(filename, pathlib.Path):
        filename = pathlib.Path(filename)
    index = index or CsvIndex()
    header = list(header) if header else []

    if is_sparse(matrix):
        matrix = ensure_dense(matrix)
    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1) if index.values else matrix.reshape(1, -1)

    is_int = np.issubdtype(matrix.dtype, np.integer)
    data_fmt = "%d" if is_int else "%.9e"   # float32: 9 significant digits

    has_index = len(index.values) > 0
    if has_index:
        index_col = np.array(index.values).reshape(-1, 1)
        matrix = np.hstack([index_col, matrix])
        header = [index.name, *header]
        fmt = ["%d"] + [data_fmt] * (matrix.shape[1] - 1)
    else:
        fmt = data_fmt

    header_str = ",".join(header)

    # ── Fast path using pandas ──
    try:
        import pandas as pd

        df = pd.DataFrame(matrix)
        if has_index:
            df.iloc[:, 0] = df.iloc[:, 0].astype(int)
        with open(filename, "w", newline="") as f:
            if header_str:
                f.write(header_str + "\n")
            df.to_csv(f, index=False, header=False,
                      float_format=None if is_int else "%.9e")
        return
    except ImportError:
        pass

    np.savetxt(filename, matrix, fmt=fmt, delimiter=",",
               header=header_str, comments="")