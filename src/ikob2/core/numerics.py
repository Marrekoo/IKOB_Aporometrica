"""
Central numeric conventions for ikob2.

Every module imports DTYPE / sparse helpers from here, so precision and
sparsity policy can be changed in exactly one place.

It also holds the safe-division idiom and memory accounting.
"""

import numpy as np

from ikob2.params import DEFAULTS

try:
    from scipy import sparse as _sp
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _sp = None
    HAVE_SCIPY = False

# ── Performance knobs ────────────────────────────────────────────────
DTYPE = np.float32       # float32 halves memory; switch to float64 if needed
USE_SPARSE = DEFAULTS.numerics.use_sparse   # weight matrices as CSR when sparse
SPARSE_MAX_DENSITY = DEFAULTS.numerics.sparse_max_density
# ─────────────────────────────────────────────────────────────────────


# ── Basic construction / coercion ────────────────────────────────────

def zeros(shape, dtype=None) -> np.ndarray:
    return np.zeros(shape, dtype=dtype or DTYPE)


def as_dtype(arr) -> np.ndarray:
    """Coerce to a DTYPE ndarray without copying when already correct."""
    return np.asarray(arr, dtype=DTYPE)


def is_sparse(x) -> bool:
    return HAVE_SCIPY and _sp.issparse(x)


def maybe_to_sparse(arr, max_density: float | None = None):
    """Convert a dense array to CSR if the sparsity policy is enabled and
    the array is sparse enough (share of non-zeros below `max_density`,
    default numerics.sparse_max_density). A nearly full matrix is faster
    as a dense array: building a CSR copy costs more than the products it
    would save."""
    if not USE_SPARSE or not HAVE_SCIPY:
        return arr
    if _sp.issparse(arr):
        return arr.tocsr()
    limit = SPARSE_MAX_DENSITY if max_density is None else max_density
    if arr.size and np.count_nonzero(arr) >= limit * arr.size:
        return arr
    return _sp.csr_matrix(arr)


def ensure_dense(arr) -> np.ndarray:
    """Densify (needed for elementwise ops that would break on sparse)."""
    if is_sparse(arr):
        return np.asarray(arr.todense(), dtype=DTYPE)
    return np.asarray(arr)


# ── Sparse/dense safe linear algebra ─────────────────────────────────

def matvec(W, v) -> np.ndarray:
    """W @ v as a 1-D DTYPE ndarray, regardless of W's type.

    scipy.sparse matmuls can return matrix-shaped results; the ravel()
    normalisation here prevents subtle broadcasting bugs downstream.
    """
    out = W @ v
    return np.asarray(out, dtype=DTYPE).ravel()


def matvec_T(W, v) -> np.ndarray:
    """W.T @ v as a 1-D DTYPE ndarray."""
    out = W.T @ v
    return np.asarray(out, dtype=DTYPE).ravel()


# ── Domain idioms ────────────────────────────────────────────────────

def safe_divide(numerator, denominator, fill: float = 0.0) -> np.ndarray:
    """numerator / denominator where denominator > 0, else *fill*.

    Used where zero-competition or empty zones must contribute nothing.
    """
    num = as_dtype(numerator)
    den = as_dtype(denominator)
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, num / safe, fill).astype(DTYPE, copy=False)


# ── Memory accounting ────────────────────────────────────────────────

def nbytes_of(x) -> int:
    """Byte size of a dense or sparse array (sparse-aware)."""
    if is_sparse(x):
        return int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)
    try:
        return int(x.nbytes)
    except AttributeError:
        import sys
        return sys.getsizeof(x)