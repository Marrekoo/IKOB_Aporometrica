"""
Central numeric conventions for ikob2.

Every module imports DTYPE / sparse helpers from here, so precision and
sparsity policy can be changed in exactly one place.

Ported from the legacy ikob.utils performance knobs, extended with the
safe-division and unreachable-masking idioms that were previously
copy-pasted throughout the old codebase.
"""

import numpy as np

try:
    from scipy import sparse as _sp
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _sp = None
    HAVE_SCIPY = False

# ── Performance knobs ────────────────────────────────────────────────
DTYPE = np.float32       # float32 halves memory; switch to float64 if needed
USE_SPARSE = True        # derived weight/decay matrices stored as CSR
# ─────────────────────────────────────────────────────────────────────

IKOB_INFINITE = 9999.0   # sentinel generalized cost for unreachable OD pairs


# ── Basic construction / coercion ────────────────────────────────────

def zeros(shape, dtype=None) -> np.ndarray:
    return np.zeros(shape, dtype=dtype or DTYPE)


def as_dtype(arr) -> np.ndarray:
    """Coerce to a DTYPE ndarray without copying when already correct."""
    return np.asarray(arr, dtype=DTYPE)


def is_sparse(x) -> bool:
    return HAVE_SCIPY and _sp.issparse(x)


def maybe_to_sparse(arr):
    """Convert a dense array to CSR if the sparsity policy is enabled."""
    if not USE_SPARSE or not HAVE_SCIPY:
        return arr
    if _sp.issparse(arr):
        return arr.tocsr()
    return _sp.csr_matrix(arr)


def ensure_dense(arr) -> np.ndarray:
    """Densify (needed for elementwise ops that would break on sparse)."""
    if is_sparse(arr):
        return np.asarray(arr.todense(), dtype=DTYPE)
    return np.asarray(arr)


def sparse_maximum(a, b):
    """Element-wise maximum working for any dense/sparse combination."""
    if is_sparse(a) and is_sparse(b):
        return a.maximum(b)
    if is_sparse(a):
        a = ensure_dense(a)
    if is_sparse(b):
        b = ensure_dense(b)
    return np.maximum(a, b)


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

    Replaces the `np.where(x > 0, a / np.where(x > 0, x, 1), 0)` pattern
    repeated throughout the legacy competition/reachability code.
    """
    num = as_dtype(numerator)
    den = as_dtype(denominator)
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, num / safe, fill).astype(DTYPE, copy=False)


def mask_unreachable(time_matrix, money_matrix=None, threshold: float = 0.5):
    """Unreachable OD pairs (time <= threshold) get time = IKOB_INFINITE.

    If a money matrix is supplied, its unreachable entries become 0 so
    that t + tau * m == IKOB_INFINITE regardless of tau (legacy PT-skim
    convention from ikob.utils.compute_pt_time_money).
    """
    t = np.where(time_matrix > threshold, time_matrix, IKOB_INFINITE).astype(DTYPE)
    if money_matrix is None:
        return t
    m = np.where(time_matrix > threshold, money_matrix, 0.0).astype(DTYPE)
    return t, m


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