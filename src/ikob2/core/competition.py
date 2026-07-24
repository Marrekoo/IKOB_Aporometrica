"""
Destination-level competition term (Shen 1998 pass 1).

    V_j = sum_k P_k * f(c_kj)  =  (D.T @ P)_j

Returns the RAW competition vector, zeros included. Zero-competition
zones (opportunities nobody can reach) must contribute ZERO
accessibility, not O / floor; the division is handled downstream with
safe_divide. The old floor-at-1e-6 behaviour let a single empty-but-
employed zone inject O * 1e6 into every origin's accessibility.
"""

import numpy as np

from ikob2.core.numerics import as_dtype, matvec_T


def compute_competition(decay_matrix, population) -> np.ndarray:
    """
    Parameters
    ----------
    decay_matrix : (n, n) dense ndarray or CSR matrix, f(c_ij)
    population : (n,) origin population

    Returns
    -------
    (n,) dense float32 competition per destination zone (may contain
    zeros — pair with safe_divide downstream).
    """
    return matvec_T(decay_matrix, as_dtype(population))