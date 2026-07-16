import numpy as np
import pytest

from ikob2.core.competition import compute_competition
from ikob2.core.numerics import HAVE_SCIPY, maybe_to_sparse


DECAY = np.array([
    [1.0, 0.5],
    [0.5, 1.0],
])
POPULATION = np.array([10.0, 20.0])
# D_0 = 10*1 + 20*0.5 = 20
# D_1 = 10*0.5 + 20*1 = 25
EXPECTED = [20.0, 25.0]


def test_competition_simple_dense():
    comp = compute_competition(DECAY, POPULATION)
    assert np.allclose(comp, EXPECTED)


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed")
def test_competition_sparse_matches_dense():
    sparse_decay = maybe_to_sparse(DECAY.astype(np.float32))
    comp = compute_competition(sparse_decay, POPULATION)
    assert np.allclose(comp, EXPECTED)
    assert comp.ndim == 1  # matvec_T must ravel sparse matmul output


def test_competition_floor_prevents_zero():
    decay = np.zeros((2, 2))
    comp = compute_competition(decay, POPULATION)
    assert np.all(comp > 0)