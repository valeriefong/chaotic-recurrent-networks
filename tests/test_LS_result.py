import pytest 
import numpy as np
from rnn_lyapunov.analysis.lyapunov import LyapunovResult



def test_max_exponent():
    result = LyapunovResult(
        exponents=np.array([0.3, 0.1, -0.2, -0.5]),
        local_exponents=np.zeros((10, 4)),
        participation_ratio=np.zeros(10),
    )
    assert result.max_exponent == pytest.approx(0.3)

def test_entropy():
    result = LyapunovResult(
        exponents=np.array([0.3, 0.1, -0.2, -0.5]),
        local_exponents=np.zeros((10, 4)),
        participation_ratio=np.zeros(10),
    )
    assert result.entropy == pytest.approx(0.4)  # sum of the positive exponents: 0.3 + 0.1

def test_lyapunov_dimension():
    result = LyapunovResult(
        exponents=np.array([0.3, 0.1, -0.2, -0.5]),
        local_exponents=np.zeros((10, 4)),
        participation_ratio=np.zeros(10),
    )
    assert result.lyapunov_dimension == 3.4



def test_lyapunov_dimension_all_negative():
    result = LyapunovResult(
        exponents=np.array([-0.1, -0.2]),
        local_exponents=np.zeros((5, 2)),
        participation_ratio=np.zeros(5),
    )
    assert result.lyapunov_dimension == pytest.approx(0.0)


def _result(exponents, n_steps=10):
    exponents = np.asarray(exponents, dtype=float)
    return LyapunovResult(
        exponents=exponents,
        local_exponents=np.zeros((n_steps, exponents.size)),
        participation_ratio=np.zeros(n_steps),
    )


# Every test above feeds a PRE-SORTED array, so none of them could catch an ordering bug. The QR
# solver returns exponents in Gram-Schmidt direction order, which is only descending on average
# over infinite time; at finite T_sim near-degenerate exponents come back out of order. So each
# scalar has to hold on a shuffled array too.
SHUFFLED = [-0.2, 0.3, -0.5, 0.1]   # same multiset as the sorted cases above


def test_max_exponent_unsorted_input():
    assert _result(SHUFFLED).max_exponent == pytest.approx(0.3)


def test_entropy_unsorted_input():
    assert _result(SHUFFLED).entropy == pytest.approx(0.4)


def test_lyapunov_dimension_unsorted_input():
    assert _result(SHUFFLED).lyapunov_dimension == pytest.approx(3.4)


def test_positive_exponent_hidden_behind_negative_first_direction():
    # exponents[0] is negative while the spectrum does contain a positive exponent. Reading
    # exponents[0] as the largest exponent would report a stable network with dimension 0.
    result = _result([-0.001, 0.004, -0.01, -0.2])
    assert result.max_exponent == pytest.approx(0.004)
    assert result.lyapunov_dimension > 1.0


def test_exponents_attribute_is_not_reordered():
    # exponents must stay column-aligned with local_exponents, so the sort lives in .spectrum
    # and must not touch the stored array.
    result = _result(SHUFFLED)
    _ = (result.max_exponent, result.entropy, result.lyapunov_dimension)
    np.testing.assert_array_equal(result.exponents, np.array(SHUFFLED))
    np.testing.assert_array_equal(result.spectrum, np.array([0.3, 0.1, -0.2, -0.5]))


def test_dimension_of_all_zero_spectrum():
    # Every exponent exactly zero. Must not return N.
    assert _result(np.zeros(8)).lyapunov_dimension == pytest.approx(0.0)


def test_dimension_of_limit_cycle():
    # lambda_1 = 0 exactly, everything else contracting -> D = 1
    assert _result([0.0, -0.3, -0.7]).lyapunov_dimension == pytest.approx(1.0)
