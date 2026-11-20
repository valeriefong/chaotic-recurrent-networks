import torch
import pytest
import numpy as np
from rnn_lyapunov.models.dynamically_critical_autograd import DynamicallyCriticalAutogradRNN
from rnn_lyapunov.models.dynamically_critical import DynamicallyCriticalRNN
from rnn_lyapunov.models.random import RandomRNN
from rnn_lyapunov.models.random_autograd import RandomAutogradRNN
from rnn_lyapunov.models.random_linear import RandomLinearRNN
from rnn_lyapunov.analysis.lyapunov import LyapunovSpectrum


# --- Ground-truth check: for a STRICTLY LINEAR system, Lyapunov exponents are exactly the real
# parts of the Jacobian's eigenvalues (a closed form, no simulation needed) -- this is the most
# direct possible test of whether the QR-based algorithm in compute() is implemented correctly,
# independent of tanh/DC-construction/anything else model-specific.
def test_full_spectrum_matches_jacobian_eigenvalues_for_strictly_linear_system():
    # g=0.9 (stable regime, matches g_crit=1 for this ensemble) keeps x bounded near the origin
    # rather than growing without bound -- there's no saturating nonlinearity here to stop it.
    rnn = RandomLinearRNN(N=30, g=0.9, seed_net=7)
    # dt=0.01, not 0.05: the NEGATIVE TAIL of the spectrum carries the discretization error, and it
    # converges much more slowly than lambda_max does. Measured max|computed - exact| over the full
    # spectrum is 0.078 at dt=0.05, 0.017 at dt=0.01, 0.008 at dt=0.005 -- clean first order in dt.
    # At dt=0.05 this test asserts something forward Euler cannot deliver at atol=0.05.
    # (lambda_max alone is accurate to 3e-4 even at dt=0.05 -- the tail is the demanding part.)
    result = LyapunovSpectrum(rnn).compute(T_sim=200, dt=0.01, T_transient=0.0, n_LE=None)

    # Closed-form answer: real parts of the (constant, state-independent) continuous-time
    # Jacobian's eigenvalues, sorted descending -- exactly what compute() should also produce.
    J = (-np.eye(30) + rnn.W.numpy()) / rnn.tau
    expected = np.sort(np.linalg.eigvals(J).real)[::-1]

    # .spectrum, not .exponents: `expected` is in rank order, so the comparison is only meaningful
    # against rank order. .exponents is in QR direction order, which can depart from rank order.
    assert np.allclose(result.spectrum, expected, atol=0.05)


# --- Sanity checks on RandomRNN with a chosen g ---
def test_compute_stable_regime():
    # g < 1: fixed point at origin is stable, expect lambda_max < 0
    rnn = RandomRNN(N=20, g=0.5)
    spectrum = LyapunovSpectrum(rnn)
    result = spectrum.compute(T_sim=500, dt=0.1) # returns LyapunovResult object
    # max exponent is property of result object
    assert result.max_exponent < 0

def test_compute_chaotic_regime():
    # g > 1: expect lambda_max > 0
    rnn = RandomRNN(N=20, g=5.0)
    spectrum = LyapunovSpectrum(rnn)
    result = spectrum.compute(T_sim=500, dt=0.1) # returns LyapunovResult object
    assert result.max_exponent > 0


def test_spectrum_is_sorted_descending():
    # .spectrum is the physical spectrum, largest first. .exponents stays in raw QR direction order
    # (see LyapunovResult.spectrum), column-aligned with local_exponents.
    rnn = RandomRNN(N=20, g=5.0) # chaotic regime: largest exponent positive, smallest negative
    result = LyapunovSpectrum(rnn).compute(T_sim=500, dt=0.1)
    assert np.all(np.diff(result.spectrum) <= 0)
    assert result.local_exponents.shape[1] == result.exponents.size


# --- Cross-validation: analytic Jacobian vs autograd Jacobian ---
def test_jacobian_matches_analytic_and_autograd():
    # Same N, g, seed_net -> RandomAutogradRNN and RandomRNN build the *same* W
    # (both seed np.random the same way), so jacobians at the same state should match.
    analytic = RandomRNN(N=50, g=3.0, seed_net=10)
    autograd = RandomAutogradRNN(N=50, g=3.0, seed_net=10)
    assert torch.allclose(analytic.W, autograd.W)  # confirms same W before comparing jacobians

    x = torch.randn(50) # random initial state
    dt = 0.1
    J_analytic = analytic.jacobian(x, dt)
    J_autograd = autograd.jacobian(x, dt)  # uses inherited jacrev
    assert torch.allclose(J_analytic, J_autograd)


def test_compute_matches_between_random_analytic_and_autograd():
    # Full pipeline: same W, same trajectory algorithm -> should agree closely.
    # Keep N and T_sim small here since autograd jacrev is much slower per step.
    analytic = RandomRNN(N=15, g=3.0, seed_net=10)
    autograd = RandomAutogradRNN(N=15, g=3.0, seed_net=10)
    result_a = LyapunovSpectrum(analytic).compute(T_sim=50, dt=0.1, seed_IC=12, seed_ONS=12)
    result_b = LyapunovSpectrum(autograd).compute(T_sim=50, dt=0.1, seed_IC=12, seed_ONS=12)
    assert np.allclose(result_a.exponents, result_b.exponents, atol=1e-4)


def test_compute_matches_between_dynamically_critical_analytic_and_autograd():
    # Full pipeline: same W, same trajectory algorithm -> should agree closely.
    # Keep N and T_sim small here since autograd jacrev is much slower per step.
    analytic = DynamicallyCriticalRNN(N=15, g=3.0, seed_net=10)
    autograd = DynamicallyCriticalAutogradRNN(N=15, g=3.0, seed_net=10)
    result_a = LyapunovSpectrum(analytic).compute(T_sim=50, dt=0.1, seed_IC=12, seed_ONS=12)
    result_b = LyapunovSpectrum(autograd).compute(T_sim=50, dt=0.1, seed_IC=12, seed_ONS=12)
    assert np.allclose(result_a.exponents, result_b.exponents, atol=1e-4)


# --- Cross-validation: QR-based spectrum vs brute-force DNS estimate ---
def test_dns_matches_compute_max_exponent():
    # dns() brute-forces the largest exponent by tracking trajectory divergence;
    # it should agree (loosely -- it's noisier) with compute()'s max_exponent.
    rnn = RandomRNN(N=20, g=5.0)
    spectrum = LyapunovSpectrum(rnn)
    result = spectrum.compute(T_sim=500, dt=0.1)
    lambda_max_dns, _ = spectrum.dns(T_sim=500, dt=0.1)
    assert np.isclose(lambda_max_dns.item(), result.max_exponent, atol=0.05)


# Notes on the choices above: g = 0.5 is safely stable and g = 5 safely chaotic at these small N,
# so the sign tests are robust; N and T_sim are kept small so the suite stays fast. Jacobian
# comparisons can use a tight tolerance (float32 roundoff only), while full compute() comparisons
# need a looser one, because roundoff accumulates differently along two separately evolved
# trajectories.
