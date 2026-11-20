"""
Tests for the fixed-step RK4 integrator and its exact tangent map.

The logic of the suite:

  1. `flow_jacobian` (the Jacobian of the vector field, used by RK4) is validated against the
     analytic Jacobian of the Euler step, via the Euler identity Df(x, dt) = I + dt * J(x).
     If this passes, both are the same derivative, one with dt folded in and one without.
  2. The RK4 tangent map is validated against central finite differences of the RK4 step
     itself. This is the check that matters: it certifies that the tangent map is the exact
     derivative of the step map actually being iterated, which is the property the QR
     algorithm needs and the property adaptive RK45 does not have.
  3. Convergence order is measured, not assumed -- including on the barcode model, whose
     ReLU kink is expected to reduce it.
  4. The Euler path is pinned exactly, so switching integrators is the ONLY thing that
     can change a number.
  5. End-to-end against a closed-form spectrum, the one case where the right answer is
     known independently of any solver.
"""
import numpy as np
import pytest
import torch

from rnn_lyapunov.analysis.lyapunov import LyapunovSpectrum
from rnn_lyapunov.models.base import RNN as RNN_BASE
from rnn_lyapunov.models.barcode import BarcodeRNN
from rnn_lyapunov.models.dynamically_critical import DynamicallyCriticalRNN
from rnn_lyapunov.models.random import RandomRNN
from rnn_lyapunov.models.random_linear import RandomLinearRNN


def _models(N=30, g=1.5):
    return [
        ("Random", RandomRNN(N=N, g=g, seed_net=0)),
        ("DC", DynamicallyCriticalRNN(N=N, g=g, seed_net=0, critical_modes=1)),
        ("Barcode", BarcodeRNN(N=N, g=7.0, seed_net=0)),
        ("RandomLinear", RandomLinearRNN(N=N, g=g, seed_net=0)),
    ]


@pytest.mark.parametrize("dt", [0.01, 0.001])
def test_flow_jacobian_matches_trusted_analytic_step_jacobian(dt):
    """Df(x, dt) == I + dt * J(x), the forward-Euler relation.

    Validates flow_jacobian against the analytic Jacobian of the Euler step. Models that do
    not override flow_jacobian are covered too: there the
    test confirms that autodiff of `dynamics` agrees with the hand-derived step Jacobian,
    which independently checks the hand derivation.
    """
    torch.manual_seed(0)
    for name, m in _models():
        x = torch.randn(m.N) * 0.7
        Df = m.jacobian(x, dt)
        J = m.flow_jacobian(x)
        approx = torch.eye(m.N) + dt * J
        err = (Df - approx).abs().max().item()
        scale = max(Df.abs().max().item(), 1.0)
        assert err / scale < 1e-5, f"{name}: |Df - (I + dt J)| = {err:.3e} (dt={dt})"


def test_analytic_flow_jacobian_overrides_match_autodiff():
    """Every hand-derived flow_jacobian must equal autodiff of `dynamics`.

    The overrides exist purely for speed -- the base class's jacrev fallback is already
    correct. So the override must agree with it exactly, or it would be a wrong answer with no
    error in the code path that is actually used. Checked in float64 so the comparison is
    not roundoff-limited.
    """
    from torch.func import jacrev
    torch.set_default_dtype(torch.float64)
    try:
        for name, m in _models():
            assert type(m).flow_jacobian is not RNN_BASE.flow_jacobian, (
                f"{name} has no analytic override -- it would fall back to autodiff, which "
                f"costs ~5x per build and RK4 needs four per step")
            for k, v in list(vars(m).items()):
                if torch.is_tensor(v):
                    setattr(m, k, v.double())
            m._eye_cache = None
            torch.manual_seed(0)
            x = (torch.randn(m.N) * 0.7).double()
            ana = m.flow_jacobian(x)
            auto = jacrev(m.dynamics)(x)
            err = (ana - auto).abs().max().item()
            scale = max(auto.abs().max().item(), 1.0)
            assert err / scale < 1e-12, f"{name}: |analytic - autodiff| = {err:.3e}"
    finally:
        torch.set_default_dtype(torch.float32)


def test_rk4_tangent_map_is_exact_derivative_of_rk4_step():
    """Central finite differences of step_rk4 must reproduce the RK4 tangent map.

    This is the load-bearing correctness test for RK4 in the QR loop.
    """
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    try:
        dt = 0.01
        eps = 1e-6
        for name, m in _models():
            # promote model tensors to float64 so the FD check is not roundoff-limited
            for k, v in list(vars(m).items()):
                if torch.is_tensor(v):
                    setattr(m, k, v.double())
            x = (torch.randn(m.N) * 0.7).double()
            n_col = 4
            Q = torch.linalg.qr(torch.randn(m.N, n_col).double())[0]

            _, Q_tan = m.step_and_tangent(x, dt, Q, integrator='rk4')

            Q_fd = torch.zeros_like(Q_tan)
            for j in range(n_col):
                v = Q[:, j]
                plus = m.step_rk4(x + eps * v, dt)
                minus = m.step_rk4(x - eps * v, dt)
                Q_fd[:, j] = (plus - minus) / (2 * eps)

            err = (Q_tan - Q_fd).abs().max().item()
            scale = max(Q_tan.abs().max().item(), 1.0)
            # Barcode's ReLU is only piecewise smooth; FD across a kink is meaningless, so
            # allow a looser bound there. The tanh models must be tight.
            tol = 1e-4 if name == "Barcode" else 1e-7
            assert err / scale < tol, f"{name}: |tangent - FD| = {err:.3e}"
    finally:
        torch.set_default_dtype(torch.float32)


def test_step_and_tangent_euler_matches_step_and_jacobian_exactly():
    """integrator='euler' must reproduce `jacobian(x,dt) @ Q` and `step(x,dt)` exactly.

    Not approximately -- exactly. This is what guarantees that any change in a computed
    exponent comes from the integrator switch and nothing else.
    """
    torch.manual_seed(0)
    dt = 0.01
    for name, m in _models():
        x = torch.randn(m.N) * 0.7
        Q = torch.linalg.qr(torch.randn(m.N, 5))[0]
        x_new, Q_new = m.step_and_tangent(x, dt, Q, integrator='euler')
        assert torch.equal(x_new, m.step(x, dt)), f"{name}: state path drifted"
        assert torch.equal(Q_new, m.jacobian(x, dt) @ Q), f"{name}: tangent path drifted"


def test_rk4_step_matches_step_and_tangent_state_update():
    """The standalone RK4 step and the coupled path must produce the same state."""
    torch.manual_seed(0)
    dt = 0.01
    for name, m in _models():
        x = torch.randn(m.N) * 0.7
        Q = torch.linalg.qr(torch.randn(m.N, 3))[0]
        x_coupled, _ = m.step_and_tangent(x, dt, Q, integrator='rk4')
        assert torch.equal(x_coupled, m.step_rk4(x, dt)), f"{name}: duplicated RK4 stages disagree"


def test_unknown_integrator_raises():
    m = RandomRNN(N=10, g=1.5, seed_net=0)
    x = torch.randn(10)
    Q = torch.eye(10)
    with pytest.raises(ValueError, match="rk45|unknown integrator"):
        m.step_and_tangent(x, 0.01, Q, integrator='rk45')
    with pytest.raises(ValueError, match="unknown integrator"):
        m.step_with(x, 0.01, integrator='rk45')


def test_rk4_beats_euler_on_closed_form_linear_spectrum():
    """End-to-end in the real solver, against a spectrum known in closed form.

    For the linear model f(x) = A x, the exact exponents of a one-step map with stability
    function R are log|R(dt * mu_i)| / dt for the eigenvalues mu_i of A. The TRUE flow
    spectrum is Re(mu_i). So the bias of each integrator is computable exactly, and the
    solver's job is only to recover its own map's exponents.

    Asserts the thing actually in question: RK4's bias against the true flow spectrum is
    far smaller than Euler's at the same dt.
    """
    N, dt, T_sim = 60, 0.05, 400.0
    m = RandomLinearRNN(N=N, g=0.8, seed_net=0)
    A = m.flow_jacobian(torch.zeros(N)).double().numpy()
    mu = np.linalg.eigvals(A)
    true_spec = np.sort(mu.real)[::-1]

    R = {'euler': lambda z: 1 + z,
         'rk4': lambda z: 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24}

    bias = {}
    for integ in ('euler', 'rk4'):
        exact_map = np.sort(np.log(np.abs(R[integ](dt * mu))) / dt)[::-1]
        bias[integ] = np.max(np.abs(exact_map - true_spec))

        got = LyapunovSpectrum(m).compute(
            T_sim=T_sim, dt=dt, T_transient=20.0, n_LE=N, T_ONS=1.0,
            integrator=integ, T_ONS_warmup=20.0,
        ).spectrum
        # solver must recover ITS OWN map's spectrum; residual here is finite-horizon,
        # not discretization
        assert np.max(np.abs(got - exact_map)) < 0.02, (
            f"{integ}: solver deviates from its own map's closed form by "
            f"{np.max(np.abs(got - exact_map)):.3e}")

    assert bias['rk4'] < bias['euler'] / 100, (
        f"RK4 bias {bias['rk4']:.3e} not >=100x better than Euler {bias['euler']:.3e}")


@pytest.mark.parametrize("name,builder,expected_order", [
    ("Random", lambda N: RandomRNN(N=N, g=1.5, seed_net=0), 3.5),
    ("Barcode", lambda N: BarcodeRNN(N=N, g=7.0, seed_net=0), 1.5),
])
def test_measured_convergence_order(name, builder, expected_order):
    """Measure the order of the RK4 step against a fine reference.

    The tanh field is analytic and should recover close to order 4. The barcode field has a
    ReLU kink, and order reduction there is EXPECTED, not a bug -- this test pins the
    reduced order rather than asserting full order, so a regression that breaks the smooth
    case is still caught.
    """
    torch.set_default_dtype(torch.float64)
    try:
        N = 30
        m = builder(N)
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v):
                setattr(m, k, v.double())
        torch.manual_seed(0)
        x0 = (torch.randn(N) * 0.5).double()

        T = 0.5
        dt_ref = T / 40000
        ref = x0.clone()
        for _ in range(40000):
            ref = m.step_rk4(ref, dt_ref)

        dts, errs = [], []
        for n_steps in (100, 200, 400, 800):
            dt = T / n_steps
            x = x0.clone()
            for _ in range(n_steps):
                x = m.step_rk4(x, dt)
            dts.append(dt)
            errs.append(max((x - ref).abs().max().item(), 1e-15))

        order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        assert order > expected_order, f"{name}: measured order {order:.2f} < {expected_order}"
    finally:
        torch.set_default_dtype(torch.float32)
