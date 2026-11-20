import numpy as np
import torch
from rnn_lyapunov.models.base import RNN

class RandomLinearRNN(RNN):
    '''
    Strictly LINEAR version of RandomRNN -- dx/dt = (-x + W @ x) / tau, no tanh (or any
    other nonlinearity) at all. Same W-construction convention as RandomRNN (Gaussian,
    scaled by g/sqrt(N), no autapses), so it's the same random matrix ensemble, just without the
    nonlinearity -- exists specifically as the control case for checking whether nonlinearity is
    stabilizing/destabilizing relative to linear theory, and as a ground-truth correctness check
    for LyapunovSpectrum.compute() itself: for a genuinely linear time-invariant system, the
    Lyapunov exponents are EXACTLY the real parts of the Jacobian's eigenvalues (a closed-form
    answer, no simulation needed) -- so this model lets you compare compute()'s real QR-algorithm
    output against that closed form directly, independent of tanh/DC-construction/anything else
    model-specific.
    '''
    def __init__(self, N:int, g:float, tau:float = 1.0, seed_net:int = 42):
        super().__init__(N)
        self.g = g
        self.tau = tau
        np.random.seed(seed_net)
        W_base = np.random.randn(N, N) / np.sqrt(N)
        W = self.g * W_base
        W = W - np.diag(np.diag(W))  # Remove self-connections W_ii (autapses)
        self.W = torch.tensor(W, dtype=torch.float32)

    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        '''dx/dt = (-x + W @ x) / tau -- no nonlinearity applied to x before the W-multiply.'''
        return (-x + self.W @ x) / self.tau

    def jacobian(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        '''
        Analytic Jacobian of the Euler step -- CONSTANT (doesn't depend on x at all), since the
        dynamics are linear: Df = (1 - dt/tau)*I + (dt/tau)*W. No sech^2(x) term like
        RandomRNN has, because there's no tanh here to differentiate.
        '''
        return (1 - dt / self.tau) * torch.eye(self.N) + (dt / self.tau) * self.W

    def flow_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Analytic Jacobian of the VECTOR FIELD: J = (-I + W) / tau. Constant -- independent
        of x, since the dynamics are linear. This is what makes this model the closed-form
        reference case: the exponents of any one-step map with stability function R are
        exactly log|R(dt * mu_i)| / dt for mu_i = eig(J).
        '''
        return (-self._eye_like(x) + self.W) / self.tau
