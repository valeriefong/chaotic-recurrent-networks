import numpy as np
import torch
from typing import Callable, Optional
from rnn_lyapunov.models.base import RNN


class StateDependentLeakRNN(RNN):
    '''
    ReLU network with a state-dependent leak (divisive normalization), the dynamics of the Barcode
    model: dx/dt = (-L(x) x + W relu(x)) / tau, with L(x) = (alpha/N) * sum(relu(x)) by default.
    Weights here are Gaussian with no self-connections; BarcodeRNN draws them as in Fang et al.
    '''
    def __init__(self, N:int, g:float, tau:float = 1.0, seed_net:int = 42,
                  alpha:float =20.0, leak_fn: Optional[Callable] = None):
        super().__init__(N)
        self.g = g
        self.tau = tau
        self.alpha = alpha  # Dynamics divisive normalization strength
        np.random.seed(seed_net)
        # W has random gaussian weights with variance g**2 / N, following Sompolinsky et al. 1988
        # First sample weights from N(0,1) * 1/sqrt(N) i.e. N(0, 1/N)
        W_base = np.random.randn(N, N) / np.sqrt(N) # Without scaling by gain parameter g
        # Scale by gain parameter g
        W = self.g * W_base  
        W = W - np.diag(np.diag(W))  # Remove self-connections W_ii (autapses)
        self.W = torch.tensor(W, dtype=torch.float32)

        if leak_fn is None:
            def _default_leak_fn(x):
                return (self.alpha / self.N) * torch.sum(torch.relu(x))
            leak_fn = _default_leak_fn
        self.leak_fn = leak_fn
        
    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Explicitly compute the dynamics of the Barcode model of Fang, Lindsey et al. (2025, eLife)
        '''
        # ReLU instead of tanh, plus a state-dependent leak (divisive normalization) that keeps the
        # activity bounded
        x_relu = torch.relu(x)

        # leak acts on x (the integrated voltage-like state), not x_relu -- matches the paper's
        # dv/dt = -(alpha/Nx * sum(x))*v + Jx: the leak keeps pulling v back toward 0 even when
        # v<0 (x_relu=0), it doesn't vanish just because the unit is below threshold.
        return (- self.leak_fn(x) * x + self.W @ x_relu) / self.tau

    def jacobian(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        '''
        Analytic Jacobian of the Euler step -- closed form, no jacrev/autograd needed (same pattern
        as RandomRNN, just with an extra term from the state-dependent leak).

        f(x) = (-L(x)*x + W @ relu(x)) / tau,  L(x) = (alpha/N) * sum(relu(x))
        H = 1{x>0} (elementwise ReLU derivative, ignoring the measure-zero kink at x=0)

        Df_dynamics = (1/tau) * [ W @ diag(H) - (alpha/N) * outer(x, H) - L(x)*I ]
        Df_step = I + dt * Df_dynamics

        This assumes leak_fn is the default (alpha/N)*sum(relu(x)) -- a custom leak_fn passed to
        __init__ would need its own analytic derivative here, this formula doesn't generalize to
        an arbitrary leak_fn.
        '''
        H = (x > 0).to(x.dtype)  # shape (N,), broadcasts as diag(H) when multiplying W's columns
        L = (self.alpha / self.N) * torch.sum(torch.relu(x))
        Df_dynamics = (self.W * H - (self.alpha / self.N) * torch.outer(x, H)
                       - L * torch.eye(self.N)) / self.tau
        return torch.eye(self.N) + dt * Df_dynamics

    def flow_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Analytic Jacobian of the VECTOR FIELD (no dt) -- this is exactly the Df_dynamics
        term of `jacobian` above, which already computes it internally before wrapping it in
        I + dt*(.). Overriding this keeps RK4 off the base class's autodiff fallback, which
        measured ~5x slower per Jacobian build; RK4 needs four builds per step.

        Carries the same caveat as `jacobian`: this closed form assumes the DEFAULT
        leak_fn = (alpha/N)*sum(relu(x)). A custom leak_fn passed to __init__ has a
        different derivative, and this formula would be wrong for it without raising any error.
        '''
        H = (x > 0).to(x.dtype)
        L = (self.alpha / self.N) * torch.sum(torch.relu(x))
        return (self.W * H - (self.alpha / self.N) * torch.outer(x, H)
                - L * self._eye_like(x)) / self.tau

    '''
    step, simulate methods are inherited from RNN base class unchanged.
    '''