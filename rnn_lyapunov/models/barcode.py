import numpy as np
import torch
from rnn_lyapunov.models.base import RNN
from rnn_lyapunov.models.state_dependent_leak import StateDependentLeakRNN


class BarcodeRNN(StateDependentLeakRNN):
    '''
    Barcode model of Fang, Lindsey et al. (2025), eLife 14:RP103512, their Eq. 2, without external
    input (autonomous). Same state-dependent-leak dynamics as StateDependentLeakRNN (inherited
    unchanged) -- this class only differs in how the connectivity is drawn.

    J ~ N(mu/N, g^2/N). `mu` is the paper's fixed background inhibition ("global subtractive
    inhibition", mu = -40, never varied in the paper). `g` is the paper's sigma directly. Their
    Figure 2B uses sigma in {1, 7, 20} with mu = -40 fixed, and sigma = 7 as the reported operating
    point (coexistence of place code and barcode activity, matching Chettih et al. 2024), so g = 7
    reproduces that point.
    '''
    def __init__(self, N: int, g: float, tau: float = 1.0, seed_net: int = 42,
                 alpha: float = 20.0, leak_fn=None, mu: float = -40.0, keep_diag: bool = False):
        '''
        keep_diag: if True, keep the self-connections J_ii, as in Fang et al. (Methods: every
            entry of J ~ N(mu/N, sigma^2/N), no diagonal zeroing). Default False removes them,
            matching Random/DC, where Engelken et al. set J_ii = 0. The matrix is drawn identically
            either way, so the two settings give the SAME off-diagonal weights for a given
            seed_net and differ only on the diagonal.
        '''
        RNN.__init__(self, N)
        self.g = g
        self.tau = tau
        self.alpha = alpha
        self.mu = mu
        self.keep_diag = keep_diag
        np.random.seed(seed_net)
        W = mu / N + (g / np.sqrt(N)) * np.random.randn(N, N)
        if not keep_diag:
            W = W - np.diag(np.diag(W))  # remove self-connections (autapses), same as every other model
        self.W = torch.tensor(W, dtype=torch.float32)

        if leak_fn is None:
            def _default_leak_fn(x):
                return (self.alpha / self.N) * torch.sum(torch.relu(x))
            leak_fn = _default_leak_fn
        self.leak_fn = leak_fn
