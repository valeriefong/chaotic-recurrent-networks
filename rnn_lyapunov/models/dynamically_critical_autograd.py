import numpy as np
import torch
from rnn_lyapunov.models.base import RNN


class DynamicallyCriticalAutogradRNN(RNN):
    '''
    Modified version of Sompolinsky RNN model with random weights and nonlinear activation function tanh()
    After sampling weights, a fraction of the eigenvalues of the connectivity matrix W are shifted near imaginary axis
    (see __init__: set that fraction with critical_modes_fraction, e.g. 0.30 = 30% of eigenvalues; critical_modes is an
    absolute count of conjugate PAIRS and does not scale with N)
    These near-critical modes produce long-lived oscillatory transients

    Same model as DynamicallyCriticalRNN (same weights for the same arguments), but the Jacobian comes
    from automatic differentiation in the RNN base class. Used in tests to verify the analytic Jacobian.
    '''
    def __init__(self, N:int, g:float, tau:float = 1.0, seed_net:int = 42, critical_modes:int = 1, shift:float = 0.01,
                 critical_modes_fraction:float = None):
        '''
        How many near-critical modes, two equivalent ways to say it:

        critical_modes_fraction: FRACTION OF ALL N EIGENVALUES placed near critical, e.g. 0.30 means 30%
            of the eigenvalues: an EXTENSIVE number of near-critical modes that scales with N.
            Converted to conjugate pairs as critical_modes = max(1, round(critical_modes_fraction * N / 2)).
            Example: 0.30 at N = 1000 -> 150 pairs = 300 eigenvalues.
        critical_modes: the same thing as an absolute number of complex-conjugate PAIRS (each pair is 2
            eigenvalues). A fixed count does NOT scale with N, so the fraction of near-critical
            eigenvalues then changes with network size.

        Pass at most one of the two. self.critical_fraction_actual records the fraction of eigenvalues
        actually shifted (2 * pairs shifted / N), which can differ from the requested fraction by
        rounding at small N, or if fewer complex pairs exist than requested.
        '''
        super().__init__(N)
        if critical_modes_fraction is not None:
            if critical_modes != 1:
                raise ValueError('pass either critical_modes (pairs) or critical_modes_fraction '
                                 '(fraction of eigenvalues), not both')
            critical_modes = max(1, round(critical_modes_fraction * N / 2))
        self.g = g
        self.tau = tau
        self.critical_modes = critical_modes
        self.critical_modes_fraction = critical_modes_fraction
        self.shift = shift

        rng = np.random.RandomState(seed_net)
        # W has random gaussian weights with variance g**2 / N, following Sompolinsky et al. 1988
        W = 1 / np.sqrt(N) * rng.randn(N, N)
        W = W - np.diag(np.diag(W))  # Remove self-connections W_ii (autapses)

        eigenvalues, eigenvectors = np.linalg.eig(W)

        # dynamics() below reuses the same form as RandomRNN/RandomAutogradRNN: f(x) = -x/tau + W @ tanh(x)/tau,
        # so the Jacobian at the fixed point x=0 is (-I + W)/tau -- the leak's -I is added on top of W every
        # time dynamics() runs. So "near-critical" here means shifting eigenvalues of W itself toward Re=1
        # (not Re=0), since that's where Re(eig(-I + W)) crosses zero.

        # Select the critical_modes complex-conjugate pairs closest to the instability boundary
        # (largest real part) and pull their real part just below critical, keeping their imaginary
        # part (oscillation frequency) -- this is what gives the long-lived oscillatory transients.
        # Slicing a fixed-size window off a real-part-sorted list can split a genuine conjugate pair
        # (e.g. a real eigenvalue or a tie landing on the boundary), so pairs are matched explicitly
        # by conjugate value instead of by adjacency in the sorted order.
        orig_eigenvalues = eigenvalues.copy()
        pos_imag_idx = np.where(np.imag(eigenvalues) > 1e-10)[0]  # one representative per conjugate pair
        pos_imag_idx = pos_imag_idx[np.argsort(-np.real(eigenvalues[pos_imag_idx]))]
        selected = pos_imag_idx[:critical_modes]
        self.critical_fraction_actual = 2 * len(selected) / N

        for idx in selected:
            partner = np.argmin(np.abs(eigenvalues - np.conj(orig_eigenvalues[idx])))
            eigenvalues[idx] = (1 - shift) + 1j * np.imag(orig_eigenvalues[idx])
            eigenvalues[partner] = (1 - shift) - 1j * np.imag(orig_eigenvalues[idx])

        # Pull back any other eigenvalue that still exceeds the critical threshold so only the
        # intentionally selected modes end up sitting near-critical
        unstable = np.real(eigenvalues) >= 1
        eigenvalues[unstable] = (1 - 10 * shift) + 1j * np.imag(eigenvalues[unstable])

        W_shifted = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
        self.W = self.g * torch.tensor(np.real(W_shifted), dtype=torch.float32)

    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Explicitly compute the dynamics f(x) = (-x + W @ tanh(x)) / tau 
        for Sompolinsky et al. 1988 model
        '''
        return (-x + self.W @ torch.tanh(x)) / self.tau
    
    '''
    No need to specify jacobian method because this model class uses torch function jacrev to compute jacobian via autograd, inherited from RNN base class
    Note that step, simulate methods are also inherited from RNN base class
    '''