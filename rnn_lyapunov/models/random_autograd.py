import numpy as np
import torch
from rnn_lyapunov.models.base import RNN

class RandomAutogradRNN(RNN):
    '''
    Same model as RandomRNN (same weights for the same seed_net), but the Jacobian comes from
    automatic differentiation (torch.func.jacrev) in the RNN base class instead of a hand-derived
    formula. Slower; used in tests to verify RandomRNN's analytic Jacobian.
    '''
    def __init__(self, N:int, g:float, tau:float = 1.0, seed_net:int = 42):
        super().__init__(N)
        self.g = g
        self.tau = tau
        np.random.seed(seed_net)
        # W has random gaussian weights with variance g**2 / N, following Sompolinsky et al. 1988
        # First sample weights from N(0,1) * 1/sqrt(N) i.e. N(0, 1/N)
        W_base = np.random.randn(N, N) / np.sqrt(N) # Without scaling by gain parameter g
        # Scale by gain parameter g
        W = self.g * W_base  
        W = W - np.diag(np.diag(W))  # Remove self-connections W_ii (autapses)
        self.W = torch.tensor(W, dtype=torch.float32)

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