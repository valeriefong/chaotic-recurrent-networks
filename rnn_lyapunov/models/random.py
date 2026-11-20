import numpy as np
import torch
from rnn_lyapunov.models.base import RNN

class RandomRNN(RNN):
    '''
    Sompolinsky RNN model with randomly sampled weights and nonlinear activation function tanh(). Dynamics are modulated by a gain parameter g scaling overall connection strength.
    In the large network limit, transition to chaotic dynamics occurs at g = 1.0
    Connectivity matrix W has entries W_ij ~ N(0, g**2 / N)
    RandomRNN implements the analytic Jacobian for this model; RandomAutogradRNN is the same model with the Jacobian computed by automatic differentiation (torch.func.jacrev), used in tests to verify it.
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
    
    def jacobian(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        '''
        Override the base class Jacobian using jacrev with the analytic Jacobian for this model.
        See Engelken et al. (2023), equation (3).

        Analytic Jacobian of the Euler step is given by:
        Df = (1 - dt/tau)*I + W * diag(dt/tau / cosh(x)**2)
        Note that broadcasting W * v gives W[i,j] * v[j] = W @ diag(v)
        '''
        sech2_x = dt / torch.cosh(x) ** 2  # d/dt[tanh(x)] = diag(sec**2(x)) so store as shape (N,)
        # Df will be shape (N, N)
        Df = (1 - dt/self.tau) * torch.eye(self.N) + self.W * sech2_x / self.tau  
        return Df 

    def flow_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Analytic Jacobian of the VECTOR FIELD: J = (-I + W @ diag(sech^2(x))) / tau.
        Identical to `jacobian` above with dt factored out (Df = I + dt*J). Overriding this
        keeps RK4 off the base class's autodiff fallback, which measured ~5x slower per
        Jacobian build; RK4 needs four builds per step, so the fallback dominated cost.
        '''
        sech2_x = 1.0 / torch.cosh(x) ** 2
        return (-self._eye_like(x) + self.W * sech2_x) / self.tau

    '''
    Note that step, simulate methods are inherited from RNN base class, so we do not specify it again here
    '''