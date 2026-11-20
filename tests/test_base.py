import pytest 
import torch
from rnn_lyapunov.models.base import RNN

def test_rnn_init():
    with pytest.raises(TypeError):
        RNN(N=10)  
        # should raise TypeError because RNN cannot be instantiated directly

def test_rnn_subclass():
    class TestRNN(RNN):
        def __init__(self, N):
            super().__init__(N)

    with pytest.raises(TypeError):
        TestRNN(N=10)

def test_simulate_shape():
    class ConstantRNN(RNN):
        def dynamics(self, x):
            return torch.zeros_like(x)
    
    rnn = ConstantRNN(N=5)
    x0 = torch.zeros(5)
    traj = rnn.simulate(x0, T=1.0, dt=0.1)
    steps = int(1.0 / 0.1)  # 10 steps
    # T/dt steps + the initial state = 11 rows
    assert traj.shape == ((steps + 1), 5)