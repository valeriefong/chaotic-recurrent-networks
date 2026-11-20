import torch
from rnn_lyapunov.models.random import RandomRNN

def test_weight_matrix():
    rnn = RandomRNN(N=20, g=10, seed_net=42)
    assert rnn.W.shape == (20, 20)

def test_reproducibility():
    rnn1 = RandomRNN(N=12, g=2.0, seed_net=1)
    rnn2 = RandomRNN(N=12, g=2.0, seed_net=1)
    assert torch.allclose(rnn1.W, rnn2.W)

def test_no_autapses():
    # testing that self connections have been removed, following paper
    rnn = RandomRNN(N=20, g=10, seed_net=42)
    assert torch.all(torch.diag(rnn.W) == 0)

def test_step_formula():
    rnn = RandomRNN(N=10, g=5, seed_net=42)
    x = torch.randn(10)
    dt = 0.01
    step_result = rnn.step(x, dt)
    expected_result = x * (1-dt) + dt * rnn.W @ torch.tanh(x)
    assert torch.allclose(step_result, expected_result)

def test_jacobian_shape():
    rnn = RandomRNN(N=10, g=5, seed_net=42)
    x = torch.randn(10)
    J = rnn.jacobian(x, dt = 0.01)
    assert J.shape == (10, 10)

def test_jacobian_linearizes_step():
    """
    J @ dx approximately equals step(x + dx) - step(x) for small perturbations dx
    """
    rnn = RandomRNN(N=10, g=5, seed_net=42)
    x = torch.randn(10)
    dx = torch.randn(10) * 1e-5  # small perturbation
    dt = 1e-3
    J = rnn.jacobian(x, dt)
    step_result = rnn.step(x, dt)
    step_result_perturbed = rnn.step(x + dx, dt)
    delta = step_result_perturbed - step_result
    linearized_dynamics = J @ dx  # should equal step_result when linearized around x
    assert torch.allclose(delta, linearized_dynamics, atol=1e-6)